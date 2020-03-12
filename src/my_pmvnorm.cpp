#include "my_pmvnorm.h"
#include "welfords.h"

namespace {
struct get_var_order_output {
  /* upper bounds and Cholesky decompostion after sorting */
  arma::vec b, L;
};

get_var_order_output get_var_order
  (arma::vec b_in, arma::mat Sigma){
  using arma::uword;
  uword const &p = b_in.n_elem;

  get_var_order_output output;
  arma::vec &b = output.b,
            &L = output.L,
            us = b_in,
         denom = arma::diagvec(Sigma);
  b.resize(p);
  L.resize((p * (p + 1L)) / 2L);

  /* util functions */
  auto trunc_mean = [&](double const x){
    double const phi_Phi =
      std::exp(R::dnorm4(x, 0, 1, 1L) - R::pnorm5(x, 0, 1, 1L, 1L));
    return -phi_Phi;
  };
  auto swap_vec = [&](arma::vec &x, uword const i, uword const j){
    double const dum = x[i];
    x[i] = x[j];
    x[j] = dum;
  };
  auto get_trimat_idx = [&](uword const row, uword const col){
    assert(row <= col);
    return (col * (col + 1L)) / 2L + row;
  };
  auto get_diag_idx = [&](uword const ele){
    return get_trimat_idx(ele, ele);
  };
  auto vfunc = [&](double const x){
    double const phi_Phi =
      std::exp(R::dnorm4(x, 0, 1, 1L) - R::pnorm5(x, 0, 1, 1L, 1L));
    return -x * phi_Phi - phi_Phi * phi_Phi;
  };

  /* find variable order, compute upper bounds, and the Cholesky
   * decomposition */
  for(uword var_idx = 0; var_idx < p - 1; ++var_idx){
    {
      uword idx(var_idx);
      double max_crit = vfunc(us[idx] / std::sqrt(denom[idx]));
      for(uword i = idx + 1L; i < p; ++i){
        double const compet = vfunc(us[i] / std::sqrt(denom[i]));
        if(compet < max_crit){
          max_crit = compet;
          idx = i;
        }
      }

      /* swap */
      {
        Sigma.swap_rows(var_idx, idx);
        Sigma.swap_cols(var_idx, idx);

        swap_vec(us   , var_idx, idx);
        swap_vec(denom, var_idx, idx);
        swap_vec(b_in , var_idx, idx);

        for(unsigned row = 0; row < var_idx; ++row)
          std::swap(L[get_trimat_idx(row, var_idx)],
                    L[get_trimat_idx(row, idx    )]);
      }
    }

    /* update L and b */
    uword ele = get_diag_idx(var_idx);
    double const diag_ele = std::sqrt(denom[var_idx]);
    L[ele]     = diag_ele;
    b[var_idx] = b_in[var_idx];
    double const yi = trunc_mean(us[var_idx] / diag_ele);

    for(uword i = var_idx + 1L; i < p; ++i){
      ele += i;

      L[ele] = Sigma.at(i, var_idx);
      for(uword m = 0; m < var_idx; ++m)
        L[ele] -= L[get_trimat_idx(m, var_idx)] *
                  L[get_trimat_idx(m, i      )];
      L[ele] /= diag_ele;

      /* update un-scaled upper bounds and squared denominators */
      us   [i] -= L[ele] * yi;
      denom[i] -= L[ele] * L[ele];
    }
  }

  /* handle the last element */
  uword const var_idx = p - 1L,
                  ele = get_diag_idx(var_idx);
  double const diag_ele = std::sqrt(denom[var_idx]);
  L[ele]     = diag_ele;
  b[var_idx] = b_in[var_idx];

  return output;
}
} // namespace

my_pmvnorm_output my_pmvnorm
  (arma::vec const &mean_in, arma::mat const &sigma_in,
   unsigned const nsim, double const eps){
  arma::uword const p = mean_in.n_elem;
  if(sigma_in.n_cols != p or sigma_in.n_rows != p)
    throw std::invalid_argument("invalid sigma");

  /* re-scale and re-order. I think that Genz is doing something like
   * described in the following article:
   *   Gibson GJ, Glasbey CA, Elston DA (1994) Monte Carlo evaluation of multivariate normal integrals and sensitivity to variate ordering. In: Dimov IT,
   Sendov B, Vassilevski PS (eds) Advances in Numerical Methods and Applications, World Scientific Publishing, River Edge, pp 120–126
   *
   * See section 4.1.3 of Genz Monograph.
   */
  auto const order_obj = get_var_order(-mean_in, sigma_in);

  arma::vec const mean = -order_obj.b;
  arma::mat const sigma_chol = std::move(order_obj.L);

  welfords out;
  arma::vec draw(p - 1L), u(p - 1L);
  unsigned i;
  unsigned const min_run(p * 10L);
  constexpr double const alpha(2.575829);
  for(i = 0; i < nsim;){
    unsigned const i_max = i + min_run;
    for(; i < i_max; ++i){
      auto const func = [&](arma::vec const &u){
        double w(1.);
        unsigned j;
        double const *s = sigma_chol.begin();
        for(j = 0; j < p - 1; ++j){
          double b(-mean[j]);
          for(unsigned k = 0; k < j; ++k)
            b -= *s++ * draw[k];
          b /= *s++;

          double const qb = R::pnorm5(b, 0, 1, 1L, 0L);
          w       *= qb;
          draw[j]  = R::qnorm5(qb * u[j], 0, 1, 1L, 0L);
        }

        double b(-mean[j]);
        for(unsigned k = 0; k < j; ++k)
          b -= *s++ * draw[k];
        b /= *s++;

        double const qb = R::pnorm5(b, 0, 1, 1L, 0L);
        w *= qb;

        return w;
      };

      u.for_each([](double &val) { return val = unif_rand(); });
      double new_term = func(u);
      u.for_each([](double &val) { return val = 1 - val; });
      new_term += func(u);

      out += new_term / 2.;
    }

    if(alpha * std::sqrt(out.var() / (double)i) < eps)
      break;
  }

  return { out.mean(), std::sqrt(out.var() / (double)i), i };
}
