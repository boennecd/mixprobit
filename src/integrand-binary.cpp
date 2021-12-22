#include "integrand-binary.h"
#include "lapack.h"
#include "utils.h"
#include "pnorm.h"
#include "dnorm.h"

namespace integrand {
mix_binary::mix_binary(
  arma::ivec const &y, arma::vec const &eta, arma::mat const &Zin,
  arma::mat const &Sigma, arma::mat const *X):
y(y), eta(eta), Zorg(Zin),
Z(([](arma::mat const &Zin, arma::mat const &S) -> arma::mat {
#ifndef NDEBUG
  std::size_t const k = Zin.n_rows;
#endif
  assert(S.n_rows == k);
  assert(S.n_cols == k);

  if(Zin.n_rows <= Zin.n_cols)
    return arma::chol(S) * Zin;

  arma::mat new_vcov{Zin.t() * S * Zin};
  return arma::chol(new_vcov);
})(Zin, Sigma)), X(X),
Sigma(Sigma){
  assert(eta.n_elem == n);
  assert(Z.n_cols   == n);
  assert(n_par > 1L);
  assert(!X || X->n_cols == n);

  if(!is_dim_reduced()){
    // compute the Cholesky decomposition
    arma::mat C{arma::chol(Sigma)};
    {
      double * Sigma_chol_ij{Sigma_chol};
      for(arma::uword j = 0; j < C.n_cols; ++j, Sigma_chol_ij += j)
        std::copy(C.colptr(j), C.colptr(j) + j + 1, Sigma_chol_ij);
    }

    // store the inverse of Sigma
    arma::inv_sympd(C, Sigma);
    double * Sigma_inv_ij{Sigma_inv};
    for(arma::uword j = 0; j < C.n_cols; ++j, Sigma_inv_ij += j)
      std::copy(C.colptr(j), C.colptr(j) + j + 1, Sigma_inv_ij);
  } // TODO: handle the else!
}

double mix_binary::operator()(double const *par, bool const ret_log) const {
  std::copy(par, par + n_par, par_vec.begin());
  double out(0);
  for(unsigned i = 0; i < n; ++i){
    double const lp = eta[i] + colvecdot(Z, i, par_vec);
    /* essentially all of the computation time is spend on the next line */
    out += y[i] > 0 ? pnorm_std(lp, 1L, 1L) :
                      pnorm_std(lp, 0L, 1L);
  }

  if(ret_log)
    return out;
  return std::exp(out);
}

arma::vec mix_binary::gr(double const *par) const {
  std::copy(par, par + n_par, par_vec.begin());
  arma::vec gr(n_par, arma::fill::zeros);
  for(unsigned i = 0; i < n; ++i){
    double const lp = eta[i] + colvecdot(Z, i, par_vec),
                 s = y[i] > 0 ? 1 : -1;

    double const f = dnorm_std(s * lp, 1L),
                 F = pnorm_std(s * lp, 1L, 1L);
    gr += s * std::exp(f - F)  * Z.col(i);
  }

  return gr;
}

arma::mat mix_binary::Hessian(double const *par) const {
  std::copy(par, par + n_par, par_vec.begin());
  arma::mat He(n_par, n_par, arma::fill::zeros);
  constexpr int const ONE(1L);
  constexpr char const U('U');
  int const n_pari = n_par;

  for(unsigned i = 0; i < n; ++i){
    double const lp = eta[i] + colvecdot(Z, i, par_vec);

    double const lpu = y[i] > 0 ? lp : -lp,
                 fl  = dnorm_std(lpu, 1L),
                 f   = std::exp(fl),
                 Fl  = pnorm_std(lpu, 1L, 1L),
                 F   = std::exp(Fl),
                 fac = - std::exp(fl - 2 * Fl) * (lpu * F + f);

    dsyr_call(&U, &n_pari, &fac, Z.colptr(i), &ONE, He.memptr(), &n_pari);
  }

  return arma::symmatu(He);
}

void mix_binary::Jacobian(double const *par, arma::vec &jac) const {
  if(is_dim_reduced())
    // TODO: have to implement this
    throw std::runtime_error("mix_binary::Jacobian not implemented");

  using arma::uword;
  std::copy(par, par + n_par, par_vec.begin());
  assert(X);
  uword const vcov_dim = (n_par * (n_par + 1L)) / 2L;
  assert(d_Sigma_chol.n_rows == vcov_dim);
  assert(d_Sigma_chol.n_cols == vcov_dim);
  assert(jac.n_elem == get_n_jac());

  jac.zeros();
  double &integrand = jac[0];
  arma::vec fix_part(jac.memptr() + 1L, X->n_rows, false);
  arma::vec vcov_part(jac.memptr() + 1L + X->n_rows, vcov_dim, false);

  for(unsigned i = 0; i < n; ++i){
    double const lp = eta[i] + colvecdot(Z, i, par_vec),
               pnrm = y[i] > 0 ? pnorm_std(lp, 1L, 1L)
                               : pnorm_std(lp, 0L, 1L),
               dnrm =            dnorm_std(lp, 1L),
               fac  = y[i] > 0 ?  std::exp(dnrm - pnrm)
                               : -std::exp(dnrm - pnrm);

    integrand += pnrm;
    fix_part  += fac * X->col(i);
  }

  integrand = std::exp(integrand);
  fix_part *= integrand;

  /* if C^TC = Sigma then we need to compute
   *
   *   [weight] * 1/2 C^(-1)(x.x^T - I)C^(-1).
   *
   */

  int n_parse = n_par;
  constexpr int incx{1};
  F77_CALL(dtpsv)("U", "N", "N", &n_parse, Sigma_chol, par_vec.memptr(),
                  &incx, 1, 1, 1);

  double * vc_part_i{vcov_part.begin()};
  double const * sigma_inv_i{Sigma_inv};
  for(uword j = 0; j < n_par; ++j){
    for(uword i = 0; i < j; ++i)
      *vc_part_i++ = integrand * (par_vec[i] * par_vec[j] - *sigma_inv_i++);
    *vc_part_i++ = .5 * integrand * (par_vec[j] * par_vec[j] - *sigma_inv_i++);
  }
}

void mix_binary::Jacobian_post_process(arma::vec &jac) const {
  if(is_dim_reduced())
    // TODO: have to implement this
    throw std::runtime_error("mix_binary::Jacobian not implemented");
}
}
