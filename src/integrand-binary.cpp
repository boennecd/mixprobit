#include "integrand-binary.h"
#include "lapack.h"
#include "utils.h"
#include "pnorm.h"
#include "dnorm.h"

namespace integrand {
double mix_binary::operator()(double const *par, bool const ret_log) const {
  memcpy(par_vec.begin(), par, sizeof(double) * n_par);
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
  memcpy(par_vec.begin(), par, sizeof(double) * n_par);

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
  memcpy(par_vec.begin(), par, sizeof(double) * n_par);
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
    throw std::runtime_error("mix_binary::Jacobian not implemented");

  using arma::uword;
  memcpy(par_vec.begin(), par, sizeof(double) * n_par);
  assert(X);
  uword const vcov_dim = (n_par * (n_par + 1L)) / 2L;
  assert(d_Sigma_chol.n_rows == vcov_dim);
  assert(d_Sigma_chol.n_cols == vcov_dim);
  assert(jac.n_elem == get_n_jac());

  jac.zeros();
  double &integrand = jac[0];
  arma::vec fix_part(jac.memptr() + 1L, X->n_rows, false);
  arma::vec vcov_part(jac.memptr() + 1L + X->n_rows, vcov_dim, false);

  wk_mem.zeros();
  for(unsigned i = 0; i < n; ++i){
    double const lp = eta[i] + colvecdot(Z, i, par_vec),
               pnrm = y[i] > 0 ? pnorm_std(lp, 1L, 1L) :
                                 pnorm_std(lp, 0L, 1L),
               dnrm =            dnorm_std(lp, 1L),
               fac  = y[i] > 0 ?  std::exp(dnrm - pnrm) :
                                 -std::exp(dnrm - pnrm);

    integrand += pnrm;
    fix_part  += fac * X->col(i);
    wk_mem    += fac * Zorg.col(i);
  }

  integrand  = std::exp(integrand);
  fix_part  *= integrand;
  wk_mem    *= integrand;

  /* TODO: we can do this afterwards */
  uword ij(0);
  for(uword j = 0; j < n_par; ++j)
    for(uword i = 0; i <= j; ++i, ++ij){
      double const ele = par_vec.at(i) * wk_mem.at(j),
                   *sp = d_Sigma_chol.memptr() + ij;

      for(uword k = 0; k <= ij; ++k, sp += vcov_dim)
        vcov_part.at(k) += ele * *sp;
    }
}
}
