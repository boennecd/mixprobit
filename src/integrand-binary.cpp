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
n_par(std::min<std::size_t>(Zin.n_rows, n)), X(X),
Sigma(Sigma) {
  assert(eta.n_elem == n);
  assert(Zin.n_cols   == n);
  assert(n_par > 1L);
  assert(!X || X->n_cols == n);

  if(!is_dim_reduced){
    // compute the Cholesky decomposition
    arma::mat C{arma::chol(Sigma)};
    {
      double * Sigma_chol_ij{Sigma_chol};
      for(arma::uword j = 0; j < C.n_cols; ++j, Sigma_chol_ij += j)
        std::copy(C.colptr(j), C.colptr(j) + j + 1, Sigma_chol_ij);
    }

    Z = C * Zin;

    // store the inverse of Sigma
    arma::inv_sympd(C, Sigma);
    double * Sigma_inv_ij{Sigma_inv};
    for(arma::uword j = 0; j < C.n_cols; ++j, Sigma_inv_ij += j)
      std::copy(C.colptr(j), C.colptr(j) + j + 1, Sigma_inv_ij);

  } else {
    // the covariance matrix used in the integration
    arma::mat new_vcov{Zin.t() * Sigma * Zin};

    // compute the Cholesky decomposition
    arma::mat C(arma::chol(new_vcov));
    Z = C;
    arma::inplace_trans(C);

    // store the matrix we need
    ZC_inv = arma::solve(arma::trimatl(C), Zin.t());
    arma::mat decom = ZC_inv.t() * ZC_inv;
    arma::inplace_trans(ZC_inv);

    double * Sigma_inv_ij{Sigma_inv};
    for(arma::uword j = 0; j < decom.n_cols; ++j, Sigma_inv_ij += j)
      std::copy(decom.colptr(j), decom.colptr(j) + j + 1, Sigma_inv_ij);
  }
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
  using arma::uword;
  std::copy(par, par + n_par, par_vec.begin());
  assert(jac.n_elem == get_n_jac());

  jac.zeros();
  double &integrand = jac[0];
  uword const n_fixef = has_fixef() ? X->n_rows : n;
  arma::vec fix_part(jac.memptr() + 1L, n_fixef, false);
  double * const vcov_part{jac.memptr() + 1L + n_fixef};

  for(unsigned i = 0; i < n; ++i){
    double const lp = eta[i] + colvecdot(Z, i, par_vec),
               pnrm = y[i] > 0 ? pnorm_std(lp, 1L, 1L)
                               : pnorm_std(lp, 0L, 1L),
               dnrm =            dnorm_std(lp, 1L),
               fac  = y[i] > 0 ?  std::exp(dnrm - pnrm)
                               : -std::exp(dnrm - pnrm);

    integrand += pnrm;
    if(has_fixef())
      fix_part += fac * X->col(i);
    else
      fix_part[i] = fac;
  }

  integrand = std::exp(integrand);
  fix_part *= integrand;

  if(is_dim_reduced){
    /* we need to change the derivatives w.r.t. the covariance matrix as these
     * are currently for Psi = Z.Sigma.Z^T. To this, we need to compute
     *
     *   [weight] * 1/2 *  Z^T.C^(-1).(x.x^T - I).C^(-T).Z
     *
     * and then only the upper part
     */
    wk_mem = ZC_inv * par_vec;

    double * vc_part_i{vcov_part};
    double const * sigma_inv_i{Sigma_inv};
    for(uword j = 0; j < full_dim; ++j){
      for(uword i = 0; i < j; ++i)
        *vc_part_i++ = integrand * (wk_mem[i] * wk_mem[j] - *sigma_inv_i++);
      *vc_part_i++ = .5 * integrand * (wk_mem[j] * wk_mem[j] - *sigma_inv_i++);
    }

  } else {
    /* if C^TC = Sigma then we need to compute
     *
     *   [weight] * 1/2 C^(-1)(x.x^T - I)C^(-T).
     *
     */
    int n_parse = n_par;
    constexpr int incx{1};
    F77_CALL(dtpsv)("U", "N", "N", &n_parse, Sigma_chol, par_vec.memptr(),
                    &incx, 1, 1, 1);

    double * vc_part_i{vcov_part};
    double const * sigma_inv_i{Sigma_inv};
    for(uword j = 0; j < n_par; ++j){
      for(uword i = 0; i < j; ++i)
        *vc_part_i++ = integrand * (par_vec[i] * par_vec[j] - *sigma_inv_i++);
      *vc_part_i++ = .5 * integrand * (par_vec[j] * par_vec[j] - *sigma_inv_i++);
    }
  }
}
}
