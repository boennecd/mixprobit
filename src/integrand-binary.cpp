#include "integrand-binary.h"
#include "lapack.h"

namespace integrand {
double mix_binary::operator()(double const *par, bool const ret_log) const {
  memcpy(par_vec.begin(), par, sizeof(double) * n_par);
  double out(0);
  for(unsigned i = 0; i < n; ++i){
    double const lp = eta[i] + arma::dot(Z.unsafe_col(i), par_vec);
    /* essentially all of the computation time is spend on the next line */
    out += y[i] > 0 ? R::pnorm5(lp, 0, 1, 1L, 1L) :
                      R::pnorm5(lp, 0, 1, 0L, 1L);
  }

  if(ret_log)
    return out;
  return std::exp(out);
}

arma::vec mix_binary::gr(double const *par) const {
  memcpy(par_vec.begin(), par, sizeof(double) * n_par);

  arma::vec gr(n_par, arma::fill::zeros);
  for(unsigned i = 0; i < n; ++i){
    double const lp = eta[i] + arma::dot(Z.unsafe_col(i), par_vec),
                 s = y[i] > 0 ? 1 : -1;

    double const f = R::dnorm4(s * lp, 0, 1,     1L),
                 F = R::pnorm5(s * lp, 0, 1, 1L, 1L);
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
    double const lp = eta[i] + arma::dot(Z.unsafe_col(i), par_vec);

    double const lpu = y[i] > 0 ? lp : -lp,
                 fl  = R::dnorm4(lpu, 0, 1,     1L),
                 f   = std::exp(fl),
                 Fl  = R::pnorm5(lpu, 0, 1, 1L, 1L),
                 F   = std::exp(Fl),
                 fac = - std::exp(fl - 2 * Fl) * (lpu * F + f);

    dsyr_call(&U, &n_pari, &fac, Z.colptr(i), &ONE, He.memptr(), &n_pari);
  }

  return arma::symmatu(He);
}

void mix_binary::Jacobian(double const *par, arma::vec &jac) const {
  memcpy(par_vec.begin(), par, sizeof(double) * n_par);
  assert(X);
  assert(jac.n_elem == get_n_jac());

  jac.zeros();
  double &integrand = jac[0];
  arma::vec fix_part(jac.memptr() + 1L, X->n_rows, false);
  for(unsigned i = 0; i < n; ++i){
    double const lp = eta[i] + arma::dot(Z.unsafe_col(i), par_vec),
               pnrm = y[i] > 0 ? R::pnorm5(lp, 0, 1, 1L, 1L) :
                                 R::pnorm5(lp, 0, 1, 0L, 1L),
               dnrm =            R::dnorm4(lp, 0, 1, 1L),
               fac  = y[i] > 0 ?  std::exp(dnrm - pnrm) :
                                 -std::exp(dnrm - pnrm);

    integrand += pnrm;
    fix_part  += fac * X->unsafe_col(i);
  }

  integrand = std::exp(integrand);
  fix_part *= integrand;
}
}
