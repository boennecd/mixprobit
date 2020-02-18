#include "integrand-binary.h"

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

  for(unsigned i = 0; i < n; ++i){
    double const lp = eta[i] + arma::dot(Z.unsafe_col(i), par_vec);

    double const lpu = y[i] > 0 ? lp : -lp,
                 fl  = R::dnorm4(lpu, 0, 1,     1L),
                 f   = std::exp(fl),
                 Fl  = R::pnorm5(lpu, 0, 1, 1L, 1L),
                 F   = std::exp(Fl),
                 fac = - std::exp(fl - 2 * Fl) * (lpu * F + f);

    /* TODO: replace with rank-one update */
    He += (fac * Z.col(i)) * Z.col(i).t();
  }

  return He;
}
}
