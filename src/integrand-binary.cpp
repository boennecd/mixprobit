#include "integrand-binary.h"

double mix_binary::operator()(double const *par) const {
  memcpy(par_vec.begin(), par, sizeof(double) * n_par);
  double out(0);
  for(unsigned i = 0; i < n; ++i){
    double const lp = eta[i] + arma::dot(Z.col(i), par_vec);
    out += y[i] > 0 ? R::pnorm5(lp, 0, 1, 1L, 1L) :
                      R::pnorm5(lp, 0, 1, 0L, 1L);
  }
  out = std::exp(out);

  return out;
}
