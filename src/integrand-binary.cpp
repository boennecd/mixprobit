#include "integrand-binary.h"

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
