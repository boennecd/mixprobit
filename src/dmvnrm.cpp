#include "dmvnrm.h"
#include "utils.h"

static double const log2pi = std::log(2.0 * M_PI);

double dmvnrm(arma::vec x, arma::mat const &cov_chol_inv,
              bool const logd){
  arma::uword const xdim = cov_chol_inv.n_cols;

  double const rootisum = arma::sum(log(cov_chol_inv.diag())),
              constants = -(double)xdim/2.0 * log2pi,
            other_terms = rootisum + constants;

  /* TODO: use backsolve instead... */
  inplace_tri_mat_mult(x, cov_chol_inv);
  double const out = other_terms - 0.5 * arma::dot(x, x);

  if(logd)
    return out;
  return std::exp(out);
}
