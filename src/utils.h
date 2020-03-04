#ifndef MIX_UTILS_H
#define MIX_UTILS_H
#include "arma-wrap.h"

/* C++ version of the dtrmv BLAS function */
inline void inplace_tri_mat_mult(arma::vec &x, arma::mat const &trimat){
  arma::uword const n = trimat.n_cols;

  for(unsigned j = n; j-- > 0;){
    double tmp(0.);
    for(unsigned i = 0; i <= j; ++i)
      tmp += trimat.at(i, j) * x[i];
    x[j] = tmp;
  }
}

/* d vech(chol(X)) / d vech(X). See mathoverflow.net/a/232129/134083
 *
 * Args:
 *    X: symmetric positive definite matrix.
 *    upper: logical for whether vech denotes the upper triangular part.
 */
arma::mat dchol(arma::mat const&, bool const upper = false);

#endif
