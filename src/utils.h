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

/* compute the dot product with a given column and a vector */
inline double colvecdot(arma::mat const &X, size_t const i,
                        arma::vec const &x) {
  double out(0.);
  int const n = X.n_rows;
  double const *d1 = X.colptr(i),
               *d2 = x.memptr();
  for(int i = 0; i < n; ++i, ++d1, ++d2)
    out += *d1 * *d2;
  return out;
}

inline double log_exp_add(double const t1, double const t2){
  double const largest = std::max(t1, t2);
  return largest + log(exp(t1 - largest) + exp(t2 - largest));
}

/* d vech(chol(X)) / d vech(X). See mathoverflow.net/a/232129/134083
 *
 * Args:
 *    X: symmetric positive definite matrix.
 *    upper: logical for whether vech denotes the upper triangular part.
 */
arma::mat dchol(arma::mat const&, bool const upper = false);

/* very simple importance sampler with so-called location and scaled
 * balanced antithetic variables for multivariate normal distributed
 * variables */
template<typename I>
class antithetic {
  I const &integrand;
public:
  antithetic(I const &integrand): integrand(integrand) { }

  double operator()(arma::vec &par_vec) const {
    double new_term = integrand(par_vec.begin());
    par_vec *= -1;
    new_term += integrand(par_vec.begin());

    double const old_scale = ([&]{
      double out(0.);
      for(auto x : par_vec)
        out += x * x;
      return out;
    })();

    double const p_val = R::pchisq(old_scale, (double)par_vec.n_elem,
                                   1L, 0L),
             new_scale = R::qchisq(1 - p_val, (double)par_vec.n_elem,
                                   1L, 0L);

    par_vec *= std::sqrt(new_scale / old_scale);
    new_term += integrand(par_vec.begin());
    par_vec *= -1;
    new_term += integrand(par_vec.begin());
    return new_term / 4.;
  }
};

#endif
