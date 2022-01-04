#ifndef MIX_UTILS_H
#define MIX_UTILS_H
#include "arma-wrap.h"
#include <memory.h>

#include <R_ext/RS.h> // F77_NAME F77_CALL

extern "C" {
  void F77_NAME(dtpsv)
  (const char * /* uplo */, const char * /* trans */, const char * /* diag */,
   const int * /* n */, const double * /* ap */, double * /* x */,
   const int* /* incx */, size_t, size_t, size_t);
}

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

/** Computes LL^T where L is a lower triangular matrix. The argument is a
 a vector with the non-zero elements in column major order. The diagonal
 entries are on the log scale. The method computes both L and LL^T.  */
inline void get_pd_mat(double const *theta, arma::mat &L, arma::mat &res){
  unsigned const dim{L.n_rows};
  L.zeros();
  for(unsigned j = 0; j < dim; ++j){
    L.at(j, j) = std::exp(*theta++);
    for(unsigned i = j + 1; i < dim; ++i)
      L.at(i, j) = *theta++;
  }

  res = L * L.t();
}

struct dpd_mat {
  /**
   * return the required memory to get the derivative as part of the chain
   * rule  */
  static size_t n_wmem(unsigned const dim){
    return dim * dim;
  }

  /**
   * computes the derivative w.r.t. theta as part of the chain rule. That is,
   * the derivatives of f(LL^T) where d/dX f(X) evaluated at X = LL^T
   * is supplied.
   */
  static void get(arma::mat const &L, double * __restrict__ res,
                  double const * derivs, double * __restrict__ wk_mem){
    unsigned const dim{L.n_rows};
    arma::mat D(const_cast<double*>(derivs), dim, dim, false);
    arma::mat jac(wk_mem, dim, dim, false);
    jac = D * L;

    double * __restrict__ r = res;
    for(unsigned j = 0; j < dim; ++j){
      *r++ = 2 * L.at(j, j) * jac.at(j, j);
      for(unsigned i = j + 1; i < dim; ++i)
        *r++ = 2 * jac.at(i, j);
    }
  }
};

/* d vech(chol(X)) / d vech(X). See mathoverflow.net/a/232129/134083
 *
 * Args:
 *    X: symmetric positive definite matrix.
 *    upper: logical for whether vech denotes the upper triangular part.
 */
arma::mat dchol(arma::mat const&, bool const upper = false);

/* Given a symmetric positive define matrices Sigma and K, let
 *
 *   H = K + Sigma^(-1)
 *
 * Then given the gradient with respect to H^-1, the function applies the chain
 * rule and computes the derivatives with respect to Sigma. These is given by
 *
 *   vec(Sigma^-1H^-1vec^-1(<gradient>)H^-1Sigma^-1)
 */
arma::mat dcond_vcov(arma::mat const &H, arma::mat const &dH_inv,
                     arma::mat const &Sigma);

/* Given a symmetric positive definite matrix Sigma, vector v, matrix Z,
 * and matrix L, this function computes the derivative w.r.t. Sigma for the
 * expression
 *
 *  K = I + Z^T.Sigma.Z
 *  x = L.K^(-1).v
 *
 * given the Jacobian of g(x) w.r.t. x. The result is
 *
 *   -vec(Z.K^(-1).L^T.vec^(-1)(<gradient>).v^TK^(-1).Z^T)
 */
arma::mat dcond_vcov_rev
  (arma::mat const &K, arma::mat const &Z, arma::mat const &L,
   arma::vec const &v, arma::vec const &d_x);

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
