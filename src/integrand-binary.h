#include "integrand.h"

namespace integrand {

class mix_binary final : public base_integrand {
  arma::ivec const &y;
  arma::vec  const &eta;
  arma::mat  const Z;
  std::size_t const n = y.n_elem, n_par = Z.n_rows;
  mutable arma::vec par_vec = arma::vec(n_par);
  arma::mat const * const X;

  static arma::mat set_Z(arma::mat const &Zin, arma::mat const &S){
    std::size_t const k = Zin.n_rows;
    assert(S.n_rows == k);
    assert(S.n_cols == k);

    return arma::chol(S) * Zin;
  }

public:
  mix_binary(arma::ivec const &y, arma::vec const &eta,
             arma::mat const &Zin, arma::mat const &Sigma,
             arma::mat const *X = nullptr):
  y(y), eta(eta), Z(set_Z(Zin, Sigma)), X(X) {
    assert(eta.n_elem == n);
    assert(Z.n_cols   == n);
    assert(n_par > 1L);
    if(X)
      assert(X->n_cols == n);
  }

  double operator()
  (double const*, bool const ret_log = false) const;

  arma::vec gr(double const*) const;
  arma::mat Hessian(double const*) const;

  std::size_t get_n_par() const {
    return n_par;
  };

  /* returns the integrand and the derivatives w.r.t. a fixed effect
   * coefficient vector and the upper triangular part of the covariance
   * matrix */
  void Jacobian(double const*, arma::vec&) const;

  std::size_t get_n_jac() const {
    assert(X);
    return X->n_rows + 1L;
  }
};
} // namespace integrand
