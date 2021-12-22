#include "integrand.h"
#include "utils.h"

namespace integrand {

/*
 * The class works with integrals of the type
 *
 *   int phi^(K)(u; 0, Sigma)Phi^(n)(eta + Z.u) du
 *     = int phi^(K)(u)Phi^(n)(eta + Z.C^T.u) du
 *
 * where C^TC = Sigma and we applied a change of variable with u = C^Tx.
 * The eta is given by eta = X.beta.
 *
 * To compute the derivatives, we note that the derivatives w.r.t. Sigma has an
 * integrand for the form
 *
 *   1/2 Sigma^(-1) (u.u^T - Sigma) Sigma^(-1)
 *     * phi^(K)(u; 0, Sigma)Phi^(n)(eta + Z.u)
 *   = 1/2 C^(-1)(u.u^T - I)C^(-T)
 *     * phi^(K)(u)Phi^(n)(eta + Z.C^T.u)
 *
 * For beta, we simply need the derivatives wr.r.t. eta from which the chain
 * rule can easily be applied.
 *
 * When n < K, we can instead work with
 *
 *   int phi^(n)(u; 0, Z.Sigma.Z^T)Phi^(n)(eta + u) der u
 *
 * To compute the derivatives w.r.t. Sigma, we first compute the derivatives
 * w.r.t. Psi = Z.Sigma.Z^T as before. Then we apply the chain rule to get
 * the derivatives w.r.t. Sigma.
 */
class mix_binary final : public base_integrand {
  arma::ivec const &y;
  arma::vec  const &eta;
  arma::mat  const &Zorg;
  arma::mat Z;
  std::size_t const n = y.n_elem, n_par = Z.n_rows;
  arma::mat const * const X;
  arma::mat const &Sigma;
  // TODO: would be nice to avoid working memory here?
  mutable arma::vec par_vec = arma::vec(n_par);

  std::unique_ptr<double[]> obj_mem{new double[n_par * (n_par + 1)]};
  /// the non-zero elements of the Cholesky decomposition of Sigma
  double * const Sigma_chol{obj_mem.get()};
  /// the upper triangular part of the inverse of sigma
  double * const Sigma_inv{Sigma_chol + (n_par * (n_par + 1)) / 2};

  bool is_dim_reduced() const {
    return Z.n_rows < Zorg.n_rows;
  }

public:
  mix_binary(arma::ivec const &y, arma::vec const &eta,
             arma::mat const &Zin, arma::mat const &Sigma,
             arma::mat const *X = nullptr);

  double operator()
  (double const*, bool const ret_log = false) const;

  arma::vec gr(double const*) const;
  arma::mat Hessian(double const*) const;

  std::size_t get_n_par() const {
    return n_par;
  };

  /* returns the integrand and the derivatives w.r.t. a fixed effect
   * coefficient vector and the upper triangular part of the covariance
   * matrix. You need to call Jacobian_post_process to get final result. */
  void Jacobian(double const *par, arma::vec &jac) const;
  void Jacobian_post_process(arma::vec &jac) const;

  std::size_t get_n_jac() const {
    arma::uword const dim_d_sigma = (n_par * (n_par + 1L)) / 2L;
    assert(X);
    return 1L + X->n_rows + dim_d_sigma;
  }
};
} // namespace integrand
