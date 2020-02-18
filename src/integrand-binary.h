#include "integrand.h"

namespace integrand {

class mix_binary final : public base_integrand {
  arma::ivec const &y;
  arma::vec  const &eta;
  arma::mat  const Z;
  std::size_t const n = y.n_elem, n_par = Z.n_rows;
  mutable arma::vec par_vec = arma::vec(n_par);

  static arma::mat set_Z(arma::mat const &Zin, arma::mat const &S){
    std::size_t const k = Zin.n_rows;
    assert(S.n_rows == k);
    assert(S.n_cols == k);

    return arma::chol(S) * Zin;
  }

public:
  mix_binary(arma::ivec const &y, arma::vec const &eta,
             arma::mat const &Zin, arma::mat const &Sigma):
  y(y), eta(eta), Z(set_Z(Zin, Sigma)) {
    assert(eta.n_elem == n);
    assert(Z.n_cols   == n);
    assert(n_par > 1L);
  }

  double operator()
  (double const*, bool const ret_log = false) const;

  arma::vec gr(double const*) const;
  arma::mat Hessian(double const*) const;

  std::size_t get_n_par() const {
    return n_par;
  };
};
} // namespace integrand
