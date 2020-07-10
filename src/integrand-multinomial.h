#ifndef INTEGRAND_MULTINOMIAL_H
#define INTEGRAND_MULTINOMIAL_H

#include "integrand.h"
#include "gaus-Hermite.h"

namespace integrand {
class multinomial_mode_helper;

/**
 Let

 \begin{align*}
 l(a, \vec u) &= \log\phi(a) + \sum_{k' \neq k}
   \log\Phi\left(a - (\vec\beta_k - \vec\beta_{k'})^\top\vec x_i
   - (\vec z_{ik} - \vec z_{ik'})^\top\vec u\right) \\
  &= \log\phi(a) + \sum_{k' \neq k}
   \log\Phi\left(a + \eta_{k'} - \tilde z_{k'}^\top\vec u\right)
  \end{align*}

 Then this function evaluates int \exp l(a, \vec u) da for different values
 of \vec u. There are n - 1 k' values.
 */
class multinomial final : public base_integrand {
  friend class multinomial_helper;

  /* K x (n - 1) matrix with Z^\top */
  arma::mat const Z;
  /* K offset values */
  arma::vec const eta;

  GaussHermite::HermiteData const &nodes;
  arma::vec const &w = nodes.w,
              &w_log = nodes.w_log,
                  &x = nodes.x;
  size_t const n_par = Z.n_rows,
               n_alt = Z.n_cols,
               n_nodes = x.n_elem;
  mutable arma::vec par_vec = arma::vec(n_par),
    /* objects needed to hold intermediaries */
                        wk1 = arma::vec(n_alt),
                        wk2 = arma::vec(n_alt),
                         lp = arma::vec(n_alt);
  bool const is_adaptive;

  static arma::mat set_Z(arma::mat const &Zin, arma::mat const &S){
#ifndef NDEBUG
    std::size_t const k = Zin.n_rows;
#endif
    assert(S.n_rows == k);
    assert(S.n_cols == k);

    return arma::chol(S) * Zin;
  }

public:
  multinomial(arma::mat const &Z, arma::vec const &eta,
              arma::mat const &Sigma, size_t const n_nodes,
              bool const is_adaptive):
  Z(set_Z(Z, Sigma)), eta(eta),
  nodes(GaussHermite::gaussHermiteDataCached(n_nodes)),
  is_adaptive(is_adaptive) {
    if(eta.n_elem != Z.n_cols)
      throw std::invalid_argument(
          "multinomial::multinomial(): Invalid eta");
    else if(Z.n_rows < 1L)
      throw std::invalid_argument(
          "multinomial::multinomial(): Invalid Z");
    else if(Sigma.n_rows != n_par or Sigma.n_cols != n_par)
      throw std::invalid_argument(
          "multinomial::multinomial(): Invalid Sigma");
  }

  double operator()
  (double const*, bool const ret_log = false) const;

  arma::vec gr(double const*) const;
  arma::mat Hessian(double const*) const;

  std::size_t get_n_par() const {
    return n_par;
  };

  /** function which finds a mode for the multinomial class and to
    *  compute the second order derivatives to perform adaptive GHQ. */
  struct mode_res {
    double const location = std::numeric_limits<double>::quiet_NaN(),
                    scale = std::numeric_limits<double>::quiet_NaN();
    bool found_mode = false;

    mode_res(double const location, double const scale,
             bool const found_mode):
      location(location), scale(scale), found_mode(found_mode) { }
    mode_res() = default;
  };

  friend class multinomial_mode_helper;
  mode_res find_mode(double const*) const;
};
} // namespace integrand

#endif

