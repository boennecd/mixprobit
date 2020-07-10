#ifndef INTEGRAND_MULTINOMIAL_H
#define INTEGRAND_MULTINOMIAL_H

#include "integrand.h"
#include "gaus-Hermite.h"

#define DEFAULT_N_NODES 8L
#define DEFAULT_IS_ADAPTIVE true

namespace integrand {
class multinomial_mode_helper;

/**
 Approximation of

 \begin{align*}
 h(\vec u) &= \int\phi(a)\prod_{k = 1}^c
 \log\Phi\left(a + \eta_k - \vec z_k^\top\vec u\right)da
 \end{align*}

 and the derivatives of \log h(\vec u).
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
  /* objects needed to hold intermediaries */
  mutable arma::vec par_vec, wk1, wk2, lp;
  bool const is_adaptive;

  /** values used to cache old mode results */
  mutable double mode_old  = 0,
                 delta_old = 5.16;

  static arma::mat set_Z(arma::mat const &Zin, arma::mat const &S){
#ifndef NDEBUG
    std::size_t const k = Zin.n_rows;
#endif
    assert(S.n_rows == k);
    assert(S.n_cols == k);

    return S * Zin;
  }

public:
  /** Args:
   *    Z: p x c matrix with covariates.
   *    eta: c vector with offsets.
   *    Sigma: scale matrix for Z.
   *    n_nodes: number of quadrature nodes.
   *    is_adaptive: whether or not to use adaptive GHQ.
   *    wk_mem: pointer to p + 3 * c doubles to use as working memory.
   */
  multinomial(arma::mat const &Z, arma::vec const &eta,
              arma::mat const &Sigma_sqrt, double * wk_mem,
              size_t const n_nodes = DEFAULT_N_NODES,
              bool const is_adaptive = DEFAULT_IS_ADAPTIVE):
  Z(set_Z(Z, Sigma_sqrt)), eta(eta),
  nodes(GaussHermite::gaussHermiteDataCached(n_nodes)),
  par_vec(wk_mem                     , n_par, false),
  wk1    (wk_mem + n_par             , n_alt, false),
  wk2    (wk_mem + n_par +      n_alt, n_alt, false),
  lp     (wk_mem + n_par + 2L * n_alt, n_alt, false),
  is_adaptive(is_adaptive) {
    if(eta.n_elem != Z.n_cols)
      throw std::invalid_argument(
          "multinomial::multinomial(): Invalid eta");
    else if(Z.n_rows < 1L)
      throw std::invalid_argument(
          "multinomial::multinomial(): Invalid Z");
    else if(Sigma_sqrt.n_rows != n_par or Sigma_sqrt.n_cols != n_par)
      throw std::invalid_argument(
          "multinomial::multinomial(): Invalid Sigma");
  }

  double operator()
  (double const*, bool const ret_log = false) const;

  arma::vec gr(double const*) const;
  arma::mat Hessian(double const*) const;

  std::size_t get_n_par() const {
    return n_par;
  }

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

/**
 Approximation of

 \begin{align*}
 h(\vec u) &= \prod_{i = 1}^n\int\phi(a)\prod_{k = 1}^c
 \log\Phi\left(a + \eta_{ik} - \vec z_{ik}^\top\vec u\right)da
 \end{align*}

 and the derivatives of \log h(\vec u). The member functions are __not__
 thread safe. That is, multiple instances of the class can be used in
 parallel but the member function of a particular instance cannot.
 */
class multinomial_group final : public base_integrand {
  std::vector<multinomial> factors;
  size_t const n_alt, n_par;
  std::unique_ptr<double[]> wk_mem =
    std::unique_ptr<double[]>(new double[n_par + 3L * n_alt]);

public:
  multinomial_group(
    size_t const n_alt, arma::mat const &Z, arma::vec const &eta,
    arma::mat const &Sigma, size_t const n_nodes = DEFAULT_N_NODES,
    bool const is_adaptive = DEFAULT_IS_ADAPTIVE):
  n_alt(n_alt), n_par(Z.n_rows) {
    if(eta.n_elem != Z.n_cols)
      throw std::invalid_argument(
          "multinomial_group::multinomial_group(): invalid eta");
    else if(Z.n_cols % n_alt != 0L)
      throw std::invalid_argument(
          "multinomial_group::multinomial_group(): invalid Z");

    size_t const n_obs = Z.n_cols / n_alt;
    arma::mat const Sigma_chol = arma::chol(Sigma);
    factors.reserve(n_obs);

    arma::mat Z_i;
    arma::vec eta_i;
    for(size_t i = 0; i < Z.n_cols; i += n_alt){
      Z_i = Z.cols(i, i + n_alt - 1L);
      eta_i = eta.subvec(i, i + n_alt - 1L);

      factors.emplace_back(Z_i, eta_i, Sigma_chol, wk_mem.get(), n_nodes,
                           is_adaptive);
    }
  }

  double operator()
  (double const*, bool const ret_log = false) const;

  arma::vec gr(double const*) const;
  arma::mat Hessian(double const*) const;

  std::size_t get_n_par() const {
    return n_par;
  }
};

} // namespace integrand

#undef DEFAULT_N_NODES
#undef DEFAULT_IS_ADAPTIVE

#endif
