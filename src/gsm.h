#ifndef H_MIXED_GSM
#define H_MIXED_GSM
#include "arma-wrap.h"

/**
 * The class is used to compute
 *
 *   log int phi^(K)(u, h, H^(-1))Phi^(n)(-X^c.beta - Z^c.u) d u
 *
 * with
 *
 *   H = Z^(oT).Z^o + Sigma^(-1)
 *   h = H^(-1)Z^(oT)(-X^o.beta)
 *
 * with given design matrices and model parameters Sigma and beta. One approach
 * that is used to re-write the integral as
 *
 * log int phi^(K)(u)Phi^(n)(-X^c.beta - Z^c.h - Z^c.C^T.u) d u
 *
 * with C^TC = H^-1 is the Cholesky decomposition. Thus, given the derivatives
 * w.r.t.
 *
 *   omega = -X^c.beta - Z^c.h
 *
 * and H from which we can compute the rest of the derivatives.
 *
 * The design matrices are stored as their transpose to be consistent with the
 * rest of the package.
 */
class gsm_cens_term {
  arma::mat const &Zo, &Zc;
  arma::mat const &Xo, &Xc;
  arma::vec const &beta;
  arma::mat const &Sigma;

  arma::uword n_obs{Xo.n_cols},
              n_cen{Xc.n_cols},
              n_rng{Zc.n_rows},
            n_fixef{beta.n_elem};

  arma::ivec const y_pass = arma::ivec(n_cen, arma::fill::ones);

public:
  struct gsm_cens_output {
    double log_like;
    int inform;
  };

  gsm_cens_term
  (arma::mat const &Zo, arma::mat const &Zc, arma::mat const &Xo,
   arma::mat const &Xc, arma::vec const &beta, arma::mat const&Sigma);

  /// evaluates the log of the integral
  gsm_cens_output func
    (int const maxpts, int const key, double const abseps,
     double const releps) const;

  /// evaluates the log of the integral and the gradient of it
  gsm_cens_output gr
    (arma::vec &gr,  int const maxpts, int const key, double const abseps,
     double const releps) const;
};

/// approximates the log marginal likelihood term for a mixed GSM cluster
class mixed_gsm_cluster {
  /**
   * the design matrix for the random effects. Both for the observed and the
   * censored.
   */
  arma::mat Zo, Zc;
  arma::mat Zo_inner{Zo * Zo.t()};
  /**
   * the design matrix for the fixed effects. Both for the observed and the
   * censored.
   */
  arma::mat Xo, Xc;
  /**
   * the design matrix for the time-varying fixed effects. Both for the observed
   * and the censored.
   */
  arma::mat Xo_prime, Xc_prime;
  /// the observed times for the censored and uncensored individuals
  arma::vec yo, yc;


  /// the number of censored observations
  arma::uword n_cens() const {
    return Xc.n_cols;
  }
  /// the number of observed observations
  arma::uword n_obs() const {
    return Xc.n_cols;
  }
  /// the total number of observations
  arma::uword n_total() const {
    return n_cens() + n_obs();
  }

public:
  mixed_gsm_cluster
  (arma::mat const &X, arma::mat const &X_prime, arma::mat const &Z,
   arma::vec const &y, arma::vec const &event);

  /// computes the log marginal likelihood.
  /// TODO: need more arguments for other methods?
  double operator()
    (arma::vec const &beta, arma::mat const &Sigma,
     bool const is_adaptive = false) const;

  /**
   * computes the gradient, adds it to gr, and returns the log marginal
   * likelihood.
   */
  double grad
    (arma::vec &gr, arma::vec const &beta, arma::mat const &Sigma,
     bool const is_adaptive = false);
};

#endif
