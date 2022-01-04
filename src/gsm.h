#ifndef H_MIXED_GSM
#define H_MIXED_GSM
#include "arma-wrap.h"
#include <memory.h>
#include <string.h>

enum class gsm_approx_method {
  spherical_radial,
  adaptive_spherical_radial,
  cdf_approach
};

inline gsm_approx_method string_to_gsm_approx_method
  (std::string const &which){
  if(which == "spherical_radial")
    return gsm_approx_method::spherical_radial;
  else if(which == "adaptive_spherical_radial")
    return gsm_approx_method::adaptive_spherical_radial;
  else if(which == "cdf_approach")
    return gsm_approx_method::cdf_approach;

  throw std::invalid_argument("not matching method");
  return gsm_approx_method::spherical_radial;
}

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
 * log int phi^(K)(u; 0, H^(-1))Phi^(n)(-X^c.beta - Z^c.h - Z^c.u) d u
 *
 * Thus, given the derivatives w.r.t.
 *
 *   omega = -X^c.beta - Z^c.h
 *
 * and H we can compute the rest of the derivatives.
 *
 * The design matrices are stored as their transpose to be consistent with the
 * rest of the package.
 */
class gsm_cens_term {
  arma::mat const &Zo, &Zc;
  arma::mat const &Xo, &Xc;
  arma::vec const &beta;
  arma::mat const &Sigma;

  arma::uword const n_obs, n_cen, n_rng, n_fixef;

  arma::ivec const y_pass = arma::ivec(n_cen, arma::fill::ones);

public:
  struct gsm_cens_output {
    double log_like;
    int inform, intvls;
  };

  gsm_cens_term
  (arma::mat const &Zo, arma::mat const &Zc, arma::mat const &Xo,
   arma::mat const &Xc, arma::vec const &beta, arma::mat const &Sigma);

  /// evaluates the log of the integral
  gsm_cens_output func
    (int const maxpts, int const key, double const abseps,
     double const releps, gsm_approx_method const method_use) const;

  /// evaluates the log of the integral and the gradient of it
  gsm_cens_output gr
    (arma::vec &gr, int const maxpts, int const key, double const abseps,
     double const releps, gsm_approx_method const method_use) const;

private:
  gsm_cens_output func_spherical_radial
    (int const maxpts, int const key, double const abseps,
     double const releps, bool const use_adaptive) const;

  gsm_cens_output gr_spherical_radial
    (arma::vec &gr, int const maxpts, int const key, double const abseps,
     double const releps, bool const use_adaptive) const;
};

/**
 * The class the normal density
 *
 *   - d / 2 * log(2 * pi) - 2^-1 * log|K| - 2^-1 * v^T.K^(-1).v
 *
 * where
 *
 *   v = -X.beta
 *   K = Z.Sigma.Z^T + I
 *
 * The Jacobian w.r.t. beta and Sigma are given
 *
 *   v.K^(-1).X
 *   (-vec(Z^T.K^(-1).Z) + v^T.K^-1.Z (x) v^T.K^-1.Z) / 2
 *
 * The design matrices are stored as their transpose to be consistent with the
 * rest of the package.
 */
class gsm_normal_term {
  arma::mat const &X, &Z, &Sigma;
  arma::vec const &beta;

  arma::uword const n_obs, n_rng, n_fixef;

  /// working memory
  std::unique_ptr<double[]> wk_mem{new double[(n_obs * (n_obs + 1)) / 2]};
  /// the upper triangular matrix for the Cholesky decomposition of Sigma
  double * const K_chol{wk_mem.get()};

  /// the determinant of K
  double log_K_deter;

public:
  gsm_normal_term
  (arma::mat const &X, arma::mat const &Z, arma::mat const &Sigma,
   arma::vec const &beta);

  /// computes the log density
  double func() const;

  /// computes the log density and the gradient
  double gr(arma::vec &gr) const;
};

/// approximates the log marginal likelihood term for a mixed GSM cluster
class mixed_gsm_cluster {
  /**
   * the design matrix for the random effects. Both for the observed and the
   * censored.
   */
  arma::mat Zo, Zc;
  /**
   * the design matrix for the fixed effects. Both for the observed and the
   * censored.
   */
  arma::mat Xo, Xc;
  /// the design matrix for the derivative of the fixed effects.
  arma::mat Xo_prime;
  /// the observed times for the censored and uncensored individuals
  arma::vec yo, yc;

public:
  /// the number of censored observations
  arma::uword n_cens() const {
    return Xc.n_cols;
  }
  /// the number of observed observations
  arma::uword n_obs() const {
    return Xo.n_cols;
  }
  /// the total number of observations
  arma::uword n_total() const {
    return n_cens() + n_obs();
  }
  /// the number of random effects
  arma::uword n_rng() const {
    return Zo.n_rows;
  }
  /// the number of fixed effects
  arma::uword n_fixef() const {
    return Xo.n_rows;
  }

  mixed_gsm_cluster
  (arma::mat const &X, arma::mat const &X_prime, arma::mat const &Z,
   arma::vec const &y, arma::vec const &event);

  struct mixed_gsm_cluster_res {
    double log_like;
    int inform;
  };

  /// computes the log marginal likelihood.
  mixed_gsm_cluster_res operator()
    (arma::vec const &beta, arma::mat const &Sigma, int const maxpts,
     int const key, double const abseps, double const releps,
     gsm_approx_method const method_use) const;

  /**
   * computes the gradient, setting the result gr, and returns the log marginal
   * likelihood.
   */
  mixed_gsm_cluster_res grad
    (arma::vec &gr, arma::vec const &beta, arma::mat const &Sigma,
     int const maxpts, int const key, double const abseps,
     double const releps, gsm_approx_method const method_use) const ;
};

#endif
