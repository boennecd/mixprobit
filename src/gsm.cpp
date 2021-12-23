#include "gsm.h"
#include "ranrth-wrapper.h"
#include "integrand-binary.h"
#include "utils.h"

using namespace integrand;
using namespace ranrth_aprx;

gsm_cens_term::gsm_cens_term
  (arma::mat const &Zo, arma::mat const &Zc, arma::mat const &Xo,
   arma::mat const &Xc, arma::vec const &beta, arma::mat const &Sigma):
  Zo{Zo}, Zc{Zc}, Xo{Xo}, Xc{Xc}, beta{beta}, Sigma{Sigma} {
    // check all the dimensions
    assert(Zo.n_cols == n_obs);
    assert(Zc.n_cols == n_cen);

    assert(Sigma.n_cols == n_rng);
    assert(Sigma.n_rows == n_rng);

    assert(Xo.n_rows == n_fixef);
    assert(Xc.n_rows == n_fixef);
  }

gsm_cens_term::gsm_cens_output gsm_cens_term::func
  (int const maxpts, int const key, double const abseps,
   double const releps) const {
  if(n_cen < 1)
    return {0, 0};

  arma::vec eta = n_fixef > 0 ? -(Xc.t() * beta)
                              : arma::vec(n_cen, arma::fill::zeros);

  if(n_obs < 1){
    mix_binary bin(y_pass, eta, Zc, Sigma);
    mvn<mix_binary> m(bin);
    set_integrand(std::unique_ptr<base_integrand>(
        new adaptive<mvn<mix_binary > > (m, true)));

    auto res = integral_arpx(maxpts, key, abseps, releps);
    return { std::log(res.value), res.inform };
  }

  arma::mat H = Zo * Zo.t() + arma::inv_sympd(Sigma);
  arma::mat const H_inv = arma::inv_sympd(H);
  eta += Zc.t() * arma::solve(H, Zo * (Xo.t() * beta));

  arma::chol(H, H_inv, "lower");
  arma::mat const Z_pass = -H * Zc;

  mix_binary bin(y_pass, eta, Z_pass, H_inv);
  mvn<mix_binary> m(bin);
  set_integrand(std::unique_ptr<base_integrand>(
      new adaptive<mvn<mix_binary > > (m, true)));
  auto res = integral_arpx(maxpts, key, abseps, releps);
  return { std::log(res.value), res.inform };
}

gsm_cens_term::gsm_cens_output gsm_cens_term::gr
  (arma::vec &gr, int const maxpts, int const key, double const abseps,
   double const releps) const {
  arma::uword const vcov_dim{(Sigma.n_cols * (Sigma.n_cols + 1)) / 2};
  gr.zeros(beta.size() + vcov_dim);
  if(n_cen < 1)
    return {0, 0};

  arma::vec eta = n_fixef > 0 ? -(Xc.t() * beta)
                              : arma::vec(n_cen, arma::fill::zeros);

  arma::vec d_beta(gr.begin(), n_fixef, false),
            d_Sig(d_beta.end(), vcov_dim, false);

  if(n_obs < 1){
    mix_binary bin(y_pass, eta, Zc, Sigma);
    mvn<mix_binary> m(bin);
    set_integrand(std::unique_ptr<base_integrand>(
        new adaptive<mvn<mix_binary > > (m, true)));

    auto res = jac_arpx(maxpts, key, abseps, releps);
    double const int_val{res.value[0]};

    res.value /= int_val; // account for the log(f)
    arma::vec d_eta(res.value.begin() + 1, n_cen, false),
             d_vcov(d_eta.end(), vcov_dim, false);

    if(n_fixef > 0)
      d_beta -= Xc * d_eta;
    d_Sig += d_vcov;

    return { std::log(int_val), res.inform };
  }

  arma::mat const H = Zo * Zo.t() + arma::inv_sympd(Sigma);
  arma::vec const eta_rhs = Zo * (Xo.t() * beta);
  // TODO: perhaps smarter to make a decomposition of H as we do solve again
  //       later?
  arma::mat const H_inv = arma::inv_sympd(H);
  eta += Zc.t() * arma::solve(H, eta_rhs);

  arma::mat Z_pass = -arma::chol(H_inv, "lower") * Zc;

  mix_binary bin(y_pass, eta, Z_pass, H_inv);
  mvn<mix_binary> m(bin);
  set_integrand(std::unique_ptr<base_integrand>(
      new adaptive<mvn<mix_binary > > (m, true)));
  auto res = jac_arpx(maxpts, key, abseps, releps);

  double const int_val{res.value[0]};

  res.value /= int_val; // account for the log(f)
  arma::vec d_eta(res.value.begin() + 1, n_cen, false),
           d_vcov(d_eta.end(), vcov_dim, false);

  // handle the fixed effects and then the covariance matrix parameters
  arma::vec const Zc_d_eta = n_fixef > 0 ? Zc * d_eta : arma::vec();
  if(n_fixef > 0)
    d_beta = Xo * Zo.t() * solve(H, Zc_d_eta) - Xc * d_eta;

  /* application of the chain rule in the reverse direction. First we fill in
   * the whole derivative w.r.t. H^(-1). Here, we have to account for the terms
   * from nabla_eta as eta is given by VH^(-1)v. Thus, the terms we need to add
   * is V^T.nabla_eta^T.v^T.
   */
  arma::mat dH_inv(n_rng, n_rng);
  {
    double const * d_vcov_ij{d_vcov.begin()};
    if(n_fixef > 0)
      for(arma::uword j = 0; j < n_rng; ++j, ++d_vcov_ij){
        for(arma::uword i = 0; i < j; ++i, ++d_vcov_ij){
          dH_inv(i, j) = *d_vcov_ij / 2 + Zc_d_eta[i] * eta_rhs[j];
          dH_inv(j, i) = *d_vcov_ij / 2 + Zc_d_eta[j] * eta_rhs[i];
        }
        dH_inv(j, j) = *d_vcov_ij + Zc_d_eta[j] * eta_rhs[j];
      }
    else
      for(arma::uword j = 0; j < n_rng; ++j, ++d_vcov_ij){
        for(arma::uword i = 0; i < j; ++i, ++d_vcov_ij){
          dH_inv(i, j) = *d_vcov_ij / 2;
          dH_inv(j, i) = *d_vcov_ij / 2;
        }
        dH_inv(j, j) = *d_vcov_ij;
      }
  }
  arma::mat dSigma_one = dcond_vcov(H, dH_inv, Sigma);

  {
    double * d_Sig_ij{d_Sig.begin()};
    for(arma::uword j = 0; j < n_rng; ++j, ++d_Sig_ij){
      for(arma::uword i = 0; i < j; ++i, ++d_Sig_ij)
        *d_Sig_ij = dSigma_one(i, j) + dSigma_one(j, i);
      *d_Sig_ij = dSigma_one(j, j);
    }
  }

  return { std::log(int_val), res.inform };
}
