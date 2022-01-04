#include "gsm.h"
#include "ranrth-wrapper.h"
#include "integrand-binary.h"
#include "utils.h"
#include <stdexcept>

using namespace integrand;
using namespace ranrth_aprx;

gsm_cens_term::gsm_cens_term
  (arma::mat const &Zo, arma::mat const &Zc, arma::mat const &Xo,
   arma::mat const &Xc, arma::vec const &beta, arma::mat const &Sigma):
  Zo{Zo}, Zc{Zc}, Xo{Xo}, Xc{Xc}, beta{beta}, Sigma{Sigma},
  n_obs{Xo.n_cols}, n_cen{Xc.n_cols}, n_rng{Zc.n_rows}, n_fixef{beta.n_elem} {
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
   double const releps, bool const use_adaptive) const {
  if(n_cen < 1)
    return {0, 0, 0};

  arma::vec eta = n_fixef > 0 ? -(Xc.t() * beta)
                              : arma::vec(n_cen, arma::fill::zeros);

  if(n_obs < 1){
    auto res = ([&]{
      if(!use_adaptive){
        set_integrand(std::unique_ptr<base_integrand>(
            new mix_binary(y_pass, eta, Zc, Sigma)));
        return integral_arpx(maxpts, key, abseps, releps);

      }

      mix_binary bin(y_pass, eta, Zc, Sigma);
      mvn<mix_binary> m(bin);
      set_integrand(std::unique_ptr<base_integrand>(
          new adaptive<mvn<mix_binary > > (m, true)));

      return integral_arpx(maxpts, key, abseps, releps);
    })();

    return { std::log(res.value), res.inform, res.inivls };
  }

  arma::mat const H = Zo * Zo.t() + arma::inv_sympd(Sigma);
  arma::vec const eta_rhs = Zo * (Xo.t() * beta);
  arma::mat const H_inv = arma::inv_sympd(H);
  eta += Zc.t() * arma::solve(H, eta_rhs);

  auto res = ([&]{
    if(!use_adaptive){
      set_integrand(std::unique_ptr<base_integrand>(
          new mix_binary(y_pass, eta, Zc, H_inv)));
      return integral_arpx(maxpts, key, abseps, releps);

    }

    mix_binary bin(y_pass, eta, Zc, H_inv);
    mvn<mix_binary> m(bin);
    set_integrand(std::unique_ptr<base_integrand>(
        new adaptive<mvn<mix_binary > > (m, true)));

    return integral_arpx(maxpts, key, abseps, releps);
  })();

  return { std::log(res.value), res.inform, res.inivls };
}

gsm_cens_term::gsm_cens_output gsm_cens_term::gr
  (arma::vec &gr, int const maxpts, int const key, double const abseps,
   double const releps, bool const use_adaptive) const {
  arma::uword const vcov_dim{(Sigma.n_cols * (Sigma.n_cols + 1)) / 2};
  gr.zeros(beta.size() + vcov_dim);
  if(n_cen < 1)
    return {0, 0, 0};

  arma::vec eta = n_fixef > 0 ? -(Xc.t() * beta)
                              : arma::vec(n_cen, arma::fill::zeros);

  arma::vec d_beta(gr.begin(), n_fixef, false),
             d_Sig(d_beta.end(), vcov_dim, false);

  if(n_obs < 1){
    auto res = ([&]{
      if(!use_adaptive){
        set_integrand(std::unique_ptr<base_integrand>(
            new mix_binary(y_pass, eta, Zc, Sigma)));
        return jac_arpx(maxpts, key, abseps, releps);

      }

      mix_binary bin(y_pass, eta, Zc, Sigma);
      mvn<mix_binary> m(bin);
      set_integrand(std::unique_ptr<base_integrand>(
          new adaptive<mvn<mix_binary > > (m, true)));

      return jac_arpx(maxpts, key, abseps, releps);
    })();

    double const int_val{res.value[0]};

    res.value /= int_val; // account for the log(f)
    arma::vec d_eta(res.value.begin() + 1, n_cen, false),
             d_vcov(d_eta.end(), vcov_dim, false);

    if(n_fixef > 0)
      d_beta -= Xc * d_eta;
    d_Sig += d_vcov;

    return { std::log(int_val), res.inform, res.inivls };
  }

  arma::mat const H = Zo * Zo.t() + arma::inv_sympd(Sigma);
  arma::vec const eta_rhs = Zo * (Xo.t() * beta);
  arma::mat const H_inv = arma::inv_sympd(H);
  eta += Zc.t() * arma::solve(H, eta_rhs);

  auto res = ([&]{
    if(!use_adaptive){
      set_integrand(std::unique_ptr<base_integrand>(
          new mix_binary(y_pass, eta, Zc, H_inv)));
      return jac_arpx(maxpts, key, abseps, releps);

    }

    mix_binary bin(y_pass, eta, Zc, H_inv);
    mvn<mix_binary> m(bin);
    set_integrand(std::unique_ptr<base_integrand>(
        new adaptive<mvn<mix_binary > > (m, true)));

    return jac_arpx(maxpts, key, abseps, releps);
  })();

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

  return { std::log(int_val), res.inform, res.inivls };
}

gsm_normal_term::gsm_normal_term
  (arma::mat const &X, arma::mat const &Z, arma::mat const &Sigma,
   arma::vec const &beta):
  X{X}, Z{Z}, Sigma{Sigma}, beta{beta}, n_obs{Z.n_cols}, n_rng(Sigma.n_cols),
  n_fixef{beta.n_elem} {
    assert(X.n_rows == n_fixef);
    assert(X.n_cols == n_obs);
    assert(Z.n_rows == n_rng);
    assert(Sigma.n_rows == n_rng);

    arma::mat K = Z.t() * Sigma * Z;
    K.diag() += 1;

    arma::mat K_chol_full = arma::chol(K);
    double * K_cp_to{K_chol};
    for(arma::uword j = 0; j < n_obs; ++j)
      for(arma::uword i = 0; i <= j; ++i)
        *K_cp_to++ = K_chol_full(i, j);

    log_K_deter = 0;
    for(arma::uword i = 0; i < n_obs; ++i)
      log_K_deter += 2 * std::log(K_chol_full(i, i));
  }

double gsm_normal_term::func() const {
  if(n_fixef == 0)
    return 0;

  arma::vec v = -X.t() * beta;
  int n = n_obs, incx = 1;
  F77_CALL(dtpsv)
    ("U", "T", "N", &n, K_chol, v.memptr(), &incx, 1, 1, 1);

  double out = -static_cast<double>(n_obs) * log(2 * M_PI);
  out -= log_K_deter;
  for(double vi : v)
    out -= vi * vi;

  return out / 2;
}

double gsm_normal_term::gr(arma::vec &gr) const {
  arma::uword const vcov_dim{(n_rng * (n_rng + 1)) / 2};
  gr.zeros(beta.size() + vcov_dim);
  if(n_fixef == 0)
    return 0;

  arma::vec v = -X.t() * beta;
  int n = n_obs, incx = 1;
  F77_CALL(dtpsv)
    ("U", "T", "N", &n, K_chol, v.memptr(), &incx, 1, 1, 1);

  double out = -static_cast<double>(n_obs) * log(2 * M_PI);
  out -= log_K_deter;
  for(double vi : v)
    out -= vi * vi;

  arma::vec d_beta(gr.begin(), n_fixef, false),
             d_Sig(d_beta.end(), vcov_dim, false);

  // handle the fixed effects
  F77_CALL(dtpsv)
    ("U", "N", "N", &n, K_chol, v.memptr(), &incx, 1, 1, 1);

  d_beta += X * v;

  // handle the covariance matrix
  arma::vec Z_K_inv_v = Z * v;
  arma::mat K_chol_inv_Z = Z.t();
  for(arma::uword i = 0; i < K_chol_inv_Z.n_cols; ++i)
    F77_CALL(dtpsv)
    ("U", "T", "N", &n, K_chol, K_chol_inv_Z.colptr(i), &incx, 1, 1, 1);
  arma::mat const K_chol_inv_Z_outer = K_chol_inv_Z.t() * K_chol_inv_Z;

  double * d_Sig_ij{d_Sig.memptr()};
  for(arma::uword j = 0; j < n_rng; ++j, ++d_Sig_ij){
    for(arma::uword i = 0; i < j; ++i, ++d_Sig_ij)
      *d_Sig_ij = Z_K_inv_v(i) * Z_K_inv_v(j) - K_chol_inv_Z_outer(i, j);
    *d_Sig_ij = (Z_K_inv_v(j) * Z_K_inv_v(j) - K_chol_inv_Z_outer(j, j)) / 2;
  }

  return out / 2;
}

mixed_gsm_cluster::mixed_gsm_cluster
  (arma::mat const &X, arma::mat const &X_prime, arma::mat const &Z,
   arma::vec const &y, arma::vec const &event){
  arma::uword const n_obs_all{X.n_cols};
  if(X_prime.n_cols != n_obs_all)
    throw std::invalid_argument("X_prime.n_cols != n_total()");
  else if(Z.n_cols != n_obs_all)
    throw std::invalid_argument("Z.n_cols != n_total()");
  else if(y.n_elem != n_obs_all)
    throw std::invalid_argument("X_prime.n_elem != n_total()");
  else if(event.n_elem != n_obs_all)
    throw std::invalid_argument("event.n_elem != n_total()");
  else if(X.n_rows != X_prime.n_rows)
    throw std::invalid_argument("X.n_rows != X_prime.n_rows");

  // fill in the indices for the observed and censored individuals
  arma::uvec idx_obs(n_obs_all),
             idx_cen(n_obs_all);
  arma::uword obs_counter{}, cen_counter{};
  for(arma::uword i = 0; i < n_obs_all; ++i)
    if(event[i] > 0)
      idx_obs[obs_counter++] = i;
    else
      idx_cen[cen_counter++] = i;

  idx_obs.reshape(obs_counter, 1);
  idx_cen.reshape(cen_counter, 1);

  Zo = Z.cols(idx_obs);
  Xo = X.cols(idx_obs);
  Xo_prime = X_prime.cols(idx_obs);
  yo = y(idx_obs);

  Zc = Z.cols(idx_cen);
  Xc = X.cols(idx_cen);
  yc = y(idx_cen);
}

mixed_gsm_cluster::mixed_gsm_cluster_res mixed_gsm_cluster::operator()
  (arma::vec const &beta, arma::mat const &Sigma, int const maxpts,
   int const key, double const abseps, double const releps,
   bool const use_adaptive) const {
  assert(beta.n_elem == n_fixef());
  assert(Sigma.n_rows == n_rng());
  assert(Sigma.n_cols == n_rng());

  mixed_gsm_cluster_res out;
  out.log_like = 0;
  out.inform = 0;

  if(n_obs() > 0){
    for(arma::uword i = 0; i < Xo_prime.n_cols; ++i)
      out.log_like += log(colvecdot(Xo_prime, i, beta));
    out.log_like += gsm_normal_term(Xo, Zo, Sigma, beta).func();

  } if(n_cens() > 0){
    auto const res = gsm_cens_term(Zo, Zc, Xo, Xc, beta, Sigma)
      .func(maxpts, key, abseps, releps, use_adaptive);
    out.log_like += res.log_like;
    out.inform = res.inform;

  }

  return out;
}

mixed_gsm_cluster::mixed_gsm_cluster_res mixed_gsm_cluster::grad
  (arma::vec &gr, arma::vec const &beta, arma::mat const &Sigma,
   int const maxpts, int const key, double const abseps,
   double const releps, bool const use_adaptive) const {
  assert(beta.n_elem == n_fixef());
  assert(Sigma.n_rows == n_rng());
  assert(Sigma.n_cols == n_rng());

  mixed_gsm_cluster_res out;
  out.log_like = 0;
  out.inform = 0;

  arma::vec tmp; // for the gradient output of each call
  gr.zeros(beta.n_elem + (n_rng() * (n_rng() + 1)) / 2);

  if(n_obs() > 0){
    for(arma::uword j =  0; j < Xo_prime.n_cols; ++j){
      double const lp{colvecdot(Xo_prime, j, beta)};
      out.log_like += log(lp);
      for(arma::uword i = 0; i < Xo_prime.n_rows; ++i)
        gr[i] += Xo_prime(i, j)/lp;
    }
    out.log_like += gsm_normal_term(Xo, Zo, Sigma, beta).gr(tmp);
    gr += tmp;

  } if(n_cens() > 0){
    auto const res = gsm_cens_term(Zo, Zc, Xo, Xc, beta, Sigma)
      .gr(tmp, maxpts, key, abseps, releps, use_adaptive);
    out.log_like += res.log_like;
    out.inform = res.inform;
    gr += tmp;

  }

  return out;
}
