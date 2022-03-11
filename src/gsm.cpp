#include "gsm.h"
#include "ranrth-wrapper.h"
#include "integrand-binary.h"
#include "utils.h"
#include "restrict-cdf.h"
#include <stdexcept>
#include <numeric>

using namespace integrand;
using namespace ranrth_aprx;

/**
 * fills the derivative of a symmetric matrix given the derivatives of the full
 * matrix
 */
inline void fill_vcov_derivs(double *derivs, arma::mat const &ders_pass){
  arma::uword const n_rng{ders_pass.n_rows};
  for(arma::uword j = 0; j < n_rng; ++j, ++derivs){
    for(arma::uword i = 0; i < j; ++i, ++derivs)
      *derivs = ders_pass(i, j) + ders_pass(j, i);
    *derivs = ders_pass(j, j);
  }
}

gsm_cens_term::gsm_cens_term
  (arma::mat const &Zo, arma::mat const &Zc, arma::mat const &Xo,
   arma::mat const &Xc, arma::vec const &beta, arma::mat const &Sigma):
  Zo{Zo}, Zc{Zc}, Xo{Xo}, Xc{Xc}, beta{beta}, Sigma{Sigma} {
    // check all the dimensions
    assert(Zo.n_cols == n_obs());
    assert(Zc.n_cols == n_cen());

    assert(Sigma.n_cols == n_rng());
    assert(Sigma.n_rows == n_rng());

    assert(Xo.n_rows == n_fixef());
    assert(Xc.n_rows == n_fixef());
  }

gsm_cens_term::gsm_cens_output gsm_cens_term::func
  (int const maxpts, int const key, double const abseps,
   double const releps, gsm_approx_method const method_use) const {
  if(n_cen() < 1)
    return {0, 0, 0};

  if(method_use == gsm_approx_method::spherical_radial ||
     method_use == gsm_approx_method::adaptive_spherical_radial)
    return func_spherical_radial
      (maxpts, key, abseps, releps,
       method_use == gsm_approx_method::adaptive_spherical_radial);
  else if(method_use != gsm_approx_method::cdf_approach)
    throw std::invalid_argument("method is not implemented");

  // the CDF approach
  arma::vec eta = n_fixef() > 0 ? Xc.t() * beta
                                : arma::vec(n_cen(), arma::fill::zeros);
  if(n_obs() < 1){
    arma::mat const K = ([&]{
      arma::mat out = Zc.t() * Sigma * Zc;
      out.diag() += 1;
      return out;
    })();

    restrictcdf::cdf<restrictcdf::likelihood>::set_working_memory(K.n_rows, 1L);
    auto res = restrictcdf::cdf<restrictcdf::likelihood>
      (eta, K, true).approximate(maxpts, abseps, releps);

    return { std::log(res.finest[0]), res.inform, res.minvls };
  }

  arma::mat const H = Zo * Zo.t() + arma::inv_sympd(Sigma);
  arma::vec const eta_rhs = Zo * (Xo.t() * beta);
  arma::mat const K =  ([&]{
    arma::mat out = Zc.t() * arma::solve(H, Zc, arma::solve_opts::likely_sympd);
    out.diag() += 1;
    return out;
  })();
  eta -= Zc.t() * arma::solve(H, eta_rhs);

  restrictcdf::cdf<restrictcdf::likelihood>::set_working_memory(K.n_rows, 1L);
  auto res = restrictcdf::cdf<restrictcdf::likelihood>
    (eta, K, true).approximate(maxpts, abseps, releps);

  return { std::log(res.finest[0]), res.inform, res.minvls };
}

gsm_cens_term::gsm_cens_output gsm_cens_term::func_spherical_radial
  (int const maxpts, int const key, double const abseps,
   double const releps, bool const use_adaptive) const {
  arma::vec eta = n_fixef() > 0 ? -(Xc.t() * beta)
                                : arma::vec(n_cen(), arma::fill::zeros);

  if(n_obs() < 1){
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
        new adaptive<mvn<mix_binary > > (m, true, 1000L)));

    return integral_arpx(maxpts, key, abseps, releps);
  })();

  return { std::log(res.value), res.inform, res.inivls };
}

gsm_cens_term::gsm_cens_output gsm_cens_term::gr
  (arma::vec &gr,int const maxpts, int const key, double const abseps,
   double const releps, gsm_approx_method const method_use) const {
  arma::uword const vcov_dim{(Sigma.n_cols * (Sigma.n_cols + 1)) / 2};
  gr.zeros(beta.size() + vcov_dim);
  if(n_cen() < 1)
    return {0, 0, 0};

  if(method_use == gsm_approx_method::spherical_radial ||
     method_use == gsm_approx_method::adaptive_spherical_radial)
    return gr_spherical_radial
    (gr, maxpts, key, abseps, releps,
     method_use == gsm_approx_method::adaptive_spherical_radial);
  else if(method_use != gsm_approx_method::cdf_approach)
    throw std::invalid_argument("method is not implemented");

  // the cdf approach
  arma::vec eta = n_fixef() > 0 ? Xc.t() * beta
                                : arma::vec(n_cen(), arma::fill::zeros);

  arma::vec d_beta(gr.begin(), n_fixef(), false),
             d_Sig(d_beta.end(), vcov_dim, false);

  auto get_d_vcov_full = [&](double const *d_vcov_ij){
    arma::uword const n_full{Zc.n_cols};
    arma::mat d_vcov_full(n_full, n_full);

    for(arma::uword j = 0; j < n_full; ++j)
      for(arma::uword i = 0; i <= j; ++i, ++d_vcov_ij){
        // restrictcdf scales down the off diagonal entries by two
        d_vcov_full(i, j) = *d_vcov_ij;
        d_vcov_full(j, i) = *d_vcov_ij;
      }

    return d_vcov_full;
  };

  if(n_obs() < 1){
    arma::mat const K = ([&]{
      arma::mat out = Zc.t() * Sigma * Zc;
      out.diag() += 1;
      return out;
    })();

    restrictcdf::cdf<restrictcdf::deriv>::set_working_memory(K.n_rows, 1L);
    auto res = restrictcdf::cdf<restrictcdf::deriv>
      (eta, K, true).approximate(maxpts, abseps, releps);

    double const int_val{res.finest[0]};

    res.finest /= int_val; // account for the log(f)
    arma::vec d_eta(res.finest.begin() + 1, n_cen(), false);
    double const * const d_vcov{d_eta.end()};

    d_beta += Xc * d_eta;

    arma::mat const d_vcov_full = get_d_vcov_full(d_vcov);
    arma::mat const d_Sigma = Zc * d_vcov_full * Zc.t();
    fill_vcov_derivs(d_Sig.memptr(), d_Sigma);

    return { std::log(int_val), res.inform, res.minvls };
  }

  arma::mat const H = Zo * Zo.t() + arma::inv_sympd(Sigma);
  arma::vec const eta_rhs = Zo * (Xo.t() * beta);
  arma::mat const K =  ([&]{
    arma::mat out = Zc.t() * arma::solve(H, Zc, arma::solve_opts::likely_sympd);
    out.diag() += 1;
    return out;
  })();
  eta -= Zc.t() * arma::solve(H, eta_rhs);

  restrictcdf::cdf<restrictcdf::deriv>::set_working_memory(K.n_rows, 1L);
  auto res = restrictcdf::cdf<restrictcdf::deriv>
    (eta, K, true).approximate(maxpts, abseps, releps);

  double const int_val{res.finest[0]};

  res.finest /= int_val; // account for the log(f)
  arma::vec d_eta(res.finest.begin() + 1, n_cen(), false);
  double const * const d_vcov{d_eta.end()};

  // handle the derivatives w.r.t. the fixed effects
  arma::vec const Zc_d_eta = n_fixef() > 0 ? Zc * d_eta : arma::vec();
  if(n_fixef() > 0)
    d_beta += Xc * d_eta -
      Xo * Zo.t() * arma::solve(H, Zc_d_eta, arma::solve_opts::likely_sympd);

  // handle the derivatives w.r.t the covariance matrix
  arma::mat d_H_inv = Zc * get_d_vcov_full(d_vcov) * Zc.t();

  if(n_fixef() > 0)
    for(arma::uword j = 0; j < d_H_inv.n_cols; ++j){
      for(arma::uword i = 0; i < j; ++i){
        d_H_inv(i, j) -= Zc_d_eta[i] * eta_rhs[j];
        d_H_inv(j, i) -= Zc_d_eta[j] * eta_rhs[i];
      }
      d_H_inv(j, j) -= Zc_d_eta[j] * eta_rhs[j];
    }

  arma::mat const d_Sigma = dcond_vcov(H, d_H_inv, Sigma);
  fill_vcov_derivs(d_Sig.memptr(), d_Sigma);

  return { std::log(int_val), res.inform, res.minvls };
}

gsm_cens_term::gsm_cens_output gsm_cens_term::gr_spherical_radial
  (arma::vec &gr, int const maxpts, int const key, double const abseps,
   double const releps, bool const use_adaptive) const {
  arma::uword const vcov_dim{(Sigma.n_cols * (Sigma.n_cols + 1)) / 2};

  arma::vec eta = n_fixef() > 0 ? -(Xc.t() * beta)
                                : arma::vec(n_cen(), arma::fill::zeros);

  arma::vec d_beta(gr.begin(), n_fixef(), false),
             d_Sig(d_beta.end(), vcov_dim, false);

  if(n_obs() < 1){
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
    arma::vec d_eta(res.value.begin() + 1, n_cen(), false),
             d_vcov(d_eta.end(), vcov_dim, false);

    if(n_fixef() > 0)
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
  arma::vec d_eta(res.value.begin() + 1, n_cen(), false);
  double const * const d_vcov{d_eta.end()};

  // handle the fixed effects and then the covariance matrix parameters
  arma::vec const Zc_d_eta = n_fixef() > 0 ? Zc * d_eta : arma::vec();
  if(n_fixef() > 0)
    d_beta = Xo * Zo.t() * solve(H, Zc_d_eta) - Xc * d_eta;

  /* application of the chain rule in the reverse direction. First we fill in
   * the whole derivative w.r.t. H^(-1). Here, we have to account for the terms
   * from nabla_eta as eta is given by VH^(-1)v. Thus, the terms we need to add
   * is V^T.nabla_eta^T.v^T.
   */
  arma::mat dH_inv(n_rng(), n_rng());
  {
    double const * d_vcov_ij{d_vcov};
    if(n_fixef() > 0)
      for(arma::uword j = 0; j < n_rng(); ++j, ++d_vcov_ij){
        for(arma::uword i = 0; i < j; ++i, ++d_vcov_ij){
          dH_inv(i, j) = *d_vcov_ij / 2 + Zc_d_eta[i] * eta_rhs[j];
          dH_inv(j, i) = *d_vcov_ij / 2 + Zc_d_eta[j] * eta_rhs[i];
        }
        dH_inv(j, j) = *d_vcov_ij + Zc_d_eta[j] * eta_rhs[j];
      }
      else
        for(arma::uword j = 0; j < n_rng(); ++j, ++d_vcov_ij){
          for(arma::uword i = 0; i < j; ++i, ++d_vcov_ij){
            dH_inv(i, j) = *d_vcov_ij / 2;
            dH_inv(j, i) = *d_vcov_ij / 2;
          }
          dH_inv(j, j) = *d_vcov_ij;
        }
  }
  arma::mat d_Sigma = dcond_vcov(H, dH_inv, Sigma);
  fill_vcov_derivs(d_Sig.memptr(), d_Sigma);

  return { std::log(int_val), res.inform, res.inivls };
}

gsm_normal_term::gsm_normal_term
  (arma::mat const &X, arma::mat const &Z, arma::mat const &Sigma,
   arma::vec const &beta):
  X{X}, Z{Z}, Sigma{Sigma}, beta{beta} {
    assert(X.n_rows == n_fixef());
    assert(X.n_cols == n_obs());
    assert(Z.n_rows == n_rng());
    assert(Sigma.n_rows == n_rng());

    arma::mat K = Z.t() * Sigma * Z;
    K.diag() += 1;

    arma::mat K_chol_full = arma::chol(K);
    double * K_cp_to{K_chol};
    for(arma::uword j = 0; j < n_obs(); ++j)
      for(arma::uword i = 0; i <= j; ++i)
        *K_cp_to++ = K_chol_full(i, j);

    log_K_deter = 0;
    for(arma::uword i = 0; i < n_obs(); ++i)
      log_K_deter += 2 * std::log(K_chol_full(i, i));
  }

double gsm_normal_term::func() const {
  if(n_fixef() == 0)
    return 0;

  arma::vec v = -X.t() * beta;
  int n = n_obs(), incx = 1;
  F77_CALL(dtpsv)
    ("U", "T", "N", &n, K_chol, v.memptr(), &incx, 1, 1, 1);

  double out = -static_cast<double>(n_obs()) * log(2 * M_PI);
  out -= log_K_deter;
  for(double vi : v)
    out -= vi * vi;

  return out / 2;
}

double gsm_normal_term::gr(arma::vec &gr) const {
  arma::uword const vcov_dim{(n_rng() * (n_rng() + 1)) / 2};
  gr.zeros(beta.size() + vcov_dim);
  if(n_fixef() == 0)
    return 0;

  arma::vec v = -X.t() * beta;
  int n = n_obs(), incx = 1;
  F77_CALL(dtpsv)
    ("U", "T", "N", &n, K_chol, v.memptr(), &incx, 1, 1, 1);

  double out = -static_cast<double>(n_obs()) * log(2 * M_PI);
  out -= log_K_deter;
  for(double vi : v)
    out -= vi * vi;

  arma::vec d_beta(gr.begin(), n_fixef(), false),
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
  for(arma::uword j = 0; j < n_rng(); ++j, ++d_Sig_ij){
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
   gsm_approx_method const method_use) const {
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

  }
  if(n_cens() > 0){
    auto const res = gsm_cens_term(Zo, Zc, Xo, Xc, beta, Sigma)
      .func(maxpts, key, abseps, releps, method_use);
    out.log_like += res.log_like;
    out.inform = res.inform;

  }

  return out;
}

mixed_gsm_cluster::mixed_gsm_cluster_res mixed_gsm_cluster::grad
  (arma::vec &gr, arma::vec const &beta, arma::mat const &Sigma,
   int const maxpts, int const key, double const abseps,
   double const releps, gsm_approx_method const method_use) const {
  assert(beta.n_elem == n_fixef());
  assert(Sigma.n_rows == n_rng());
  assert(Sigma.n_cols == n_rng());

  mixed_gsm_cluster_res out;
  out.log_like = 0;
  out.inform = 0;

  arma::vec tmp; // for the gradient output of each call
  gr.zeros(n_fixef() + (n_rng() * (n_rng() + 1)) / 2);

  if(n_obs() > 0){
    for(arma::uword j =  0; j < Xo_prime.n_cols; ++j){
      double const lp{colvecdot(Xo_prime, j, beta)};
      out.log_like += log(lp);
      for(arma::uword i = 0; i < Xo_prime.n_rows; ++i)
        gr[i] += Xo_prime(i, j)/lp;
    }
    out.log_like += gsm_normal_term(Xo, Zo, Sigma, beta).gr(tmp);
    gr += tmp;

  }
  if(n_cens() > 0){
    auto const res = gsm_cens_term(Zo, Zc, Xo, Xc, beta, Sigma)
      .gr(tmp, maxpts, key, abseps, releps, method_use);
    out.log_like += res.log_like;
    out.inform = res.inform;
    gr += tmp;

  }

  return out;
}

namespace {
arma::mat pedigree_vcov
  (std::vector<arma::mat> const &scale_mats, arma::vec const &sigs){
  arma::uword const dim{scale_mats[0].n_rows};
  arma::mat out(dim, dim, arma::fill::zeros);
  out.diag() += 1;

  for(size_t i = 0; i < scale_mats.size(); ++i)
    out += sigs[i] * scale_mats[i];
  return out;
}

} // namespace

mixed_gsm_cluster_pedigree::mixed_gsm_cluster_pedigree
  (std::vector<arma::mat> const &scale_mats, arma::mat const &X,
   arma::mat const &X_prime, arma::vec const &y, arma::vec const &event):
  scale_mats{scale_mats}{
    arma::uword const n_obs_all{X.n_cols};
    if(X_prime.n_cols != n_obs_all)
      throw std::invalid_argument("X_prime.n_cols != n_total()");
    else if(y.n_elem != n_obs_all)
      throw std::invalid_argument("X_prime.n_elem != n_total()");
    else if(event.n_elem != n_obs_all)
      throw std::invalid_argument("event.n_elem != n_total()");
    else if(X.n_rows != X_prime.n_rows)
      throw std::invalid_argument("X.n_rows != X_prime.n_rows");

    if(n_scales() < 1)
      throw std::invalid_argument("no scale matrices");
    for(auto &mat : scale_mats)
      if(mat.n_rows != n_obs_all || mat.n_cols != n_obs_all)
        throw std::invalid_argument("no scale matrices (dimensions are not consistent with other arguments)");

    // fill in the indices for the observed and censored individuals
    idx_obs.resize(n_obs_all);
    idx_cens.resize(n_obs_all);
    arma::uword obs_counter{}, cen_counter{};
    for(arma::uword i = 0; i < n_obs_all; ++i)
      if(event[i] > 0)
        idx_obs[obs_counter++] = i;
      else
        idx_cens[cen_counter++] = i;

    idx_obs.reshape(obs_counter, 1);
    idx_cens.reshape(cen_counter, 1);

    Xo = X.cols(idx_obs);
    Xo_prime = X_prime.cols(idx_obs);
    yo = y(idx_obs);

    Xc = X.cols(idx_cens);
    yc = y(idx_cens);
  }

/// computes the log marginal likelihood.
mixed_gsm_cluster_pedigree::mixed_gsm_cluster_pedigree_res
  mixed_gsm_cluster_pedigree::operator()
  (arma::vec const &beta, arma::vec const &sigs, int const maxpts,
   int const key, double const abseps, double const releps,
   gsm_approx_method const method_use) const {
  assert(beta.n_elem == n_fixef());
  assert(sigs.n_elem == n_scales());

  mixed_gsm_cluster_pedigree_res out;
  out.log_like = 0;
  out.inform = 0;

  arma::mat Zo(n_rng(), n_obs(), arma::fill::zeros),
            Zc(n_rng(), n_cens(), arma::fill::zeros);
  for(arma::uword i = 0; i < n_obs(); ++i)
    Zo(idx_obs[i], i) = 1;
  for(arma::uword i = 0; i < n_cens(); ++i)
    Zc(idx_cens[i], i) = 1;

  auto Sigma = pedigree_vcov(scale_mats, sigs);

  if(n_obs() > 0){
    for(arma::uword i = 0; i < Xo_prime.n_cols; ++i)
      out.log_like += log(colvecdot(Xo_prime, i, beta));
    out.log_like += gsm_normal_term(Xo, Zo, Sigma, beta).func();

  }
  if(n_cens() > 0){
    auto const res = gsm_cens_term(Zo, Zc, Xo, Xc, beta, Sigma)
      .func(maxpts, key, abseps, releps, method_use);
    out.log_like += res.log_like;
    out.inform = res.inform;

  }

  return out;
}

/**
 * computes the gradient, setting the result gr, and returns the log marginal
 * likelihood.
 */
mixed_gsm_cluster_pedigree::mixed_gsm_cluster_pedigree_res
  mixed_gsm_cluster_pedigree::grad
  (arma::vec &gr, arma::vec const &beta, arma::vec const &sigs,
   int const maxpts, int const key, double const abseps,
   double const releps, gsm_approx_method const method_use) const {
  assert(beta.n_elem == n_fixef());
  assert(sigs.n_elem == n_scales());

  mixed_gsm_cluster_pedigree_res out;
  out.log_like = 0;
  out.inform = 0;

  arma::mat Zo(n_rng(), n_obs(), arma::fill::zeros),
  Zc(n_rng(), n_cens(), arma::fill::zeros);
  for(arma::uword i = 0; i < n_obs(); ++i)
    Zo(idx_obs[i], i) = 1;
  for(arma::uword i = 0; i < n_cens(); ++i)
    Zc(idx_cens[i], i) = 1;

  auto Sigma = pedigree_vcov(scale_mats, sigs);

  arma::vec tmp; // for the gradient output of each call
  arma::vec gr_full
    (n_fixef() + (n_rng() * (n_rng() + 1)) / 2, arma::fill::zeros);

  if(n_obs() > 0){
    for(arma::uword j =  0; j < Xo_prime.n_cols; ++j){
      double const lp{colvecdot(Xo_prime, j, beta)};
      out.log_like += log(lp);
      for(arma::uword i = 0; i < Xo_prime.n_rows; ++i)
        gr_full[i] += Xo_prime(i, j)/lp;
    }
    out.log_like += gsm_normal_term(Xo, Zo, Sigma, beta).gr(tmp);
    gr_full += tmp;

  }
  if(n_cens() > 0){
    auto const res = gsm_cens_term(Zo, Zc, Xo, Xc, beta, Sigma)
      .gr(tmp, maxpts, key, abseps, releps, method_use);
    out.log_like += res.log_like;
    out.inform = res.inform;
    gr_full += tmp;

  }

  // set the gradient
  gr.resize(n_fixef() + n_scales());
  std::copy(gr_full.begin(), gr_full.begin() + n_fixef(), gr.begin());

  for(size_t i = 0; i < n_scales(); ++i){
    auto &scale_mat{scale_mats[i]};
    double const *d_Sigma{gr_full.begin() + n_fixef()};
    double d_sig{};
    for(arma::uword k = 0; k < n_rng(); ++k, d_Sigma += k)
      d_sig += std::inner_product
        (d_Sigma, d_Sigma + k + 1, scale_mat.colptr(k), 0.);
    gr[n_fixef() + i] = d_sig;
  }

  return out;
}
