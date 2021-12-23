#include "ranrth-wrapper.h"
#include <cmath>
#include <R.h>
#include <assert.h>
#include "welfords.h"
#include "threat-safe-random.h"
#include <vector>

namespace {
using integrand::base_integrand;
static std::unique_ptr<base_integrand> current_integrand;

static bool comp_integrand = false;
void eval_current_integrand(double*, double const*, int const);

// the quantile used to judge convergence
constexpr double alpha_crit{2.575829};

/// special case when it is integration is one dimension
ranrth_aprx::integral_arpx_res integral_arpx_1D
  (int const maxpts, double const abseps, double const releps){
  ranrth_aprx::integral_arpx_res out;
  double &value{out.value},
           &err{out.err};
  int &intvls{out.inivls},
      &inform{out.inform};

  comp_integrand = true;

  constexpr int n_sample_inner{64L};
  double samples[n_sample_inner];
  intvls = 0;
  inform = 1;

  int n_samp{50};
  welfords est;
  for(; intvls < maxpts && inform != 0; ){
    // determine the number of samples
    n_samp *= 2;
    n_samp = std::min(n_samp, (maxpts - intvls + 1) / 2);

    // perform the MC estimation
    int n_samples_taken{};
    for(; n_samples_taken < n_samp;){
      int const n_it{std::min(n_samp - n_samples_taken, n_sample_inner)};
      n_samples_taken += n_it;

      for(int r = 0; r < n_it; ++r)
        samples[r] = rngnorm_wrapper();

      for(int r = 0; r < n_it; ++r){
        double func_val{}, res;
        eval_current_integrand(&func_val, samples + r, 1);
        res = func_val;
        samples[r] *= -1;
        eval_current_integrand(&func_val, samples + r, 1);
        res += func_val;
        est += res / 2;
      }
    }

    intvls += 2 * n_samples_taken;

    // check if the methods has converged
    double const mean_est{est.mean()},
                      err{std::sqrt(est.var() / (intvls / 2.))};

    if(alpha_crit * err < std::max(abseps, releps * std::abs(mean_est))){
      inform = 0;
      break;
    }
  }

  value = est.mean();
  err = std::sqrt(est.var() / (intvls / 2.));
  return out;
}

/// special case when it is integration in one dimension
ranrth_aprx::jac_arpx_res jac_arpx_1D
  (int const maxpts, double const abseps, double const releps){
  ranrth_aprx::jac_arpx_res out;
  arma::vec &value{out.value},
            &err{out.err};
  int &intvls{out.inivls},
      &inform{out.inform};

  comp_integrand = false;

  constexpr int n_sample_inner{64L};
  int const nf = current_integrand->get_n_jac();
  double samples[n_sample_inner];
  intvls = 0;
  inform = 1;
  value.set_size(nf);
  err.set_size(nf);

  int n_samp{50};
  // working memory. We do not need value and err till the end
  double * const wk1{value.begin()},
         * const wk2{err.begin()};
  std::vector<welfords> ests(nf, welfords{});
  for(; intvls < maxpts && inform != 0; ){
    // determine the number of samples
    n_samp *= 2;
    n_samp = std::min(n_samp, (maxpts - intvls + 1) / 2);

    // perform the MC estimation
    int n_samples_taken{};
    for(; n_samples_taken < n_samp;){
      int const n_it{std::min(n_samp - n_samples_taken, n_sample_inner)};
      n_samples_taken += n_it;

      for(int r = 0; r < n_it; ++r)
        samples[r] = rngnorm_wrapper();

      for(int r = 0; r < n_it; ++r){
        eval_current_integrand(wk1, samples + r, nf);
        samples[r] *= -1;
        eval_current_integrand(wk2, samples + r, nf);
        for(int est_i = 0; est_i < nf; ++est_i)
          ests[est_i] += (wk1[est_i] + wk2[est_i]) / 2;
      }
    }

    intvls += 2 * n_samples_taken;

    // check if the methods has converged
    bool passed{true};
    for(int est_i = 0; est_i < nf && passed; ++est_i){
      double const mean_est{ests[est_i].mean()},
                        err{std::sqrt(ests[est_i].var() / (intvls / 2.))};
      passed &=
        alpha_crit * err < std::max(abseps, releps * std::abs(mean_est));
    }

    if(passed){
      inform = 0;
      break;
    }
  }

  for(int est_i = 0; est_i < nf; ++est_i){
    value[est_i] = ests[est_i].mean();
    err[est_i] =  std::sqrt(ests[est_i].var() / (intvls / 2.));
  }

  return out;
}
}

extern "C"
{
  void F77_NAME(ranrtheval)(
    int const* /* M */, int const* /* mxvals */,
    double const* /* epsabs */, double const* /* epsrel */,
    int const* /* key */, double* /* value */, double* /* error */,
    int* /* intvls */, int* /* inform */, double* /* WK */,
    int const * /* NF */);

  void F77_SUB(evalintegrand)
    (int const *m, double const *par, int const *nf, double *out){
    assert(nf);
    eval_current_integrand(out, par, *nf);
  }
}

namespace ranrth_aprx {
void set_integrand
  (std::unique_ptr<base_integrand> new_val){
  current_integrand.swap(new_val);
}

integral_arpx_res integral_arpx(
    int const maxpts, int const key, double const abseps,
    double const releps){
  assert(maxpts > 0);
  assert(key > 0L and key < 5L);
  assert(abseps > 0 or releps > 0);
  if(!current_integrand)
    Rf_error("Current integrand is not set");

  /* assign working memory */
  int const m = current_integrand->get_n_par(),
           nf = 1L;
  assert(m > 0);

  if(m < 2)
    return integral_arpx_1D(maxpts, abseps, releps);

  std::unique_ptr<double[]> wk;
  if(key == 1L)
    wk.reset(new double[6L * nf + 2 * m]);
  else
    wk.reset(new double[6L * nf + m * (m + 2L)]);

  integral_arpx_res out;
  double &value = out.value,
         &err   = out.err;
  int &intvls = out.inivls,
      &inform = out.inform;

  comp_integrand = true;
  F77_CALL(ranrtheval)(
    &m, &maxpts, &abseps, &releps, &key, &value, &err, &intvls, &inform,
    wk.get(), &nf);

  return out;
}

jac_arpx_res jac_arpx(
    int const maxpts, int const key, double const abseps,
    double const releps){
  assert(maxpts > 0);
  assert(key > 0L and key < 5L);
  assert(abseps > 0 or releps > 0);
  if(!current_integrand)
    Rf_error("Current integrand is not set");

  /* assign working memory */
  int const m = current_integrand->get_n_par(),
           nf = current_integrand->get_n_jac();
  assert(m > 0);

  if(m < 2)
    return jac_arpx_1D(maxpts, abseps, releps);

  std::unique_ptr<double[]> wk;
  if(key == 1L)
    wk.reset(new double[6L * nf + 2 * m]);
  else
    wk.reset(new double[6L * nf + m * (m + 2L)]);

  jac_arpx_res out;
  arma::vec &value = out.value,
            &err   = out.err;
  value.set_size(nf);
  err.set_size(nf);
  int &intvls = out.inivls,
      &inform = out.inform;

  comp_integrand = false;
  F77_CALL(ranrtheval)(
      &m, &maxpts, &abseps, &releps, &key, value.memptr(), err.memptr(),
      &intvls, &inform, wk.get(), &nf);

  return out;
}
}

namespace {
void eval_current_integrand(double *out, double const *par, int const nf){
#ifndef NDEBUG
  std::size_t const m = current_integrand->get_n_par();
  for(std::size_t i = 0; i < m; ++i)
    assert((par + i));
  assert(nf == 1L or (unsigned)nf == current_integrand->get_n_jac());
  for(int i = 0; i < nf; ++i)
    assert((out + i));
#endif

  if(comp_integrand){
    double const o = current_integrand->operator()(par);
    if(!std::isfinite(o))
      // TODO: Rf_error should not be called in parallel + expensive to do these
      //       checks over and over again?
      Rf_error("Non-finite integrand value");
    *out = o;
    return;
  }

  arma::vec jac(out, nf, false);
  current_integrand->Jacobian(par, jac);
  for(int i = 0; i < nf; ++i)
    if(!std::isfinite(jac.at(i)))
      // TODO: Rf_error should not be called in parallel + expensive to do these
      //       checks over and over again?
      Rf_error("Non-finite integrand value");
}
}
