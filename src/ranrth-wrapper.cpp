#include "ranrth-wrapper.h"
#include <cmath>
#include <R.h>
#include <assert.h>

namespace {
static bool comp_integrand = false;
void eval_current_integrand(double*, double const*, int const);
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

using integrand::base_integrand;
static std::unique_ptr<base_integrand> current_integrand;

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
  assert(current_integrand);

  /* assign working memory */
  int const m = current_integrand->get_n_par(),
           nf = 1L;
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
  assert(current_integrand);

  /* assign working memory */
  int const m = current_integrand->get_n_par(),
           nf = current_integrand->get_n_jac();
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
  if(!current_integrand)
    Rf_error("Current integrand is not set");

#ifndef NDEBUG
  std::size_t const m = current_integrand->get_n_par();
  assert(m > 1L);
  for(std::size_t i = 0; i < m; ++i)
    assert((par + i));
  assert(nf == 1L or (unsigned)nf == current_integrand->get_n_jac());
  for(int i = 0; i < nf; ++i)
    assert((out + i));
#endif

  if(comp_integrand){
    double const o = current_integrand->operator()(par);
    if(!std::isfinite(o))
      Rf_error("Non-finite integrand value");
    *out = o;
    return;
  }

  arma::vec jac(out, nf, false);
  current_integrand->Jacobian(par, jac);
  for(int i = 0; i < nf; ++i)
    if(!std::isfinite(jac.at(i)))
      Rf_error("Non-finite integrand value");
}
}
