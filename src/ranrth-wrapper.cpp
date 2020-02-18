#include "ranrth-wrapper.h"
#include <cmath>
#include <R.h>
#include <assert.h>

namespace {
double eval_current_integrand(double const*);
}

extern "C"
{
  void F77_NAME(ranrtheval)(
    int const* /* M */, int const* /* mxvals */,
    double const* /* epsabs */, double const* /* epsrel */,
    int const* /* key */, double* /* value */, double* /* error */,
    int* /* intvls */, int* /* inform */, double* /* WK */);

  void F77_SUB(evalintegrand)
    (int const *m, double const *par, int const *mf, double *out){
    assert(mf and *mf == 1L);
    assert(out);
    *out =  eval_current_integrand(par);
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
  int const m = current_integrand->get_n_par();
  std::unique_ptr<double[]> wk;
  if(key == 1L)
    wk.reset(new double[6L + 2 * m]);
  else
    wk.reset(new double[6L + m * (m + 2L)]);

  integral_arpx_res out;
  double &value = out.value,
         &err   = out.err;
  int &intvls = out.inivls,
      &inform = out.inform;

  F77_CALL(ranrtheval)(
    &m, &maxpts, &abseps, &releps, &key, &value, &err, &intvls, &inform,
    wk.get());

  return out;
}
}

namespace {
double eval_current_integrand(double const *par){
  if(!current_integrand)
    Rf_error("Current integrand is not set");

#ifndef NDEBUG
  std::size_t const m = current_integrand->get_n_par();
  assert(m > 1L);
  for(std::size_t i = 0; i < m; ++i)
    assert((par + i));
#endif

  double const out = current_integrand->operator()(par);
  if(!std::isfinite(out))
    Rf_error("Non-finite integrand value");

  return out;
}
}
