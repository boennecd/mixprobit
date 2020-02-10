#include "restrict-cdf.h"

thread_local static restrictcdf::mvkbrv_ptr current_mvkbrv_ptr = nullptr;

namespace restrictcdf {
void set_mvkbrv_ptr(mvkbrv_ptr new_ptr){
  current_mvkbrv_ptr = new_ptr;
}
}

extern "C"
{
  void F77_NAME(mvkbrveval)(
      int const* /* NDIM */, int const* /* MAXVLS */, int const* /* NF */,
      double const* /* ABSEPS */, double const* /* RELEPS */,
      double* /* ABSERR */, double* /* FINEST */, int* /* INFORM */);

  void F77_SUB(mvkbrvintegrand)
    (int const *m, double *unifs, int const *mf, double *out){
    assert(current_mvkbrv_ptr);
    assert(out);
    (*current_mvkbrv_ptr)(m, unifs, mf, out);
  }
}

namespace restrictcdf {
output approximate_integral(
    int const ndim, int const n_integrands, int const maxvls,
    double const abseps, double const releps){
  output out;
  out.finest.resize(n_integrands);

  F77_CALL(mvkbrveval)(
      &ndim, &maxvls, &n_integrands, &abseps, &releps,
      &out.abserr, out.finest.memptr(), &out.inform);

  return out;
}

int likelihood::get_n_integrands
(arma::vec const &mu, arma::mat const &sigma) {
  return 1L;
}

arma::vec likelihood::integrand
(arma::vec const &draw, likelihood::comp_dat const& dat){
  return arma::vec(1L, arma::fill::ones);
}

int deriv::get_n_integrands
(arma::vec const &mu, arma::mat const &sigma) {
  arma::uword const p = mu.n_elem;
  return 1 + p + (p * (p + 1)) / 2L;
}

arma::vec deriv::integrand
(arma::vec const &draw, deriv::comp_dat const& dat){
  assert(dat.mu);
  arma::uword const p = dat.mu->n_elem;
  arma::vec out(1L + p + (p * (p + 1L)) / 2L);

  out[0L] = 1.;
  arma::vec mean_part(out.memptr() + 1L, p, false);
  /* TODO: exploit triangular matrix times vector */
  mean_part = dat.sigma_chol_inv * draw;

  /* TODO: do something smarter */
  arma::mat intermediate = mean_part * mean_part.t() - dat.signa_inv;
  double *o = out.memptr() + 1L + p;
  for(unsigned c = 0; c < p; c++)
    for(unsigned r = 0; r <= c; r++)
      *o++ = intermediate(r, c) / 2.;

  return out;
}

template class cdf<likelihood>;
template class cdf<deriv>;
}
