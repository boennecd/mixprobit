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
(arma::vec const &draw, arma::vec const &mu, arma::mat const &sigma,
 arma::mat const &sigma_chol){
  return arma::vec(1L, arma::fill::ones);
}

template class cdf<likelihood>;
}
