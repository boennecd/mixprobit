#ifndef RESTRICT_CDF_H
#define RESTRICT_CDF_H

#include "arma-wrap.h"
#include <array>
#include <limits>

namespace restrictcdf {
inline std::array<double, 2> draw_trunc_mean
  (double const b, const double u, bool comp_quantile){
  double const qb = R::pnorm5(b, 0, 1, 1L, 0L);
  if(comp_quantile)
    return { qb, R::qnorm5(qb * u, 0, 1, 1L, 0L) };
  return { qb, std::numeric_limits<double>::quiet_NaN() };
}

/* Holds output of integral approximation */
struct output {
  /* minvls: Actual number of function evaluations used.
   * inform: INFORM = 0 for normal exit, when
   *              ABSERR <= MAX(ABSEPS, RELEPS*||finest||)
   *           and
   *              INTVLS <= MAXCLS.
   *         INFORM = 1 If MAXVLS was too small to obtain the required
   *         accuracy. In this case a value finest is returned with
   *         estimated absolute accuracy ABSERR. */
  int minvls, inform;
  /* abserr: Maximum norm of estimated absolute accuracy of finest. */
  double abserr;
  /* Estimated NF-vector of values of the integrals. */
  arma::vec finest;
};

typedef void (*mvkbrv_ptr)(int const*, double*, int const*, double*);
void set_mvkbrv_ptr(mvkbrv_ptr);

output approximate_integral(
    int const ndim, int const n_integrands, int const maxvls,
    double const abseps, double const releps);

/* func classes used as template argument for cdf */
/* used to approximate the likelihood */
class likelihood {
public:
  static int get_n_integrands(arma::vec const&, arma::mat const&);
  static arma::vec integrand(arma::vec const&, arma::vec const&,
                             arma::mat const&, arma::mat const&);
  constexpr static bool needs_last_unif() {
    return false;
  }
};

/* Approximates the integrals of the multivariate normal cdf over a
 * rectangle from (-Inf, 0)^[# of random effects] times some function in
 * the integrand.
 *
 * First, call this constructor. Then call the approximate member
 * function.
 *
 * Args:
 *   funcs: class with static member functions which determines the type
 *          of integrals is approximated.
 */
template<class funcs>
class cdf {
  thread_local static int ndim, n_integrands;
  thread_local static arma::vec mu;
  thread_local static arma::mat sigma;
  thread_local static arma::vec sigma_chol;
  static constexpr bool const needs_last_unif =
    funcs::needs_last_unif();

public:
  /* function to be called from mvkbrv */
  static void eval_integrand(
      int const *ndim_in, double *unifs, int const *n_integrands_in,
      double *integrand_val){
    assert(*ndim_in         == ndim);
    assert(*n_integrands_in == n_integrands);

    arma::vec u(unifs        , ndim        , false),
            out(integrand_val, n_integrands, false),
            draw(ndim);

    double w(1.), *sc = sigma_chol.memptr();
    for(unsigned j = 0; j < (unsigned)ndim; ++j){
      double b(-mu[j]);
      for(unsigned k = 0; k < j; ++k)
        b -= *sc++ * draw[k];
      b /= *sc++;

      auto const draw_n_p = draw_trunc_mean
        (b, u[j], needs_last_unif or j + 1 < (unsigned)ndim);
      w       *= draw_n_p[0];
      draw[j]  = draw_n_p[1];
    }

    out.zeros();
    out += funcs::integrand(draw, mu, sigma, sigma_chol) * w;
  }

  /* Args:
   *   mu: mean vector.
   *   sigma: covariance matrix.
   */
  cdf(arma::vec const &mu_in, arma::mat const &sigma_in){
    ndim = mu_in.n_elem;
    n_integrands = funcs::get_n_integrands(mu_in, sigma_in);

    /* checks */
    assert(sigma_in.n_cols == (unsigned)ndim);
    assert(sigma_in.n_rows == (unsigned)ndim);
    assert(n_integrands > 0);

    /* re-scale */
    arma::vec const sds = arma::sqrt(arma::diagvec(sigma_in));
    mu = mu_in / sds;
    sigma = ([&](){
      arma::mat out = sigma_in;
      out.each_row() /= sds.t();
      out.each_col() /= sds;

      return out;
    })();

    sigma_chol = ([&]{
      arma::uword const p = sigma.size();

      arma::mat tmp(arma::chol(sigma));
      arma::vec out((p * (p + 1L)) / 2L);

      double *o = out.memptr();
      for(unsigned c = 0; c < p; c++)
        for(unsigned r = 0; r <= c; r++)
          *o++ = tmp.at(r, c);

      return out;
    })();
  }

  /* Args:
   *   maxvls: maximum number of function evaluations allowed.
   *   abseps: required absolute accuracy.
   *   releps: equired relative accuracy.
   */
  static output approximate
  (int const maxvls, double const abseps, double const releps){
    assert((abseps > 0 or releps > 0));
    assert(maxvls > 0);

    /* set pointer to this class' member function */
    set_mvkbrv_ptr(&cdf<funcs>::eval_integrand);
    return approximate_integral(ndim, n_integrands, maxvls, abseps, releps);
  }
};

/* initialize static members */
template<class funcs>
thread_local int cdf<funcs>::ndim = 0L;
template<class funcs>
thread_local int cdf<funcs>::n_integrands = 0L;
template<class funcs>
thread_local arma::vec cdf<funcs>::mu = arma::vec();
template<class funcs>
thread_local arma::mat cdf<funcs>::sigma = arma::mat();
template<class funcs>
thread_local arma::vec cdf<funcs>::sigma_chol = arma::vec();
}

#endif

