#include "mvtnorm-wrapper.h"
#include <cmath>
#include <R_ext/RS.h>

extern "C"
{
void F77_NAME(mvtdst)(
    int const*    /* N */, int const* /* NU */,
    double const* /* lower */, double const* /* upper */,
    int const* /* infin */, double const* /* correl */,
    double const* /* delta */, int const* /* maxpts  */,
    double const* /* abseps */, double const* /* releps */,
    double* /* error */, double* /* value */,
    int* /* inform */, int* /* intvls */);
}

namespace pmvnorm {
arma::ivec get_infin(arma::vec const &lower, arma::vec const &upper){
  arma::uword n = lower.size();
  assert(upper.size() == n);
  arma::ivec out(n);

  double const *l = lower.begin(),
               *u = upper.begin();
  for(auto &o : out){
    bool const li = std::isinf(*l++),
               ui = std::isinf(*u++);
    if      ( li and  ui)
      o = -1L;
    else if ( li and !ui)
      o =  0L;
    else if (!li and  ui)
      o =  1L;
    else
      o =  2L;
  }

  return out;
}

cor_vec_res get_cor_vec(const arma::mat &cov){
  cor_vec_res out;
  arma::vec     &sds = out.sds,
            &cor_vec = out.cor_vec;

  arma::uword const n = cov.n_cols;
  sds = arma::sqrt(cov.diag());
  cor_vec.resize((n * (n - 1L)) / 2L);

  assert(n == cov.n_rows);
  assert(n > 1L);

  double *o = cor_vec.begin();
  for(unsigned c = 1L; c < n; ++c)
    for(unsigned r = 0; r < c; ++r)
      *o++ = cov(r, c) / sds[c] / sds[r];

  return out;
}

cdf_res cdf(arma::vec const &lower, arma::vec const &upper,
            arma::ivec const &infin, arma::vec const &mean,
            arma::vec const &cor_vec, int const maxpts,
            double const abseps, double const releps){
  /* checks */
  int const dim = lower.size();
#ifndef NDEBUG
  arma::uword const udim = dim;
#endif
  assert(dim > 0L);
  assert(upper  .size() == udim);
  assert(infin  .size() == udim);
  assert(mean   .size() == udim);
  assert(cor_vec.size() == (udim * (udim - 1L)) / 2L);
  assert(abseps > 0. || releps > 0.);

  int const maxpts_arg = maxpts <= 0L ? dim * 100L : maxpts,
                    nu = 0L;
  int inform, intvls;
  double value, error;

  F77_NAME(mvtdst)(
      &dim, &nu, lower.begin(), upper.begin(), infin.begin(),
      cor_vec.begin(), mean.begin(), &maxpts_arg, &abseps, &releps, &error,
      &value, &inform, &intvls);

  return { error, value, inform, intvls };
}

cdf_res cdf(arma::vec lower, arma::vec upper, arma::vec mean,
            arma::mat const &cov, int const maxpts, double const abseps,
            double const releps){
#ifndef NDEBUG
  arma::uword const n = lower.size();
#endif
  assert(n > 0L);
  assert(upper.size() == n);
  assert(mean .size() == n);
  assert(cov.n_cols   == n);
  assert(cov.n_rows   == n);

  arma::vec sds, cor_vec;
  {
    auto tmp = get_cor_vec(cov);
    sds     = std::move(tmp.sds);
    cor_vec = std::move(tmp.cor_vec);
  }

  lower /= sds;
  upper /= sds;
  mean  /= sds;

  arma::ivec const infin = get_infin(lower, upper);

  return cdf(lower, upper, infin, mean, cor_vec, maxpts, abseps, releps);
}
}
