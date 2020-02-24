#include "mix-optim.h"
#include <testthat.h>
#include <limits>
#include <array>

typedef std::array<double, 6L> coef_data;

inline double poly2   (int npar, double *point, void *data){
  coef_data &coefs = *(coef_data*)data;
  double const x1 = *point,
               x2 = *(point + 1L);

  return
    coefs[0] +
      coefs[1] * x1 +
      coefs[2] * x2 +
      coefs[3] * x1 * x1 +
      coefs[4] * x2 * x2 +
      coefs[5] * x2 * x1;
}

inline void poly2_gr(int npar, double *point, double *grad, void *data){
  coef_data &coefs = *(coef_data*)data;
  double const x1 = *point,
               x2 = *(point + 1L);

  * grad       = coefs[1] + 2 * coefs[3] * x1 + coefs[5] * x2;
  *(grad + 1L) = coefs[2] + 2 * coefs[4] * x2 + coefs[5] * x1;
}

context("mix-optim unit tests") {
  test_that("optim works for second order polynomial") {
    coef_data dat;
    dat[0] =  4;
    dat[1] = -2;
    dat[2] = -3;
    dat[3] =  1;
    dat[4] =  2;
    dat[5] = -1;

    double const x1 = 11. / 7.,
                 x2 = 8.  / 7.,
                val = 5.  / 7.;


    arma::vec start;
    start << 0. << 0.;

    double const eps = std::sqrt(std::numeric_limits<double>::epsilon());
    auto const res = optimizers::bfgs(
      start, poly2, poly2_gr, (void *)&dat, 10000L,
      0L, -1, eps);

    expect_true(res.fail == 0L);
    expect_true(res.fncount > 0L);
    expect_true(res.grcount > 0L);
    expect_true(std::abs((res.val - val) / val) < 10 * eps);
    expect_true(std::abs((res.par[0] - x1) / x1) < std::sqrt(eps));
    expect_true(std::abs((res.par[1] - x2) / x2) < std::sqrt(eps));
  }
}

