#include <testthat.h>
#include <limits>
#include "sobol.h"

context("sobol unit tests") {
  test_that("sobol gives the same without scrambiling") {
    /*
     dput(t(randtoolbox::sobol(10, 3)))
     */
    constexpr size_t dim = 3L,
                       n = 10L;
    constexpr double const expect[dim * n] =
      { 0.5, 0.5, 0.5, 0.75, 0.25, 0.75, 0.25, 0.75, 0.25,
        0.375, 0.375, 0.625, 0.875, 0.875, 0.125, 0.625, 0.125, 0.375,
        0.125, 0.625, 0.875, 0.1875, 0.3125, 0.3125, 0.6875, 0.8125,
        0.8125, 0.9375, 0.0625, 0.5625 };

    sobol_gen gen(3L);
    double const * e = expect,
                 eps = std::sqrt(std::numeric_limits<double>::epsilon());
    arma::vec x(dim);
    for(size_t i = 0; i < n; ++i){
      gen(x);
      for(size_t j = 0; j < dim; ++j, ++e)
        expect_true(std::abs((x[j] - *e) / *e) < eps);
    }
  }

  test_that("sobol gives the same with scrambiling = 1") {
    /*
     dput(t(randtoolbox::sobol(10, 3, scrambling = 1L, seed = 656768)))
     */
    constexpr size_t dim = 3L,
                       n = 5L,
              scrambling = 1L,
                    seed = 656768;
    constexpr double const expect[dim * n] =
      { 0.287378100678325, 0.828440777026117, 0.807095920667052,
        0.167303617112339, 0.283706185407937, 0.688851526938379, 0.578640728257596,
        0.585681160911918, 0.146814747713506, 0.747379070147872, 0.44326694495976,
        0.934730102308095, 0.0063720066100359, 0.672702186740935, 0.478627008385956 };

    sobol_gen gen(3L, scrambling, seed);
    double const * e = expect,
      eps = std::sqrt(std::numeric_limits<double>::epsilon());
    arma::vec x(dim);
    for(size_t i = 0; i < n; ++i){
      gen(x);
      for(size_t j = 0; j < dim; ++j, ++e)
        expect_true(std::abs((x[j] - *e) / *e) < eps);
    }
  }
}
