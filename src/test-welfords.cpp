#include <testthat.h>
#include <limits>
#include "welfords.h"
#include <vector>

context("weolford's unit tests") {
  test_that("weolford's  gives correct mean and variance estiamte") {
    /*
     set.seed(1)
     dput(x <- round(rnorm(10), 2))
     dput(mean(x))
     dput(mean((x - mean(x))^2))
     */
    std::vector<double> const x =
      { -0.63, 0.18, -0.84, 1.6, 0.33, -0.82, 0.49, 0.74, 0.58, -0.31 };
    double const var_ex = 0.552216,
                mean_ex = 0.132,
                 eps    = std::sqrt(std::numeric_limits<double>::epsilon());

    {
      welfords w;
      for(auto xi : x)
        w += xi;

      expect_true(std::abs((w.mean() - mean_ex) / mean_ex) < eps);
      expect_true(std::abs((w.var()  - var_ex ) / var_ex ) < eps);
    }

    {
      welfords w, w2;
      auto xi = x.begin();
      for(unsigned i = 0; i < 5; ++i)
        w  += *xi++;
      for(; xi != x.end(); ++xi)
        w2 += *xi;

      w += w2;
      expect_true(std::abs((w.mean() - mean_ex) / mean_ex) < eps);
      expect_true(std::abs((w.var()  - var_ex ) / var_ex ) < eps);
    }
  }
}
