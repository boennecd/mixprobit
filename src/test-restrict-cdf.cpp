#include "restrict-cdf.h"
#include <testthat.h>
#include <limits>
#include "threat-safe-random.h"

context("restrictcdf unit tests") {
  test_that("cdf<likelihood> gives similar output to R") {
/*
 set.seed(1)
 n <- 4
 mean <- rnorm(n)
 sigma <- drop(rWishart(1L, 2L * n, diag(n)))

 lower <- rep(-Inf, n)
 upper <- rep(0, n)
 mean  <- round(mean , 3)
 sigma <- round(sigma, 3)

 library(mvtnorm)
 prob <- pmvnorm(lower, upper, mean, sigma = sigma,
 algorithm = GenzBretz(abseps = 1e-9, maxpts = 1000000L))

 dput(mean)
 dput(sigma)
 dput(prob)
*/
    Rcpp::RNGScope rngScope;
    parallelrng::set_rng_seeds(1L);
    constexpr double Inf = std::numeric_limits<double>::infinity();

    arma::vec mean;
    arma::mat sigma;

    mean << -0.626 << 0.18 << -0.836 << 1.595;

    sigma << 8.287 << -0.848 << -0.879 << -1.788 << -0.848 << 3.581
          << 2.916 << -3.957 << -0.879 << 2.916 << 7.361 << -0.648
          << -1.788 << -3.957 << -0.648 << 11.735;
    sigma.reshape(4L, 4L);

    double const abseps = std::pow(std::numeric_limits<double>::epsilon(),
                                   .33);
    double constexpr E_prop(0.0181507102495727);
    {
      auto res = restrictcdf::cdf<restrictcdf::likelihood>(
        mean, sigma).approximate(1000000L, abseps, -1);

      expect_true(res.inform == 0L);
      expect_true(res.abserr                        < 100. * abseps);
      expect_true(std::abs(res.finest[0L] - E_prop) < 100. * abseps);
    }
  }
}
