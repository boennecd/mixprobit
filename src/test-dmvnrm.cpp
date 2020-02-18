#include <testthat.h>
#include <limits>
#include "dmvnrm.h"

context("dmvnrm unit tests") {
  test_that("dmvnrm gives the same at mvtnorm::dmvnorm") {
    /*
     n <- 4L
     set.seed(1)
     Sigma <- drop(rWishart(1, 2L * n, diag(1/(2 * n), n)))
     x <- rnorm(n)
     Sigma <- round(Sigma, 4)
     x <- round(x, 4)
     dput(c(Sigma))
     dput(c(x))
     dput(mvtnorm::dmvnorm(x, sigma = Sigma, log = TRUE))
     */

    using std::sqrt;
    arma::vec x;
    x << 0.3898 << -0.6212 << -2.2147 << 1.1249;
    arma::mat Sigma;
    Sigma << 0.6065 << 0.3504 << -0.424 << 0.1585 << 0.3504 << 1.6387
          << -0.6384 << -0.0378 << -0.424 << -0.6384 << 1.2039 << 0.4026
          << 0.1585 << -0.0378 <<  0.4026 << 0.74;
    Sigma.reshape(4L, 4L);

    double const expectl = -12.6550751029262,
                 expect  = std::exp(expectl),
                 resl    = dmvnrm(x, arma::inv(arma::chol(Sigma))),
                 res     = dmvnrm(x, arma::inv(arma::chol(Sigma)), false),
                 eps     = sqrt(std::numeric_limits<double>::epsilon());
    expect_true(std::abs((expectl - resl) / expectl) < eps);
    expect_true(std::abs((expect  - res ) / expect ) < eps);
  }
}
