#include "ranrth-wrapper.h"
#include "integrand-binary.h"
#include <testthat.h>
#include <limits>

context("ranrth-wrapper unit tests") {
  test_that("ranrth-wrapper gives correct result with mix_binary") {
    /*
     set.seed(1)
     n <- 4L
    p <- 2L
    Z <- do.call(                        # random effect design matrix
    rbind, c(list(1), list(replicate(n, runif(p - 1L, -1, 1)))))
    eta <- runif(n, -1, 1)               # fixed offsets/fixed effects
    n <- NCOL(Z)                         # number of individuals
    p <- NROW(Z)                         # number of random effects
    S <- drop(                           # covariance matrix of random effects
    rWishart(1, p, diag(sqrt(1/ 2 / p), p)))

    S   <- round(S  , 3)
    Z   <- round(Z  , 3)
    eta <- round(eta, 3)

    S_chol <- chol(S)
    u <- drop(rnorm(p) %*% S_chol)       # random effects
    y <- runif(n) < pnorm(drop(u %*% Z)) # observed outcomes

#####
# use GH quadrature
    library(fastGHQuad)
    b <- 50L                             # number of nodes to use
    rule <- fastGHQuad::gaussHermiteData(b)
    f <- function(x)
    sum(mapply(pnorm, q = eta + sqrt(2) * drop(x %*% S_chol %*% Z),
               lower.tail = y, log.p = TRUE))
    idx <- do.call(expand.grid, replicate(p, 1:b, simplify = FALSE))

    xs <- local({
    args <- list(FUN = c, SIMPLIFY = FALSE)
    do.call(mapply, c(args, lapply(idx, function(i) rule$x[i])))
    })
    ws_log <- local({
    args <- list(FUN = prod)
    log(do.call(mapply, c(args, lapply(idx, function(i) rule$w[i]))))
    })

# function that makes the approximation
    f1 <- function()
    sum(exp(ws_log + vapply(xs, f, numeric(1L)))) / pi^(p / 2)

    dput(Z)
    dput(S)
    dput(eta)
    dput(as.integer(y))
    dput(f1())
     */

    using namespace ranrth_aprx;
    Rcpp::RNGScope rngScope;

    constexpr arma::uword const n = 4L,
                                p = 2L;
    arma::mat Z, S;
    Z << 1 << -0.469 << 1 << -0.256 << 1 << 0.146 << 1 << 0.816;
    Z.reshape(p, n);
    S << 0.76 << 0.3 << 0.3 << 0.178;
    S.reshape(p, p);
    arma::vec eta;
    eta << -0.597 << 0.797 << 0.889 << 0.322;
    arma::ivec y;
    y << 1L << 0L << 0L << 1L;

    constexpr double const expec  = 0.0041247747590393,
                           epsabs = 1e-5;
    for(int key = 1L; key < 5L; ++key){
      set_integrand(std::unique_ptr<integrand>(
          new mix_binary(y, eta, Z, S)));
      auto res = integral_arpx(100000L, key, epsabs, -1.);

      expect_true(res.err < 1e-3);
      expect_true((res.inform == 0 || res.inform == 1));
      if(res.inform == 0L)
        expect_true(std::abs(res.value - expec) < 100. * epsabs);
      else
        expect_true(std::abs(res.value - expec) < 100. * res.err);
    }
  }
}
