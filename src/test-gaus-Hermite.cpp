#include "gaus-Hermite.h"
#include "integrand-binary.h"
#include <testthat.h>
#include <limits>

context("GaussHermite unit tests") {
  test_that("gaussHermiteData gives the same result as R") {
    /*
     library(fastGHQuad)
     xw <- fastGHQuad::gaussHermiteData(10L)
     dput(xw$x)
     dput(xw$w)
     */
    constexpr unsigned const b = 10L;
    arma::vec ex_x, ex_w;
    ex_x << -3.43615911883774 << -2.53273167423279 << -1.75668364929988
         << -1.03661082978951 << -0.342901327223705 << 0.342901327223705
         << 1.03661082978951 << 1.75668364929988 << 2.53273167423279
         << 3.43615911883774;
    ex_w << 7.6404328552326e-06 << 0.00134364574678124
         << 0.0338743944554811 << 0.240138611082314 << 0.610862633735326
         << 0.610862633735326 << 0.240138611082315 << 0.033874394455481
         << 0.00134364574678124 << 7.64043285523265e-06;

    auto const &rules = GaussHermite::gaussHermiteDataCached(b);

    expect_true(arma::norm(ex_x - rules.x) < 1e-10);
    expect_true(arma::norm(ex_w - rules.w) < 1e-10);
  }

  test_that("approx gives correct result with mix_binary") {
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

    constexpr double const expec  = 0.0041247747590393;

    mix_binary integrand(y, eta, Z, S);
    auto const &rule = GaussHermite::gaussHermiteDataCached(30L);

    expect_true(
      std::abs(GaussHermite::approx(rule, integrand) - expec) < 1e-8);
  }
}
