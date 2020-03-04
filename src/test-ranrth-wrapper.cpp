#include "ranrth-wrapper.h"
#include "integrand-binary.h"
#include <testthat.h>
#include <limits>
#include "threat-safe-random.h"

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
    parallelrng::set_rng_seeds(1L);

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
                           abseps = 1e-5;
    for(int key = 1L; key < 5L; ++key){
      set_integrand(std::unique_ptr<integrand::base_integrand>(
          new integrand::mix_binary(y, eta, Z, S)));
      auto res = integral_arpx(100000L, key, abseps, -1.);

      expect_true(res.err < 1e-3);
      expect_true((res.inform == 0 || res.inform == 1));
      if(res.inform == 0L)
        expect_true(std::abs(res.value - expec) < 100. * abseps);
      else
        expect_true(std::abs(res.value - expec) < 100. * res.err);
    }
  }

  test_that("ranrth-wrapper gives correct result with mix_binary for derivatives") {
    /*
     set.seed(1)
     n <- 4L
    p <- 2L
    Z <- do.call(                        # random effect design matrix
    rbind, c(list(1), list(replicate(n, runif(p - 1L, -1, 1)))))
    n <- NCOL(Z)                         # number of individuals
    p <- NROW(Z)                         # number of random effects
    q <- 2L
    X <- matrix(runif(n * q, -.5, .5), nr = n)
    S <- drop(                           # covariance matrix of random effects
    rWishart(1, p, diag(sqrt(1/ 2 / p), p)))

    S    <- round(S  , 3)
    Z    <- round(Z  , 3)
    X    <- round(X  , 2)
    beta <- c(-1, 1)
    eta  <- drop(X %*% beta)

    S_chol <- chol(S)
    u <- drop(rnorm(p) %*% S_chol)       # random effects
    y <- runif(n) < pnorm(eta + drop(u %*% Z)) # observed outcomes

#####
# use GH quadrature
    library(fastGHQuad)
    b <- 50L                             # number of nodes to use
    rule <- fastGHQuad::gaussHermiteData(b)
    f <- function(x, eta)
    sum(mapply(pnorm, q = eta +  sqrt(2) * drop(x %*% S_chol %*% Z),
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
    f1 <- function(b)
    sum(exp(ws_log + vapply(
        xs, f, numeric(1L), eta = drop(X %*% b)))) / pi^(p / 2)

    dput(Z)
    dput(S)
    dput(eta)
    dput(as.integer(y))
    dput(t(X))
    dput(f1(beta))

    library(numDeriv)
    dput(jacobian(f1, beta))
     */
    using namespace ranrth_aprx;
    using namespace integrand;
    Rcpp::RNGScope rngScope;
    parallelrng::set_rng_seeds(1L);

    constexpr arma::uword const n = 4L,
                                p = 2L,
                                q = 2L;
    arma::mat Z, S, X;
    Z << 1 << -0.469 << 1 << -0.256 << 1 << 0.146 << 1 << 0.816;
    Z.reshape(p, n);
    S << 0.904 << 1.016 << 1.016 << 1.973;
    S.reshape(p, p);
    X << -0.3 << 0.13 << 0.4 << -0.44 << 0.44 << -0.29 << 0.16 << -0.32;
    X.reshape(q, n);
    arma::vec eta;
    eta << 0.43 << -0.84 << -0.73 << -0.48;
    arma::ivec y;
    y << 1L << 0L << 0L << 1L;

    constexpr double const abseps = 1e-5;
    arma::vec expec;
    expec << 0.0828866564251371 << -0.0452131647376384
          << 0.0216941956246238;
    for(int key = 1L; key < 5L; ++key){
      /*mix_binary bin(y, eta, Z, S, &X);
      mvn<mix_binary> m(bin);

      set_integrand(std::unique_ptr<base_integrand>(
          new adaptive<mvn<mix_binary > > (m))); */

      set_integrand(std::unique_ptr<base_integrand>(
          new mix_binary(y, eta, Z, S, &X)));
      auto run_test = [&]{
        auto res = jac_arpx(20000L, key, abseps, -1.);
        expect_true(res.value.n_elem == 3L);
        expect_true(res.err.n_elem == 3L);

        for(unsigned i = 0; i < res.err.n_elem; ++i){
          double const e = res.err[i];
          expect_true(e  < 1e-3);
          if(res.inform == 0L)
            expect_true(std::abs(res.value[i] - expec[i]) < 100. * abseps);
          else
            expect_true(std::abs(res.value[i] - expec[i]) < 100. * e);
        }
      };
      run_test();

      /* test w/ adaptive method */
      mix_binary bin(y, eta, Z, S, &X);
      mvn<mix_binary> m(bin);

      set_integrand(std::unique_ptr<base_integrand>(
          new adaptive<mvn<mix_binary > > (m)));
      run_test();
    }
  }
}
