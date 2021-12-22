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
                           abseps = 1e-4;
    for(int key = 1L; key < 5L; ++key){
      set_integrand(std::unique_ptr<integrand::base_integrand>(
          new integrand::mix_binary(y, eta, Z, S)));
      auto res = integral_arpx(1000000L, key, abseps, -1.);

      expect_true(res.inform == 0L);
      expect_true(res.err < 4 * abseps);
      expect_true(std::abs(res.value - expec) < 4 * abseps);
    }
  }

  test_that("ranrth-wrapper gives correct result with mix_binary for derivatives") {
    /*
     set.seed(2)
     n <- 4L
    p <- 3L
    Z <- do.call(                        # random effect design matrix
    rbind, c(list(1), list(replicate(n, runif(p - 1L, -1, 1)))))
    n <- NCOL(Z)                         # number of individuals
    p <- NROW(Z)                         # number of random effects
    q <- 2L
    X <- matrix(runif(n * q, -.5, .5), nr = n)
    S <- drop(                           # covariance matrix of random effects
    rWishart(1, 2 * p, diag(sqrt(1/ 2 / p), p)))

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
    b <- 25L                             # number of nodes to use
    rule <- fastGHQuad::gaussHermiteData(b)
    f <- function(x, eta, S_chol)
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
    f1 <- function(par){
    b <- head(par, q)
    s <- tail(par, -q)
    S <- matrix(nr = p, nc = p)
    S[upper.tri(S, TRUE)] <- s
    S[lower.tri(S)] <- t(S)[lower.tri(S)]
    sum(exp(ws_log + vapply(
        xs, f, numeric(1L), eta = drop(X %*% b), S_chol = chol(S)))) /
    pi^(p / 2)
    }

    dput(Z)
    dput(S)
    dput(eta)
    dput(as.integer(y))
    dput(t(X))
    xx <- c(beta, S[upper.tri(S, TRUE)])
    dput(f1(xx))

    library(numDeriv)
    dput(jacobian(f1, xx))
     */
    using namespace ranrth_aprx;
    using namespace integrand;
    Rcpp::RNGScope rngScope;
    parallelrng::set_rng_seeds(1L);

    constexpr arma::uword const n = 4L,
                                p = 3L,
                                q = 2L;
    arma::mat Z, S, X;
    Z << 1 << -0.63 << 0.405 << 1 << 0.147 << -0.664 << 1 << 0.888 << 0.887
      << 1 << -0.742 << 0.667;
    Z.reshape(p, n);
    S << 5.407 << -0.424 << -1.545 << -0.424 << 1.51 << 1.505 << -1.545
      << 1.505 << 2.147;
    S.reshape(p, p);
    X << -0.03 << 0.26 << 0.05 << -0.32 << 0.05 << -0.09 << -0.26 << 0.35;
    X.reshape(q, n);
    arma::vec eta;
    eta << 0.29 << -0.37 << -0.14 << 0.61;
    arma::ivec y;
    y << 0L << 0L << 0L << 0L;

    constexpr double const releps = 1e-1;
    arma::vec expec;
    expec << 0.223175338758645
          << 0.0103246106433447 << -0.0160370124670925
          << 0.021541576093451 << -0.0021754945059288
          << -0.0211698036190797 << 0.0193386723207052
          << 0.00618822615335066 << -0.00709493339802656;
    arma::uword const ex_dim = 1L + q + (p * (p + 1L)) / 2L;
    for(int key = 1L; key < 5L; ++key){
      mix_binary bin(y, eta, Z, S, &X);
      set_integrand(std::unique_ptr<base_integrand>(
          new mix_binary(y, eta, Z, S, &X)));
      auto run_test = [&]{
        auto res = jac_arpx(10000000L, key, -1, releps);
        bin.Jacobian_post_process(res.value);

        expect_true(res.value.n_elem == ex_dim);
        expect_true(res.err.n_elem == ex_dim);

        expect_true(res.inform == 0L);
        for(unsigned i = 0; i < res.err.n_elem; ++i){
          expect_true(res.err[i] < 4 * releps * std::abs(res.value[i]));
          expect_true(std::abs(res.value[i] - expec[i]) <
            4 * std::abs(expec[i]) * releps);
        }
      };
      run_test();

      /* test w/ adaptive method */

      mvn<mix_binary> m(bin);

      set_integrand(std::unique_ptr<base_integrand>(
          new adaptive<mvn<mix_binary > > (m, true)));
      run_test();
    }
  }
}
