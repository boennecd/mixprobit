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

    constexpr arma::uword n{4},
                          p{3},
                          q{2};
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

    constexpr double const rel_eps{5e-1};
    arma::vec expec;
    expec << 0.223175338758645
          << 0.0103246106433447 << -0.0160370124670925
          << 0.021541576093451 << -0.0021754945059288
          << -0.0211698036190797 << 0.0193386723207052
          << 0.00618822615335066 << -0.00709493339802656;
    {
      arma::uword const ex_dim = 1L + q + (p * (p + 1L)) / 2L;

      set_integrand(std::unique_ptr<base_integrand>(
          new mix_binary(y, eta, Z, S, &X)));
      mix_binary bin(y, eta, Z, S, &X);
      auto run_test = [&](int const key){
        auto res = jac_arpx(100000000L, key, -1, rel_eps);
        expect_true(res.value.n_elem == ex_dim);
        expect_true(res.err.n_elem == ex_dim);

        expect_true(res.inform == 0L);
        for(unsigned i = 0; i < res.err.n_elem; ++i){
          expect_true(res.err[i] < 2 * rel_eps * std::abs(res.value[i]));
          expect_true
            (std::abs(res.value[i] - expec[i]) <
              2 * rel_eps * std::abs(expec[i]));
        }
      };
      run_test(2);

      /* test w/ adaptive method */
      mvn<mix_binary> m(bin);

      set_integrand(std::unique_ptr<base_integrand>(
          new adaptive<mvn<mix_binary > > (m, true)));
      run_test(1);
      run_test(2);
      run_test(3);
      run_test(4);
    }

    /*
# function for the derivative w.r.t. the offset
      f2 <- \(par){
      eta <- head(par, n)
      s <- tail(par, -n)
      S <- matrix(nr = p, nc = p)
      S[upper.tri(S, TRUE)] <- s
      S[lower.tri(S)] <- t(S)[lower.tri(S)]
      sum(exp(ws_log + vapply(
      xs, f, numeric(1L), eta = eta, S_chol = chol(S)))) /
      pi^(p / 2)
      }

      xx <- c(eta, S[upper.tri(S, TRUE)])
      dput(f2(xx))
      dput(jacobian(f2, xx))
      */

    expec = arma::vec
      {0.223175338758645, -0.0307327929409755, -0.0147947171684584, -0.0384440750249272, -0.046402178704353, 0.021541576093451, -0.0021754945059288, -0.0211698036190797, 0.0193386723207052, 0.00618822615335066, -0.00709493339802656};

    constexpr arma::uword ex_dim{1L + n + (p * (p + 1L)) / 2L};

    set_integrand(std::unique_ptr<base_integrand>(
        new mix_binary(y, eta, Z, S)));
    mix_binary bin(y, eta, Z, S);
    auto run_test = [&](int const key){
      auto res = jac_arpx(100000000L, key, -1, rel_eps);
      expect_true(res.value.n_elem == ex_dim);
      expect_true(res.err.n_elem == ex_dim);

      expect_true(res.inform == 0L);
      for(unsigned i = 0; i < res.err.n_elem; ++i){
        expect_true(res.err[i] < 2 * rel_eps * std::abs(res.value[i]));
        expect_true
          (std::abs(res.value[i] - expec[i]) <
            2 * rel_eps * std::abs(expec[i]));
      }
    };
    run_test(2);

    /* test w/ adaptive method */
    mvn<mix_binary> m(bin);

    set_integrand(std::unique_ptr<base_integrand>(
        new adaptive<mvn<mix_binary > > (m, true)));
    run_test(1);
    run_test(2);
    run_test(3);
    run_test(4);
  }

  test_that("ranrth-wrapper gives correct result with mix_binary for derivatives with the number of observations being less than the number of random effects") {
    /*
     set.seed(2)
     n <- 2L
     p <- 4L
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

    constexpr arma::uword n{2},
                          p{4},
                          q{2};
    arma::mat Z{1, -0.63, 0.405, 0.147, 1, -0.664, 0.888, 0.887},
              S{2.653, -0.232, -0.134, -0.38, -0.232, 3.097, 0.447, -1.051, -0.134, 0.447, 4.751, 2.159, -0.38, -1.051, 2.159, 4.126},
              X{-0.37, -0.03, 0.33, 0.05};
    Z.reshape(p, n);
    S.reshape(p, p);
    X.reshape(q, n);
    arma::vec eta{0.34, -0.28};
    arma::ivec y{0, 0};

    constexpr double const rel_eps{5e-1};
    {
      arma::vec expec{0.368125090148117, 0.0231547051105411, 0.000973349562432535, 0.00622070279491852, -0.00824456594424145, 0.00271816571363873, 0.0108131821534993, -0.00678049379010119, 0.00196371803414583, 0.0106758132172105, -0.00648455781045827, 0.000897181658087495, -0.00184020017999673};
      arma::uword const ex_dim{1L + q + (p * (p + 1L)) / 2L};

      mix_binary bin(y, eta, Z, S, &X);
      set_integrand(std::unique_ptr<base_integrand>(
          new mix_binary(y, eta, Z, S, &X)));
      auto run_test = [&](int const key){
        auto res = jac_arpx(100000000L, key, -1, rel_eps);
        expect_true(res.value.n_elem == ex_dim);
        expect_true(res.err.n_elem == ex_dim);

        expect_true(res.inform == 0L);
        for(unsigned i = 0; i < res.err.n_elem; ++i){
          expect_true(res.err[i] < 2 * rel_eps * std::abs(res.value[i]));
          expect_true
            (std::abs(res.value[i] - expec[i]) <
              2 * rel_eps * std::abs(expec[i]));
        }
      };
      run_test(2);

      /* test w/ adaptive method */
      mvn<mix_binary> m(bin);

      set_integrand(std::unique_ptr<base_integrand>(
          new adaptive<mvn<mix_binary > > (m, true)));
      run_test(1);
      run_test(2);
      run_test(3);
      run_test(4);
    }

    /*
# function with the dervative w.r.t. the offset
     f2 <- function(par){
     eta <- head(par, n)
     s <- tail(par, -n)
     S <- matrix(nr = p, nc = p)
     S[upper.tri(S, TRUE)] <- s
     S[lower.tri(S)] <- t(S)[lower.tri(S)]
     sum(exp(ws_log + vapply(
     xs, f, numeric(1L), eta = eta, S_chol = chol(S)))) /
     pi^(p / 2)
     }

     xx <- c(eta, S[upper.tri(S, TRUE)])
     dput(f2(xx))

     dput(jacobian(f2, xx))
     */
    arma::vec expec{0.368125090148117, -0.0972709187968055, -0.0388955601299048, 0.00622070279491852, -0.00824456594424145, 0.00271816571363873, 0.0108131821534993, -0.00678049379010119, 0.00196371803414583, 0.0106758132172105, -0.00648455781045827, 0.000897181658087495, -0.00184020017999673};
    arma::uword const ex_dim{1L + n + (p * (p + 1L)) / 2L};

    mix_binary bin(y, eta, Z, S);
    set_integrand(std::unique_ptr<base_integrand>(
        new mix_binary(y, eta, Z, S)));
    auto run_test = [&](int const key){
      auto res = jac_arpx(100000000L, key, -1, rel_eps);
      expect_true(res.value.n_elem == ex_dim);
      expect_true(res.err.n_elem == ex_dim);

      expect_true(res.inform == 0L);
      for(unsigned i = 0; i < res.err.n_elem; ++i){
        expect_true(res.err[i] < 2 * rel_eps * std::abs(res.value[i]));
        expect_true
          (std::abs(res.value[i] - expec[i]) <
            2 * rel_eps * std::abs(expec[i]));
      }
    };
    run_test(2);

    /* test w/ adaptive method */
    mvn<mix_binary> m(bin);

    set_integrand(std::unique_ptr<base_integrand>(
        new adaptive<mvn<mix_binary > > (m, true)));
    run_test(1);
    run_test(2);
    run_test(3);
    run_test(4);
  }
}
