#include "gsm.h"
#include <testthat.h>
#include <limits>
#include "threat-safe-random.h"

using namespace arma;

context("testing gsm_cens_term") {
  test_that("gsm_cens_term gives the right result w/o censored individuals"){
    constexpr uword n_obs{4},
                    n_rng{3},
                    n_fix{4};
    arma::mat Xo(n_fix, n_obs),
              Zo(n_fix, n_obs),
              Xc(n_fix, 0),
              Zc(n_rng, 0),
           Sigma(n_rng, n_rng);
    arma::vec beta(n_fix);

    {
      auto const res = gsm_cens_term(Zo, Zc, Xo, Xc, beta, Sigma)
        .func(10000L, 1, 1e-3, 1e-3);
      expect_true(res.inform == 0);
      expect_true(res.log_like == 0);
    }

    arma::vec gr;
    auto const res = gsm_cens_term(Zo, Zc, Xo, Xc, beta, Sigma)
      .gr(gr, 10000L, 1, 1e-3, 1e-3);
    expect_true(res.inform == 0);
    expect_true(res.log_like == 0);

    constexpr uword dim_gr{n_fix + (n_rng * (n_rng + 1)) / 2};
    expect_true(gr.size() == dim_gr);
    for(double g : gr)
      expect_true(g == 0);
  }

  test_that("gsm_cens_term gives the right result with only censored individuals") {
    /*
     set.seed(2)
     n_fixef <- 2L
     Xt  <- \(ti) { ti <- log(ti); cbind(1, ti) }
     beta <- c(-2, 2)

     n <- 4L
     p <- 3L
     Z <- do.call(                        # random effect design matrix
     rbind, c(list(1), list(replicate(n, runif(p - 1L, -1, 1)))))
     n <- NCOL(Z)                         # number of individuals
     p <- NROW(Z)                         # number of random effects
     S <- drop(                           # covariance matrix of random effects
     rWishart(1, 2 * p, diag(sqrt(1/ 2 / p), p)))

     S <- round(S, 3)
     Z <- round(Z, 3)

     S_chol <- chol(S)
     u <- drop(rnorm(p) %*% S_chol)       # random effects


# get the outcomes
     rngs <- runif(n)
     y <- mapply(\(i, rng){
     offset <- -sum(u %*% Z[, i])
     f <- \(ti) pnorm(-sum(Xt(ti) * beta) - offset) - rng
     uniroot(f, c(1e-20, 1000000), tol = 1e-8)$root
     }, i = 1:n, rng = rngs)
     cens <- runif(n, 0, 2)
     event <- cens < y
     stopifnot(all(event))

     ti <- pmin(cens, y) |> round(3)
     Xc <- Xt(ti) |> round(3)
     Zc <- Z

#####
# use GH quadrature
     library(fastGHQuad)
     b <- 25L                             # number of nodes to use
     rule <- fastGHQuad::gaussHermiteData(b)
     f <- function(x, eta, S_chol)
     sum(mapply(pnorm, q = eta + sqrt(2) * drop(x %*% S_chol %*% Zc),
     lower.tail = TRUE, log.p = TRUE))
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
     b <- head(par, n_fixef)
     s <- tail(par, -n_fixef)
     S <- matrix(nr = p, nc = p)
     S[upper.tri(S, TRUE)] <- s
     S[lower.tri(S)] <- t(S)[lower.tri(S)]

     Sigma <- S
     eta <- -drop(Xc %*% b)
     sum(exp(ws_log + vapply(
     xs, f, numeric(1L), eta = eta, S_chol = chol(Sigma)))) /
     pi^(p / 2)
     }

     dput(t(Xc))
     dput(Zc)
     dput(beta)
     dput(S)
     xx <- c(beta, S[upper.tri(S, TRUE)])
     dput(log(f1(xx)))

     dput(jacobian(\(x) log(f1(x)), xx))
     */
    constexpr uword n_fixef{2},
                    n_cens{4},
                    n_obs{0},
                    n_rng{3};

    Rcpp::RNGScope rngScope;
    parallelrng::set_rng_seeds(1L);

    arma::mat Xc{1, 0.482, 1, 0.553, 1, 0.029, 1, 0.226},
              Zc{1, -0.63, 0.405, 1, 0.147, -0.664, 1, 0.888, 0.887, 1, -0.742, 0.667},
              Xo(n_fixef, n_obs),
              Zo(n_rng, n_obs),
           Sigma{1.939, -0.213, -0.123, -0.213, 0.94, 0.269, -0.123, 0.269, 4.092};
    Xc.reshape(n_fixef, n_cens);
    Zc.reshape(n_rng, n_cens);
    Sigma.reshape(n_rng, n_rng);

    arma::vec const beta{-2, 2};

    gsm_cens_term comp_obj(Zo, Zc, Xo, Xc, beta, Sigma);

    constexpr double true_func{-0.968031924910162},
                       rel_eps{5e-1};
    {
      auto const res = comp_obj.func(100000, 3L, -1, rel_eps);
      expect_true(res.inform == 0);
      expect_true(std::abs(res.log_like - true_func) <
        2 * rel_eps * std::abs(true_func));
    }

    // check the gradient
    constexpr uword dim_gr{n_fixef + (n_rng * (n_rng + 1)) / 2};
    constexpr double true_gr[dim_gr]
      {-0.572042610291542, -0.22093307695183, 0.042503945328741, -0.000902905965226302, -0.0762413721340899, 0.0114883138032356, 0.0125929452207209, -0.0545246015073341};

    arma::vec gr;
    auto const res = comp_obj.gr(gr, 10000000, 3L, -1, rel_eps);
    expect_true(res.inform == 0);
    expect_true(std::abs(res.log_like - true_func) <
      2 * rel_eps * std::abs(true_func));

    expect_true(gr.size() == dim_gr);
    for(uword i = 0; i < gr.size(); ++i)
      expect_true(std::abs(gr[i] - true_gr[i]) <
        2 * rel_eps * std::abs(true_gr[i]));
  }

  test_that("gsm_cens_term gives the right result with both censored and observed individuals") {
    /*
     set.seed(12L)
     n_fixef <- 2L
     Xt  <- \(ti) { ti <- log(ti); cbind(1, ti) }
     beta <- c(-2.5, 2)

     n <- 4L
     p <- 3L
     Z <- do.call(                        # random effect design matrix
     rbind, c(list(1), list(replicate(n, runif(p - 1L, -1, 1)))))
     n <- NCOL(Z)                         # number of individuals
     p <- NROW(Z)                         # number of random effects
     S <- drop(                           # covariance matrix of random effects
     rWishart(1, 2 * p, diag(sqrt(1/ 2 / p), p)))

     S <- round(S, 3)
     Z <- round(Z, 3)

     S_chol <- chol(S)
     u <- drop(rnorm(p) %*% S_chol)       # random effects


# get the outcomes
     rngs <- runif(n)
     y <- mapply(\(i, rng){
     offset <- -sum(u %*% Z[, i])
     f <- \(ti) pnorm(-sum(Xt(ti) * beta) - offset) - rng
     uniroot(f, c(1e-20, 1000000), tol = 1e-8)$root
     }, i = 1:n, rng = rngs)
     cens <- runif(n, 0, 2)
     event <- y < cens
     stopifnot(any(event), any(!event))

     ti <- pmin(cens, y) |> round(3)
     X <- Xt(ti) |> round(3)
     Xc <- X[!event, , drop = FALSE]
     Xo <- X[ event, , drop = FALSE]
     Zc <- Z[, !event, drop = FALSE]
     Zo <- Z[,  event, drop = FALSE]

#####
# use GH quadrature
     library(fastGHQuad)
     b <- 25L                             # number of nodes to use
     rule <- fastGHQuad::gaussHermiteData(b)
     f <- function(x, eta, S_chol)
     sum(mapply(pnorm, q = eta + sqrt(2) * drop(x %*% S_chol %*% Zc),
     lower.tail = TRUE, log.p = TRUE))
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
     b <- head(par, n_fixef)
     s <- tail(par, -n_fixef)
     S <- matrix(nr = p, nc = p)
     S[upper.tri(S, TRUE)] <- s
     S[lower.tri(S)] <- t(S)[lower.tri(S)]

     H <- tcrossprod(Zo) + solve(S)
     h <- solve(H, Zo %*% (- Xo %*% b))
     Sigma <- solve(H)
     eta <- -drop(Xc %*% b) - drop(crossprod(Zc, h))
     sum(exp(ws_log + vapply(
     xs, f, numeric(1L), eta = eta, S_chol = chol(Sigma)))) /
     pi^(p / 2)
     }

     dput(t(Xc))
     dput(t(Xo))
     dput(Zc)
     dput(Zo)
     dput(beta)
     dput(S)
     xx <- c(beta, S[upper.tri(S, TRUE)])

     dput(log(f1(xx)))
     dput(numDeriv::jacobian(\(x) log(f1(x)), xx))
     */
    constexpr uword n_fixef{2},
                     n_cens{1},
                      n_obs{3},
                      n_rng{3};

    Rcpp::RNGScope rngScope;
    parallelrng::set_rng_seeds(1L);

    arma::mat Xc{1, 0.237},
              Zc{1, -0.861, 0.636},
              Xo{1, -0.242, 1, -0.309, 1, -0.051},
              Zo{1, 0.885, -0.461, 1, -0.661, -0.932, 1, -0.642, 0.283},
           Sigma{4.853, -0.15, -1.095, -0.15, 1.306, -0.909, -1.095, -0.909, 2.621};
    Xc.reshape(n_fixef, n_cens);
    Xo.reshape(n_fixef, n_obs);
    Zc.reshape(n_rng, n_cens);
    Zo.reshape(n_rng, n_obs);
    Sigma.reshape(n_rng, n_rng);

    arma::vec const beta{-2.5, 2};

    gsm_cens_term comp_obj(Zo, Zc, Xo, Xc, beta, Sigma);

    constexpr double true_func{-0.73433709985276},
                       rel_eps{5e-1};
    {
      auto const res = comp_obj.func(100000, 3L, -1, rel_eps);
      expect_true(res.inform == 0);
      expect_true(std::abs(res.log_like - true_func) <
        2 * rel_eps * std::abs(true_func));
    }

    // check the gradient
    constexpr uword dim_gr{n_fixef + (n_rng * (n_rng + 1)) / 2};
    constexpr double true_gr[dim_gr]
      {-0.0969333728004924, -0.142173776625063, -0.0488113206428778, 0.00972943379091768, 0.000342446867236928, -0.0903543460452184, -0.00366361757944606, 0.00668709162478186};

    arma::vec gr;
    auto const res = comp_obj.gr(gr, 10000000, 3L, -1, rel_eps);
    expect_true(res.inform == 0);
    expect_true(std::abs(res.log_like - true_func) <
      2 * rel_eps * std::abs(true_func));

    expect_true(gr.size() == dim_gr);
    for(uword i = 0; i < gr.size(); ++i)
      expect_true(std::abs(gr[i] - true_gr[i]) <
        2 * rel_eps * std::abs(true_gr[i]));
  }
}
