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
      auto const res = gsm_cens_term(Zo, Zc, Xo, Xc, beta, Sigma).func
        (10000L, 1, 1e-3, 1e-3, gsm_approx_method::adaptive_spherical_radial);
      expect_true(res.inform == 0);
      expect_true(res.log_like == 0);
    }

    arma::vec gr;
    auto const res = gsm_cens_term(Zo, Zc, Xo, Xc, beta, Sigma)
      .gr(gr, 10000L, 1, 1e-3, 1e-3,
          gsm_approx_method::adaptive_spherical_radial);
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

     u <- drop(rnorm(p) %*% chol(S))      # random effects

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
     S <- S + drop(rWishart(1, p, diag(1e-1, p)))
     dput(S)
     xx <- c(beta, S[upper.tri(S, TRUE)])
     dput(log(f1(xx)))

     dput(numDeriv::jacobian(\(x) log(f1(x)), xx))
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
           Sigma{2.39285683640521, 0.232426366938143, 0.21565518669603, 0.232426366938143, 1.54762667750445, 0.856575189367694, 0.21565518669603, 0.856575189367694, 4.73047421966608};
    Xc.reshape(n_fixef, n_cens);
    Zc.reshape(n_rng, n_cens);
    Sigma.reshape(n_rng, n_rng);

    arma::vec const beta{-2, 2};

    gsm_cens_term comp_obj(Zo, Zc, Xo, Xc, beta, Sigma);

    constexpr double true_func{-0.968031924910162},
                       rel_eps{1e-1};
    {
      auto const res =
        comp_obj.func(100000, 3L, -1, rel_eps,
                      gsm_approx_method::adaptive_spherical_radial);
      expect_true(res.inform == 0);
      expect_true(std::abs(res.log_like - true_func) <
        2 * rel_eps * std::abs(true_func));
    }
    {
      auto const res = comp_obj.func(100000, 3L, -1, rel_eps,
                                     gsm_approx_method::spherical_radial);
      expect_true(res.inform == 0);
      expect_true(std::abs(res.log_like - true_func) <
        2 * rel_eps * std::abs(true_func));
    }
    {
      auto const res = comp_obj.func(100000, 3L, -1, rel_eps,
                                     gsm_approx_method::cdf_approach);
      expect_true(res.inform == 0);
      expect_true(std::abs(res.log_like - true_func) <
        2 * rel_eps * std::abs(true_func));
    }

    // check the gradient
    constexpr uword dim_gr{n_fixef + (n_rng * (n_rng + 1)) / 2};
    constexpr double true_gr[dim_gr]
      {-0.537893597021461, -0.208299165140343, 0.0401056616115828, 0.0163544243417218, -0.0629916831813027, 0.0169226034005708, 0.023017755737681, -0.0484885478113118};

    arma::vec gr;
    {
      auto const res = comp_obj.gr(gr, 10000000, 3L, -1, rel_eps,
                                   gsm_approx_method::adaptive_spherical_radial);
      expect_true(res.inform == 0);
      expect_true(std::abs(res.log_like - true_func) <
        2 * rel_eps * std::abs(true_func));

      expect_true(gr.size() == dim_gr);
      for(uword i = 0; i < gr.size(); ++i)
        expect_true(std::abs(gr[i] - true_gr[i]) <
          2 * rel_eps * std::abs(true_gr[i]));
    }

    auto const res = comp_obj.gr(gr, 10000000, 3L, -1, rel_eps,
                                 gsm_approx_method::cdf_approach);
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

     u <- drop(rnorm(p) %*% chol(S))       # random effects


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
     n_obs <- sum(event)
     h <- S %*% Zo %*%
     solve(diag(n_obs) + crossprod(Zo, S %*% Zo),- Xo %*% b)
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
     S <- S + drop(rWishart(1, p, diag(1e-1, p)))
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
           Sigma{4.9986403657076, -0.252584938139645, -1.29848636512648, -0.252584938139645, 1.4893648315846, -0.686142763137388, -1.29848636512648, -0.686142763137388, 2.9647627980052};
    Xc.reshape(n_fixef, n_cens);
    Xo.reshape(n_fixef, n_obs);
    Zc.reshape(n_rng, n_cens);
    Zo.reshape(n_rng, n_obs);
    Sigma.reshape(n_rng, n_rng);

    arma::vec const beta{-2.5, 2};

    gsm_cens_term comp_obj(Zo, Zc, Xo, Xc, beta, Sigma);

    constexpr double true_func{-0.724841472478689},
                       rel_eps{1e-1};
    {
      auto const res =
        comp_obj.func(100000, 3L, -1, rel_eps,
                      gsm_approx_method::adaptive_spherical_radial);
      expect_true(res.inform == 0);
      expect_true(std::abs(res.log_like - true_func) <
        2 * rel_eps * std::abs(true_func));
    }
    {
      auto const res = comp_obj.func(100000, 3L, -1, rel_eps,
                                     gsm_approx_method::spherical_radial);
      expect_true(res.inform == 0);
      expect_true(std::abs(res.log_like - true_func) <
        2 * rel_eps * std::abs(true_func));
    }
    {
      auto const res = comp_obj.func(100000, 3L, -1, rel_eps,
                                     gsm_approx_method::cdf_approach);
      expect_true(res.inform == 0);
      expect_true(std::abs(res.log_like - true_func) <
        2 * rel_eps * std::abs(true_func));
    }

    // check the gradient
    constexpr uword dim_gr{n_fixef + (n_rng * (n_rng + 1)) / 2};
    constexpr double true_gr[dim_gr]
      {-0.0977871634383885, -0.138326698988711, -0.0483815801503183, 0.0137908164075206, 0.00183664253847172, -0.0852235200667964, -0.00871301650391389, 0.00105158098124027};

    arma::vec gr;
    {
      auto const res = comp_obj.gr(gr, 10000000, 3L, -1, rel_eps,
                                   gsm_approx_method::adaptive_spherical_radial);
      expect_true(res.inform == 0);
      expect_true(std::abs(res.log_like - true_func) <
        2 * rel_eps * std::abs(true_func));

      expect_true(gr.size() == dim_gr);
      for(uword i = 0; i < gr.size(); ++i)
        expect_true(std::abs(gr[i] - true_gr[i])  <
          2 * rel_eps * std::abs(true_gr[i]));
    }

    auto const res = comp_obj.gr(gr, 10000000, 3L, -1, rel_eps,
                                 gsm_approx_method::cdf_approach);
    expect_true(res.inform == 0);
    expect_true(std::abs(res.log_like - true_func) <
      2 * rel_eps * std::abs(true_func));

    expect_true(gr.size() == dim_gr);
    for(uword i = 0; i < gr.size(); ++i)
      expect_true(std::abs(gr[i] - true_gr[i])  <
        2 * rel_eps * std::abs(true_gr[i]));
  }

  test_that("gsm_cens_term gives the right result with both censored and observed individuals (two)") {
    /*
     set.seed(15L)
     n_fixef <- 2L
     Xt  <- \(ti) { ti <- log(ti); cbind(1, ti) }
     beta <- c(-.5, 2)

     n <- 10L
     p <- 3L
     Z <- do.call(                        # random effect design matrix
     rbind, c(list(1), list(replicate(n, runif(p - 1L, -1, 1)))))
     n <- NCOL(Z)                         # number of individuals
     p <- NROW(Z)                         # number of random effects
     S <- drop(                           # covariance matrix of random effects
     rWishart(1, 2 * p, diag(sqrt(1/ 4 / p), p)))

     S <- round(S, 3)
     Z <- round(Z, 3)

     u <- drop(rnorm(p) %*% chol(S))       # random effects

# get the outcomes
     rngs <- runif(n)
     y <- mapply(\(i, rng){
     offset <- -sum(u %*% Z[, i])
     f <- \(ti) pnorm(-sum(Xt(ti) * beta) - offset) - rng
     uniroot(f, c(1e-20, 1000000), tol = 1e-8)$root
     }, i = 1:n, rng = rngs)
     print(y)
     cens <- runif(n, 0, 4)
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
     n_obs <- sum(event)
     h <- S %*% Zo %*%
     solve(diag(n_obs) + crossprod(Zo, S %*% Zo),- Xo %*% b)
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
     S <- S + drop(rWishart(1, p, diag(1e-1, p)))
     dput(S)
     xx <- c(beta, S[upper.tri(S, TRUE)])

     dput(log(f1(xx)))
     dput(numDeriv::jacobian(\(x) log(f1(x)), xx))
     */
    constexpr uword n_fixef{2},
                     n_cens{2},
                      n_obs{8},
                      n_rng{3};

    Rcpp::RNGScope rngScope;
    parallelrng::set_rng_seeds(1L);

    arma::mat Xc{1, -0.008, 1, 0.09},
              Zc{1, 0.374, 0.663, 1, -0.718, 0.553},
              Xo{1, -0.443, 1, -0.218, 1, -0.417, 1, 0.321, 1, 0.754, 1, 0.43, 1, -0.552, 1, -0.297},
              Zo{1, 0.204, -0.61, 1, 0.933, 0.302, 1, -0.266, 0.978, 1, 0.63, -0.492, 1, -0.791, 0.292, 1, 0.018, 0.413, 1, 0.725, 0.684, 1, -0.105, 0.929},
           Sigma{2.44716740312943, 0.511163129975597, 0.144036394679127, 0.511163129975597, 1.38613519465667, 0.273917200473281, 0.144036394679127, 0.273917200473281, 0.521801023030326};
    Xc.reshape(n_fixef, n_cens);
    Xo.reshape(n_fixef, n_obs);
    Zc.reshape(n_rng, n_cens);
    Zo.reshape(n_rng, n_obs);
    Sigma.reshape(n_rng, n_rng);

    arma::vec const beta{-0.5, 2};

    gsm_cens_term comp_obj(Zo, Zc, Xo, Xc, beta, Sigma);

    constexpr double true_func{-1.502700530338},
                       rel_eps{1e-1};
    {
      auto const res =
        comp_obj.func(100000, 3L, -1, rel_eps,
                      gsm_approx_method::adaptive_spherical_radial);
      expect_true(res.inform == 0);
      expect_true(std::abs(res.log_like - true_func) <
        2 * rel_eps * std::abs(true_func));
    }
    {
      auto const res = comp_obj.func(100000, 3L, -1, rel_eps,
                                     gsm_approx_method::spherical_radial);
      expect_true(res.inform == 0);
      expect_true(std::abs(res.log_like - true_func) <
        2 * rel_eps * std::abs(true_func));
    }

    // check the gradient
    constexpr uword dim_gr{n_fixef + (n_rng * (n_rng + 1)) / 2};
    constexpr double true_gr[dim_gr]
    {-0.0730728351317208, -0.116149544771193, -0.00285788922175744, -0.0200031469366203, 0.0328085194439066, -0.0357818568455293, -0.0442357970784554, -0.096957277588603};

    arma::vec gr;
    auto const res = comp_obj.gr(gr, 100000000, 1L, -1, rel_eps,
                                 gsm_approx_method::adaptive_spherical_radial);
    // expect_true(res.inform == 0);
    expect_true(std::abs(res.log_like - true_func) <
      2 * rel_eps * std::abs(true_func));

    expect_true(gr.size() == dim_gr);
    for(uword i = 0; i < gr.size(); ++i)
      expect_true(std::abs(gr[i] - true_gr[i])  <
        2 * rel_eps * std::abs(true_gr[i]));
  }

  test_that("gsm_normal_term gives the right result"){
    /*
     set.seed(1)
     n <- 7L
     p <- 3L
     q <- 4L
     X <- rnorm(n * p) |> matrix(n)
     Z <- rnorm(n * q) |> matrix(n)
     Sigma <- drop(rWishart(1, q, diag(q)))
     beta <- runif(p)

     f1 <- \(x){
     b <- head(x, p)
     Sv <- tail(x, -p)
     S <- matrix(0, q, q)
     S[upper.tri(S, TRUE)] <- Sv
     S[lower.tri(S)] <- t(S)[lower.tri(S)]

     v <- drop(-X %*% b)
     K <- tcrossprod(Z %*% S, Z) + diag(n)

     -n/2 * log(2 * pi) - log(det(K)) / 2 -
     drop(v %*% solve(K, v)) / 2
     }

     dput(beta)
     dput(Sigma)
     dput(t(X))
     dput(t(Z))

     xx <- c(beta, Sigma[upper.tri(Sigma, TRUE)])
     dput(f1(xx))
     dput(numDeriv::grad(f1, xx))
     */

    constexpr arma::uword n_obs{7}, n_fixef{3}, n_rng{4};

    arma::vec beta{0.484349524369463, 0.173442334868014, 0.754820944508538};
    arma::mat Sigma{5.54643975104526, -1.44137604864076, -2.6597495383547, 1.34173988980172, -1.44137604864076, 3.2500322127611, 3.12120257444617, -0.577697827024381, -2.6597495383547, 3.12120257444617, 4.86961534836983, 2.14395051943116, 1.34173988980172, -0.577697827024381, 2.14395051943116, 6.46733555813455};
    Sigma.reshape(n_rng, n_rng);
    arma::mat X{-0.626453810742332, 0.738324705129217, 1.12493091814311, 0.183643324222082, 0.575781351653492, -0.0449336090152309, -0.835628612410047, -0.305388387156356, -0.0161902630989461, 1.59528080213779, 1.51178116845085, 0.943836210685299, 0.329507771815361, 0.389843236411431, 0.821221195098089, -0.820468384118015, -0.621240580541804, 0.593901321217509, 0.487429052428485, -2.2146998871775, 0.918977371608218};
    X.reshape(n_fixef, n_obs);
    arma::mat Z{0.782136300731067, -0.47815005510862, -0.41499456329968, 0.696963375404737, 0.0745649833651906, 0.417941560199702, -0.394289953710349, 0.556663198673657, -1.98935169586337, 1.35867955152904, -0.0593133967111857, -0.68875569454952, 0.61982574789471, -0.102787727342996, 1.10002537198388, -0.70749515696212, -0.0561287395290008, 0.387671611559369, 0.763175748457544, 0.36458196213683, -0.155795506705329, -0.0538050405829051, -0.164523596253587, 0.768532924515416, -1.47075238389927, -1.37705955682861, -0.253361680136508, -0.112346212150228};
    Z.reshape(n_rng, n_obs);

    constexpr double true_ll{-13.0167247474452},
                     true_gr[]{-3.12590270951561, -1.55572984168536, -3.01735085015094, 0.000783566612432911, -0.624444068082098, 0.572994318260551, 0.58488391380321, -1.58662969802432, 0.8521242345315, -0.223216749590719, 0.673330365259303, -0.73147915674766, 0.0836169280373346};

    gsm_normal_term comp_obj(X, Z, Sigma, beta);

    double const eps{std::pow(std::numeric_limits<double>::epsilon(), 3. / 5.)};
    expect_true((comp_obj.func() - true_ll) < std::abs(true_ll) * eps);

    arma::vec gr;
    expect_true((comp_obj.gr(gr) - true_ll) < std::abs(true_ll) * eps);
    constexpr arma::uword n_par = n_fixef + (n_rng * (n_rng + 1)) / 2;
    expect_true(gr.n_elem == n_par);

    for(arma::uword i = 0; i < n_par; ++i)
      expect_true((gr[i] - true_gr[i]) < std::abs(true_gr[i]) * eps);
  }
}
