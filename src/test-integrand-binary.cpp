#include "integrand-binary.h"
#include <testthat.h>
#include <limits>

context("integrand-binary unit tests") {
  test_that("mix_binary gives the correct value, gradient, and Hessian") {
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
    rWishart(1, p, diag(1/ 2 / p, p)))

    S   <- round(S  , 3)
    Z   <- round(Z  , 3)
    eta <- round(eta, 3)

    S_chol <- chol(S)
    u <- drop(rnorm(p) %*% S_chol)       # random effects
    y <- runif(n) < pnorm(drop(u %*% Z)) # observed outcomes

    f <- function(pt){
    lp <- drop(pt %*% chol(S) %*% Z) + eta
    sum(ifelse(y,
               pnorm(lp, lower.tail = TRUE , log.p = TRUE),
    pnorm(lp, lower.tail = FALSE, log.p = TRUE)))
    }

    dput(c(S))
    dput(eta)
    dput(c(Z))
    dput(as.integer(y))

    dput(f(c(-1, 1)))
    library(numDeriv)
    dput(jacobian(f, c(-1, 1), method.args = list(eps = 1e-10)))
    dput(hessian(f, c(-1, 1), method.args = list(eps = 1e-10)))

    f2 <- function(pt){
    lp <- drop(pt %*% chol(S) %*% Z) + eta
    sum(ifelse(y,
               pnorm(lp, lower.tail = TRUE , log.p = TRUE),
    pnorm(lp, lower.tail = FALSE, log.p = TRUE))) +
    sum(dnorm(pt, log = TRUE))
    }

    dput(f2(c(-1, 1)))
    dput(jacobian(f2, c(-1, 1), method.args = list(eps = 1e-10)))
    dput(hessian(f2, c(-1, 1), method.args = list(eps = 1e-10)))
 */
    arma::uword const p = 2L,
                      n = 4L;
    arma::mat S;
    S << 0.38 << 0.15 << 0.15 << 0.089;
    S.reshape(p, p);

    arma::vec eta;
    eta << -0.597 << 0.797 << 0.889 << 0.322;

    arma::mat Z;
    Z << 1 << -0.469 << 1 << -0.256 << 1 << 0.146 << 1 << 0.816;
    Z.reshape(p, n);

    arma::ivec y;
    y << 1L << 0L << 0L << 1L;

    arma::vec x;
    x << -1 << 1;

    integrand::mix_binary bin(y, eta, Z, S);

    double const eps = std::sqrt(std::numeric_limits<double>::epsilon());

    {
      double const ex_log_val = -4.93359712901787;
      expect_true(
        std::abs((ex_log_val - bin(x.memptr(), true)) / ex_log_val) < eps);

      arma::vec gr_ex;
      gr_ex << 0.534644758277354 << 0.0269771122308241;
      arma::vec const gr = bin.gr(x.memptr());

      for(unsigned i = 0; i < p; i++)
        expect_true(
          std::abs((gr_ex[i] - gr[i]) / gr_ex[i]) < 10 * eps);

      arma::mat he_ex;
      he_ex << -1.17758907347171 << -0.0424437547172116 << -0.0424437547172116
            << -0.0211494558975862;
      he_ex.reshape(p, p);
      arma::mat const he = bin.Hessian(x.memptr());
      for(unsigned i = 0; i < p * p; i++)
        expect_true(
          std::abs((he_ex[i] - he[i]) / he_ex[i]) < 10 * eps);
    }

    integrand::mvn<integrand::mix_binary> bin_mvn(bin);

    {
      double const ex_log_val = -7.77147419542722;
      expect_true(std::abs(
          (ex_log_val - bin_mvn(x.memptr(), true)) / ex_log_val) < eps);

      arma::vec gr_ex;
      gr_ex << 1.53464475832635 << -0.973022887766845;
      arma::vec const gr = bin_mvn.gr(x.memptr());

      for(unsigned i = 0; i < p; i++)
        expect_true(
          std::abs((gr_ex[i] - gr[i]) / gr_ex[i]) < 10 * eps);

      arma::mat he_ex;
      he_ex << -2.17758907347016 << -0.0424437547254297
            << -0.0424437547254297 << -1.02114945589602;
      he_ex.reshape(p, p);
      arma::mat const he = bin_mvn.Hessian(x.memptr());
      for(unsigned i = 0; i < p * p; i++)
        expect_true(
          std::abs((he_ex[i] - he[i]) / he_ex[i]) < 10 * eps);
    }
  }

  test_that("adaptive has the correct data set after calling the constructor"){
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
    rWishart(1, p, diag(1/ 2 / p, p)))

    S   <- round(S  , 3)
    Z   <- round(Z  , 3)
    eta <- round(eta, 3)

    S_chol <- chol(S)
    u <- drop(rnorm(p) %*% S_chol)       # random effects
    y <- runif(n) < pnorm(drop(u %*% Z)) # observed outcomes

    f <- function(pt){
    lp <- drop(pt %*% chol(S) %*% Z) + eta
    sum(ifelse(y,
               pnorm(lp, lower.tail = TRUE , log.p = TRUE),
    pnorm(lp, lower.tail = FALSE, log.p = TRUE)))
    }

    dput(c(S))
    dput(eta)
    dput(c(Z))
    dput(as.integer(y))

    f <- function(pt){
    lp <- drop(pt %*% chol(S) %*% Z) + eta
    sum(ifelse(y,
               pnorm(lp, lower.tail = TRUE , log.p = TRUE),
    pnorm(lp, lower.tail = FALSE, log.p = TRUE))) +
    sum(dnorm(pt, log = TRUE))
    }

    dput(mode <- optim(c(0, 0), function(x) -f(x), control = list(reltol = 1e-12))$par)
    Hes <- hessian(f, mode, method.args = list(eps = 1e-10))
    sig_adaptive <- solve(-Hes)
    dput(half_mat <- chol(sig_adaptive))

    x <- c(-1, 1)
    dput(f(mode + x %*% half_mat) - mvtnorm::dmvnorm(x, log = TRUE) +
    determinant(sig_adaptive)$modulus / 2)

    eg1 <- eigen(sig_adaptive)
    eg2 <- eigen(-Hes)

    tcrossprod(eg1$vectors %*% diag(sqrt(eg1$values))) - sig_adaptive
    tcrossprod(eg2$vectors %*% diag(sqrt(eg2$values^(-1)))) - sig_adaptive
    (t(eg2$vectors[, 2:1]) - eg1$vectors) /  eg1$vectors

    tmp <- t(eg2$vectors)
    dput(half_mat <- (eg2$vectors %*% diag(eg2$values^(-1/2)))[, 2:1])
    tcrossprod(half_mat) - sig_adaptive

    x <- c(-1, 1)
    dput(f(drop(mode + half_mat %*% x)) - mvtnorm::dmvnorm(x, log = TRUE) +
    determinant(sig_adaptive)$modulus / 2)
    */
    using namespace integrand;
    arma::uword const p = 2L,
                      n = 4L;
    arma::mat S;
    S << 0.38 << 0.15 << 0.15 << 0.089;
    S.reshape(p, p);

    arma::vec eta;
    eta << -0.597 << 0.797 << 0.889 << 0.322;

    arma::mat Z;
    Z << 1 << -0.469 << 1 << -0.256 << 1 << 0.146 << 1 << 0.816;
    Z.reshape(p, n);

    arma::ivec y;
    y << 1L << 0L << 0L << 1L;

    mix_binary bin(y, eta, Z, S);
    mvn<mix_binary> mvn_bin(bin);
    double const eps = std::pow(
      std::numeric_limits<double>::epsilon(), 1. / 4.);
    /* w/ Cholesky */
    {
      adaptive<mvn<integrand::mix_binary> > ada(
          mvn_bin, true, 10000L, -1., eps * eps);

      arma::vec mode_ex;
      mode_ex << -0.275739778642164 << 0.0192264549867938;
      arma::mat neg_hes_inv_half;
      neg_hes_inv_half << 0.680968197783783 << 0
                       << -0.0223120126667078 << 0.990551648427405;
      neg_hes_inv_half.reshape(2L, 2L);

      for(unsigned i = 0; i < p; ++i)
        expect_true(std::abs(
            (mode_ex[i] - ada.dat.mode[i]) / mode_ex[i]) < eps);

      for(unsigned i = 0; i < p * p; ++i)
        if(ada.dat.neg_hes_inv_half[i] == 0.)
          expect_true(neg_hes_inv_half[i] == 0.);
        else
          expect_true(std::abs(
              (neg_hes_inv_half[i] - ada.dat.neg_hes_inv_half[i]) /
                neg_hes_inv_half[i]) < eps);

      arma::vec x;
      x << -1 << 1;

      double const expec_val = -5.2947411042184;
      expect_true(std::abs(
          (expec_val - ada(x.memptr(), true)) / expec_val) < eps);
    }

    /* w/ Eigen */
    {
      adaptive<mvn<integrand::mix_binary> > ada(
          mvn_bin, false, 10000L, -1., eps * eps);

      arma::vec mode_ex;
      mode_ex << -0.275739778642164 << 0.0192264549867938;
      arma::mat neg_hes_inv_half;
      neg_hes_inv_half << 0.0290325249486965 << -0.990602243561544
                       << -0.680349027255861 << -0.0199396379687283;
      neg_hes_inv_half.reshape(2L, 2L);

      for(unsigned i = 0; i < p; ++i)
        expect_true(std::abs(
            (mode_ex[i] - ada.dat.mode[i]) / mode_ex[i]) < eps);

      for(unsigned i = 0; i < p * p; ++i)
        expect_true(std::abs(
            (neg_hes_inv_half[i] - ada.dat.neg_hes_inv_half[i]) /
              neg_hes_inv_half[i]) < eps);

        arma::vec x;
        x << -1 << 1;

        double const expec_val = -5.29494527011595;
        expect_true(std::abs(
            (expec_val - ada(x.memptr(), true)) / expec_val) < eps);
    }
  }
}
