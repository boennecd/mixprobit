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
    sig_adaptive <- solve(-hessian(f, mode, method.args = list(eps = 1e-10)))
    dput(chol(sig_adaptive))

    x <- c(-1, 1)
    dput(f(mode + x %*% chol(sig_adaptive)) - mvtnorm::dmvnorm(x, log = TRUE) +
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
    adaptive<mvn<integrand::mix_binary> > ada(
        mvn_bin, 10000L, -1., eps * eps);

    arma::vec mode_ex;
    mode_ex << -0.275739778642164 << 0.0192264549867938;
    arma::mat neg_hes_inv_chol;
    neg_hes_inv_chol << 0.680968197783783 << 0
                     << -0.0223120126667078 << 0.990551648427405;

    for(unsigned i = 0; i < p; ++i)
      expect_true(std::abs(
          (mode_ex[i] - ada.dat.mode[i]) / mode_ex[i]) < eps);

    for(unsigned i = 0; i < p * p; ++i)
      if(ada.dat.neg_hes_inv_chol[i] == 0.)
        expect_true(neg_hes_inv_chol[i] == 0.);
      else
        expect_true(std::abs(
            (neg_hes_inv_chol[i] - ada.dat.neg_hes_inv_chol[i]) /
              neg_hes_inv_chol[i]) < eps);

    arma::vec x;
    x << -1 << 1;

    double const expec_val = -5.2947411042184;
    expect_true(std::abs(
        (expec_val - ada(x.memptr(), true)) / expec_val) < eps);
  }
}
