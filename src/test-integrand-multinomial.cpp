#include "integrand-multinomial.h"
#include <testthat.h>
#include <limits>

using namespace integrand;
using std::log;
using std::abs;

context("integrand-multinomial unit tests") {
  test_that("multinomial mode and second order derivative is correct"){
    /*
     eta <- numeric(3)
     z <- matrix(c(1:3, 3:1), nc = 2)
     Sig[1, 2] <- Sig[2, 1] <- .5
     z <- tcrossprod(z, chol(Sig))
     u <- numeric(2)

     tfunc <- function(){
     f <- Vectorize(function(a)
     -dnorm(a, log = TRUE) - sum(pnorm(a + eta - z %*% u, log.p = TRUE)),
     vectorize.args = "a")
     opt <- optimize(f, c(-1e10, 1e10))
     library(numDeriv)
     sc <- sqrt(1 / hessian(f, opt$minimum))

     unname(c(opt$minimum, sc))
     }
     dput(tfunc())
     u[1] <- 1
     u[2] <- -1
     dput(tfunc())
     eta[3] <- -1
     eta[1] <- 2
     dput(tfunc())
     */
    constexpr size_t const K = 2L, n = 3L;
    arma::vec eta(n, arma::fill::zeros),
    u(K, arma::fill::zeros);
    arma::mat Z;
    Z << 1 << 3 << 2 << 2 << 3 << 1;
    Z.reshape(K, n);
    size_t const n_nodes(30L);
    arma::mat Sigma = arma::diagmat(arma::vec(K, arma::fill::ones));
    Sigma.at(0, 1) = .5;
    Sigma.at(1, 0) = .5;
    arma::mat const Sigma_chol = arma::chol(Sigma);
    double wk_mem[K + 3L * n];

    auto rel_err = [&](double const truth, double const val){
      return std::abs((truth - val) / truth);
    };

    {
      multinomial int_obj(Z, eta, Sigma_chol, wk_mem, n_nodes, true);
      auto const res = int_obj.find_mode(u.memptr());
      expect_true(rel_err(0.935863707021866, res.location) < 1e-2);
      expect_true(rel_err(0.679187598741919, res.scale) < 1e-2);
    }

    u[0] = 1;
    u[1] = -1;
    {
      multinomial int_obj(Z, eta, Sigma_chol, wk_mem, n_nodes, true);
      auto const res = int_obj.find_mode(u.memptr());
      expect_true(rel_err(1.86438099891904, res.location) < 1e-2);
      expect_true(rel_err(0.648249767192147, res.scale) < 1e-2);
    }

    eta[0] = 2;
    eta[2] = -1;
    {
      multinomial int_obj(Z, eta, Sigma_chol, wk_mem, n_nodes, true);
      auto const res = int_obj.find_mode(u.memptr());
      expect_true(rel_err(2.19829169360458, res.location) < 1e-2);
      expect_true(rel_err(0.668742547634432, res.scale) < 1e-2);
    }
  }

  test_that("multinomial gives the correct value"){
  /*
   eta <- numeric(3)
   z <- matrix(c(1:3, 3:1), nc = 2)
   Sig = diag(2)
   Sig[1, 2] <- Sig[2, 1] <- .5
   z <- tcrossprod(z, chol(Sig))
   u <- numeric(2)

   tfunc <- function(){
   f <- Vectorize(function(a, u)
   exp(dnorm(a, log = TRUE) + sum(pnorm(a + eta - z %*% u, log.p = TRUE))),
   vectorize.args = "a")
   int <- function(u_val, h = identity)
   h(integrate(function(a) f(a, u = u_val), -Inf, Inf, abs.tol = 1e-12)$value)
   cat("int\n")
   dput(int(u))
   library(numDeriv)
   cat("grad\n")
   dput(grad(int, u, h = log))
   cat("Hess\n")
   dput(hessian(int, u, h = log))
   }
   tfunc()
   u[1] <- 1
   tfunc()
   eta[3] <- -1
   tfunc()
  */
    for(int is_adap = 0; is_adap < 2L; is_adap++){
    constexpr size_t const K = 2L, n = 3L;
    arma::vec eta(n, arma::fill::zeros),
                u(K, arma::fill::zeros);
    arma::mat Z;
    Z << 1 << 3 << 2 << 2 << 3 << 1;
    Z.reshape(K, n);
    size_t const n_nodes(30L);
    arma::mat Sigma = arma::diagmat(arma::vec(K, arma::fill::ones));
    Sigma.at(0, 1) = .5;
    Sigma.at(1, 0) = .5;
    arma::mat const Sigma_chol = arma::chol(Sigma);
    double wk_mem[K + 3L * n];

    auto rel_err = [&](double const truth, double const val){
      return std::abs((truth - val) / truth);
    };

    {
      multinomial int_obj(Z, eta, Sigma_chol, wk_mem, n_nodes, is_adap);
      constexpr double const expect = .25;
      expect_true(rel_err(    expect , int_obj(u.memptr(), false)) < 1e-6);
      expect_true(rel_err(log(expect), int_obj(u.memptr(), true )) < 1e-6);

      constexpr double const ex_grad[K] =
        { -3.08812611900589, -1.78293044610656 };
      auto const grad = int_obj.gr(u.memptr());
      for(size_t i = 0; i < K; ++i)
        expect_true(rel_err(ex_grad[i], grad[i]) < 1e-6);

      constexpr double const ex_Hess[K * K] =
        { -4.7583400057013, -2.32281253125894, -2.32281253125894,
          -2.07618707706723 };
      auto Hess = int_obj.Hessian(u.memptr());
      Hess.reshape(K * K, 1L);
      for(size_t i = 0; i < K * K; ++i)
        expect_true(rel_err(ex_Hess[i], Hess[i]) < 1e-5);
    }

    u[0] = 1.;
    {
      multinomial int_obj(Z, eta, Sigma_chol, wk_mem, n_nodes, is_adap);
      constexpr double const expect = 0.000726068192627846;
      expect_true(rel_err(    expect , int_obj(u.memptr(), false)) < 1e-6);
      expect_true(rel_err(log(expect), int_obj(u.memptr(), true )) < 1e-6);

      constexpr double const ex_grad[K] =
        { -8.8654891668812, -4.37134944307429 };
      auto const grad = int_obj.gr(u.memptr());
      for(size_t i = 0; i < K; ++i)
        expect_true(rel_err(ex_grad[i], grad[i]) < 1e-6);

      constexpr double const ex_Hess[K * K] =
        { -6.42832565499772, -2.74036282812059, -2.74036282812059,
          -2.70757141148856 };
      auto Hess = int_obj.Hessian(u.memptr());
      Hess.reshape(K * K, 1L);
      for(size_t i = 0; i < K * K; ++i)
        expect_true(rel_err(ex_Hess[i], Hess[i]) < 1e-5);
    }

    eta[2] = -1.;
    {
      multinomial int_obj(Z, eta, Sigma_chol, wk_mem, n_nodes, is_adap);
      constexpr double const expect = 0.000151791804531602;
      expect_true(rel_err(    expect , int_obj(u.memptr(), false)) < 1e-6);
      expect_true(rel_err(log(expect), int_obj(u.memptr(), true )) < 1e-6);

      constexpr double const ex_grad[K] =
        { -10.0323365211649, -4.22373084962153 };
      auto const grad = int_obj.gr(u.memptr());
      for(size_t i = 0; i < K; ++i)
        expect_true(rel_err(ex_grad[i], grad[i]) < 1e-6);

      constexpr double const ex_Hess[K * K] =
        { -6.55938372130066, -2.62586801563047, -2.62586801563047,
          -2.51790069062696 };
      auto Hess = int_obj.Hessian(u.memptr());
      Hess.reshape(K * K, 1L);
      for(size_t i = 0; i < K * K; ++i)
        expect_true(rel_err(ex_Hess[i], Hess[i]) < 1e-5);
    }
    }
  }

  test_that("multinomial_group gives the correct value"){
    /*
     eta <- numeric(3)
     z <- matrix(c(1:3, 3:1), nc = 2)
     Sig = diag(2)
     Sig[1, 2] <- Sig[2, 1] <- .5
     z <- tcrossprod(z, chol(Sig))
     u <- c(-.5, 1)

     tfunc <- function(){
     f <- Vectorize(function(a, u)
     exp(dnorm(a, log = TRUE) + sum(pnorm(a + eta - z %*% u, log.p = TRUE))),
     vectorize.args = "a")
     int <- function(u_val, h = identity)
     h(integrate(function(a) f(a, u = u_val), -Inf, Inf, abs.tol = 1e-12)$value)
     h <- int(u)
     library(numDeriv)
     gr <- grad(int, u, h = log)
     he <- hessian(int, u, h = log)

     list(h = h, gr = gr, he = he, eta = eta, z = z)
     }
     r1 <- tfunc()
     eta[1] <- eta[3] <- -1
     r2 <- tfunc()
     z <- diag(c(-1, 1, -1)) %*% z
     r3 <- tfunc()
     dput(r1$h * r2$h * r3$h)
     dput(drop(r1$gr + r2$gr + r3$gr))
     dput(drop(r1$he + r2$he + r3$he))
     dput(c(r1$eta, r2$eta, r3$eta))
     dput(t(rbind(r1$z, r2$z, r3$z) %*% solve(t(chol(Sig)))))
     */
    for(unsigned is_adaptive = 0L; is_adaptive < 2L; ++is_adaptive){
    constexpr size_t const K = 2L, n = 3L, c = 3L;
    arma::vec eta, u;
    eta << 0 << 0 << 0 << -1 << 0 << -1 << -1 << 0 << -1;
    u << -.5 << 1;
    arma::mat Z;
    Z << 1 << 3 << 2 << 2 << 3 << 1 << 1 << 3 << 2 << 2 << 3 << 1 << -1
      << -3 << 2 << 2 << -3 << -1;
    Z.reshape(K, n * c);
    size_t const n_nodes(30L);
    arma::mat Sigma = arma::diagmat(arma::vec(K, arma::fill::ones));
    Sigma.at(0, 1) = .5;
    Sigma.at(1, 0) = .5;

    multinomial_group obj(n, Z, eta, Sigma, n_nodes, is_adaptive);

    auto rel_err = [&](double const truth, double const val){
      return std::abs((truth - val) / truth);
    };
    constexpr double const expect_h = 0.000294478512867879;
    expect_true(rel_err(expect_h, obj(u.memptr(), false)) < 1e-5);

    auto const grad = obj.gr(u.memptr());
    expect_true(grad.n_elem == K);
    constexpr double const grad_ex[K] =
      { -4.56769959666999, -6.00593937217748 };
    for(size_t i = 0; i < K; ++i)
      expect_true(rel_err(grad_ex[i], grad[i]) < 1e-5);

    constexpr double const hess_ex[K * K] =
      { -19.1288812061111, -10.0782541696758, -10.0782541696758,
        -8.22900374990315 };
    auto hess = obj.Hessian(u.memptr());
    hess.reshape(K * K, 1L);
    for(size_t i = 0; i < K * K; ++i)
      expect_true(rel_err(hess_ex[i], hess.at(i, 0L)) < 1e-5);
    }
  }
}
