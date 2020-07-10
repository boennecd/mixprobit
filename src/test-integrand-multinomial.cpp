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

    auto rel_err = [&](double const truth, double const val){
      return std::abs((truth - val) / truth);
    };

    {
      multinomial int_obj(Z, eta, Sigma, n_nodes, true);
      auto const res = int_obj.find_mode(u.memptr());
      expect_true(rel_err(res.location, 0.935863707021866) < 1e-2);
      expect_true(rel_err(res.scale, 0.679187598741919) < 1e-2);
    }

    u[0] = 1;
    u[1] = -1;
    {
      multinomial int_obj(Z, eta, Sigma, n_nodes, true);
      auto const res = int_obj.find_mode(u.memptr());
      expect_true(rel_err(res.location, 1.86438099891904) < 1e-2);
      expect_true(rel_err(res.scale, 0.648249767192147) < 1e-2);
    }

    eta[0] = 2;
    eta[2] = -1;
    {
      multinomial int_obj(Z, eta, Sigma, n_nodes, true);
      auto const res = int_obj.find_mode(u.memptr());
      expect_true(rel_err(res.location, 2.19829169360458) < 1e-2);
      expect_true(rel_err(res.scale, 0.668742547634432) < 1e-2);
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

    auto rel_err = [&](double const truth, double const val){
      return std::abs((truth - val) / truth);
    };

    {
      multinomial int_obj(Z, eta, Sigma, n_nodes, is_adap);
      constexpr double const expect = .25;
      expect_true(rel_err(int_obj(u.memptr(), false),     expect ) < 1e-6);
      expect_true(rel_err(int_obj(u.memptr(), true ), log(expect)) < 1e-6);

      constexpr double const ex_grad[K] =
        { -3.08812611900589, -1.78293044610656 };
      auto const grad = int_obj.gr(u.memptr());
      for(size_t i = 0; i < K; ++i)
        expect_true(rel_err(grad[i], ex_grad[i]) < 1e-6);

      constexpr double const ex_Hess[K * K] =
        { -4.7583400057013, -2.32281253125894, -2.32281253125894,
          -2.07618707706723 };
      auto Hess = int_obj.Hessian(u.memptr());
      Hess.reshape(K * K, 1L);
      for(size_t i = 0; i < K * K; ++i)
        expect_true(rel_err(Hess[i], ex_Hess[i]) < 1e-5);
    }

    u[0] = 1.;
    {
      multinomial int_obj(Z, eta, Sigma, n_nodes, is_adap);
      constexpr double const expect = 0.000726068192627846;
      expect_true(rel_err(int_obj(u.memptr(), false),     expect ) < 1e-6);
      expect_true(rel_err(int_obj(u.memptr(), true ), log(expect)) < 1e-6);

      constexpr double const ex_grad[K] =
        { -8.8654891668812, -4.37134944307429 };
      auto const grad = int_obj.gr(u.memptr());
      for(size_t i = 0; i < K; ++i)
        expect_true(rel_err(grad[i], ex_grad[i]) < 1e-6);

      constexpr double const ex_Hess[K * K] =
        { -6.42832565499772, -2.74036282812059, -2.74036282812059,
          -2.70757141148856 };
      auto Hess = int_obj.Hessian(u.memptr());
      Hess.reshape(K * K, 1L);
      for(size_t i = 0; i < K * K; ++i)
        expect_true(rel_err(Hess[i], ex_Hess[i]) < 1e-5);
    }

    eta[2] = -1.;
    {
      multinomial int_obj(Z, eta, Sigma, n_nodes, is_adap);
      constexpr double const expect = 0.000151791804531602;
      expect_true(rel_err(int_obj(u.memptr(), false),     expect ) < 1e-6);
      expect_true(rel_err(int_obj(u.memptr(), true ), log(expect)) < 1e-6);

      constexpr double const ex_grad[K] =
        { -10.0323365211649, -4.22373084962153 };
      auto const grad = int_obj.gr(u.memptr());
      for(size_t i = 0; i < K; ++i)
        expect_true(rel_err(grad[i], ex_grad[i]) < 1e-6);

      constexpr double const ex_Hess[K * K] =
        { -6.55938372130066, -2.62586801563047, -2.62586801563047,
          -2.51790069062696 };
      auto Hess = int_obj.Hessian(u.memptr());
      Hess.reshape(K * K, 1L);
      for(size_t i = 0; i < K * K; ++i)
        expect_true(rel_err(Hess[i], ex_Hess[i]) < 1e-5);
    }
    }
  }
}
