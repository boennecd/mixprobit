Mixed Models with Probit Link
=============================

We make a comparison below of making an approximation of a marignal log-likelihood term that is typical in many mixed effect models with a probit link funciton.

TODO: make a better description.

First, we assign a few functions that we are going to use.

``` r
aprx <- within(list(), {
  #####
  # returns a function to perform Gaussian Hermite quadrature (GHQ).
  #
  # Args:
  #   y: n length logical vector with for whether the observation has an 
  #      event.
  #   eta: n length numeric vector with offset on z-scale.
  #   Z: p by n matrix with random effect covariates. 
  #   S: n by n matrix with random effect covaraites.
  #   b: number of nodes to use with GHQ.
  get_GHQ_R <- function(y, eta, Z, S, b){
    library(fastGHQuad)
    library(compiler)
    rule <- gaussHermiteData(b)
    S_chol <- chol(S)
    
    # integrand
    f <- function(x)
      sum(mapply(pnorm, q = eta + sqrt(2) * drop(x %*% S_chol %*% Z),
               lower.tail = y, log.p = TRUE))
    
    # get all permutations of weights and values
    idx <- do.call(expand.grid, replicate(p, 1:b, simplify = FALSE))
    xs <- local({
      args <- list(FUN = c, SIMPLIFY = FALSE)
      do.call(mapply, c(args, lapply(idx, function(i) rule$x[i])))
    })
    ws_log <- local({
      args <- list(FUN = prod)
      log(do.call(mapply, c(args, lapply(idx, function(i) rule$w[i]))))
    })
    
    # final function to return
    out <- function()
      sum(exp(ws_log + vapply(xs, f, numeric(1L)))) / pi^(p / 2)
    f   <- cmpfun(f)
    out <- cmpfun(out)
    out
  }
  
  #####
  # returns a function to perform Gaussian Hermite quadrature (GHQ) using 
  # the C++ implemtation.
  # 
  # Args:
  #   y: n length logical vector with for whether the observation has an 
  #      event.
  #   eta: n length numeric vector with offset on z-scale.
  #   Z: p by n matrix with random effect covariates. 
  #   S: n by n matrix with random effect covaraites.
  #   b: number of nodes to use with GHQ.
  get_GHQ_cpp <- function(y, eta, Z, S, b){
    mixprobit:::set_GH_rule_cached(b)
    function()
      mixprobit:::aprx_binary_mix_ghq(y = y, eta = eta, Z = Z, Sigma = S,
                                      b = b)
  }
  
  #####
  # returns a function that returns the CDF approximation like in Pawitan 
  # et al. (2004).
  #
  # Args:
  #   y: n length logical vector with for whether the observation has an 
  #      event.
  #   eta: n length numeric vector with offset on z-scale.
  #   Z: p by n matrix with random effect covariates. 
  #   S: n by n matrix with random effect covaraites.
  #   maxpts: maximum number of function values as integer. 
  #   abseps: bsolute error tolerance.
  get_cdf_R <- function(y, eta, Z, S, maxpts, abseps = 1e-5){
    library(compiler)
    library(mvtnorm)
    p <- NROW(Z)
    
    out <- function(){
      dum_vec <- ifelse(y, 1, -1)
      Z_tilde <- Z * rep(dum_vec, each = p)
      SMat <- crossprod(Z_tilde , S %*% Z_tilde)
      diag(SMat) <- diag(SMat) + 1
      pmvnorm(upper = dum_vec * eta, mean = rep(0, n), sigma = SMat,
              algorithm = GenzBretz(maxpts = maxpts, abseps = abseps))
    }
    out <- cmpfun(out)
    out
  }
  
  #####
  # returns a function that returns the CDF approximation like in Pawitan 
  # et al. (2004) using the C++ implementation.
  #
  # Args:
  #   y: n length logical vector with for whether the observation has an 
  #      event.
  #   eta: n length numeric vector with offset on z-scale.
  #   Z: p by n matrix with random effect covariates. 
  #   S: n by n matrix with random effect covaraites.
  #   maxpts: maximum number of function values as integer. 
  #   abseps: bsolute error tolerance.
  get_cdf_cpp <- function(y, eta, Z, S, maxpts, abseps = 1e-5)
    function()
      mixprobit:::aprx_binary_mix_cdf(
        y = y, eta = eta, Z = Z, Sigma = S, maxpts = maxpts,
        abseps = abseps, releps = -1)
  
  #####
  # returns a function that uses the method from Genz & Monahan (1998).
  #
  # Args:
  #   y: n length logical vector with for whether the observation has an 
  #      event.
  #   eta: n length numeric vector with offset on z-scale.
  #   Z: p by n matrix with random effect covariates. 
  #   S: n by n matrix with random effect covaraites.
  #   maxpts: maximum number of function values as integer. 
  #   abseps: bsolute error tolerance.
  get_sim_mth <- function(y, eta, Z, S, maxpts, abseps = 1e-5)
    # Args: 
    #   key: integer which determines degree of integration rule.
    function(key)
      mixprobit:::aprx_binary_mix(
        y = y, eta = eta, Z = Z, Sigma = S, mxvals = maxpts, key = key, 
        epsabs = abseps, epsrel = -1)
})
```

Then we assign a function to get a simulated data set for a single cluster within a mixed probit model with binary outcomes.

``` r
#####
# returns a simulated data set from one cluster in a mixed probit model 
# with binary outcomes.
# 
# Args:
#   n: cluster size.
#   p: number of random effects.
get_sim_dat <- function(n, p){
  out <- list(n = n, p = p)
  within(out, {
    Z <- do.call(                        # random effect design matrix
      rbind, c(list(1), list(replicate(n, runif(p - 1L, -1, 1)))))
    eta <- runif(n, -1, 1)               # fixed offsets/fixed effects
    n <- NCOL(Z)                         # number of individuals
    p <- NROW(Z)                         # number of random effects
    S <- drop(                           # covariance matrix of random effects
      rWishart(1, p, diag(sqrt(1/ 2 / p), p)))
    S_chol <- chol(S)
    u <- drop(rnorm(p) %*% S_chol)       # random effects
    y <- runif(n) < pnorm(drop(u %*% Z)) # observed outcomes
  })
}
```

Next we perform a quick example.

``` r
options(digits = 4)
set.seed(2)

#####
# parameters to change
n <- 10L              # cluster size
p <- 4L               # number of random effects
b <- 30L              # number of nodes to use with GHQ
maxpts <- p * 10000L  # factor to set the (maximum) number of
                      # evaluations of the integrand with
                      # the other methods

#####
# variables used in simulation
dat <- get_sim_dat(n = n, p = p)

#####
# get the functions to use
GHQ_R   <- with(dat, 
                aprx$get_GHQ_R  (y = y, eta = eta, Z = Z, S = S, b = b))
GHQ_cpp <- with(dat,
                aprx$get_GHQ_cpp(y = y, eta = eta, Z = Z, S = S, b = b))

cdf_aprx_R   <- with(dat, 
                     aprx$get_cdf_R  (y = y, eta = eta, Z = Z, S = S, 
                                      maxpts = maxpts))
cdf_aprx_cpp <- with(dat, 
                     aprx$get_cdf_cpp(y = y, eta = eta, Z = Z, S = S, 
                                      maxpts = maxpts))

sim_aprx <- with(dat, 
                 aprx$get_sim_mth(y = y, eta = eta, Z = Z, S = S, 
                                  maxpts = maxpts))

#####
# compare results. Start with the simulation based methods with a lot of
# samples. We take this as the ground truth
capital_T_truth_maybe1 <- with(
  dat,
  aprx$get_cdf_cpp(y = y, eta = eta, Z = Z, S = S, maxpts = 1e7, 
                   abseps = 1e-11))()
capital_T_truth_maybe2 <- with(
  dat,
  aprx$get_sim_mth(y = y, eta = eta, Z = Z, S = S, maxpts = 1e7, 
                   abseps = 1e-11)(4L))
dput(capital_T_truth_maybe1)
#> structure(0.00928768943662851, inform = 1L, error = 1.26643795888541e-08)
dput(capital_T_truth_maybe2)
#> structure(0.00928835926643047, error = 2.45853867038835e-06, inform = 1L, inivls = 9999921L)
all.equal(c(capital_T_truth_maybe1), c(capital_T_truth_maybe2))
#> [1] "Mean relative difference: 7.212e-05"
capital_T_truth_maybe <- c(capital_T_truth_maybe1)

# compare with using fewer samples and GHQ
all.equal(capital_T_truth_maybe,   GHQ_R())
#> [1] "Mean relative difference: 0.0007289"
all.equal(capital_T_truth_maybe,   GHQ_cpp())
#> [1] "Mean relative difference: 0.0007289"
all.equal(capital_T_truth_maybe, c(cdf_aprx_R()))
#> [1] "Mean relative difference: 9.588e-06"
all.equal(capital_T_truth_maybe, c(cdf_aprx_cpp()))
#> [1] "Mean relative difference: 1.201e-05"
all.equal(capital_T_truth_maybe, c(sim_aprx(1L)))
#> [1] "Mean relative difference: 0.0131"
all.equal(capital_T_truth_maybe, c(sim_aprx(2L)))
#> [1] "Mean relative difference: 0.005055"
all.equal(capital_T_truth_maybe, c(sim_aprx(3L)))
#> [1] "Mean relative difference: 0.001071"
all.equal(capital_T_truth_maybe, c(sim_aprx(4L)))
#> [1] "Mean relative difference: 0.003025"

# compare computations times
system.time(GHQ_R()) # way too slow (seconds!). Use C++ method instead
#>    user  system elapsed 
#>    20.9     0.0    20.9
microbenchmark::microbenchmark(
  `GHQ (C++)` = GHQ_cpp(),
  `CDF` = cdf_aprx_R(), `CDF (C++)` = cdf_aprx_cpp(),
  `Genz & Monahan (1)` = sim_aprx(1L), `Genz & Monahan (2)` = sim_aprx(2L),
  `Genz & Monahan (3)` = sim_aprx(3L), `Genz & Monahan (4)` = sim_aprx(4L),
  times = 10)
#> Unit: milliseconds
#>                expr    min     lq   mean median     uq    max neval
#>           GHQ (C++) 609.06 610.21 612.47 611.52 612.85 620.11    10
#>                 CDF  20.42  20.45  20.82  20.87  21.10  21.36    10
#>           CDF (C++)  19.81  19.86  20.07  20.08  20.29  20.45    10
#>  Genz & Monahan (1)  30.37  30.45  31.13  31.08  31.59  32.33    10
#>  Genz & Monahan (2)  30.22  30.33  31.24  30.64  30.96  36.69    10
#>  Genz & Monahan (3)  28.85  28.88  29.33  29.27  29.73  30.03    10
#>  Genz & Monahan (4)  28.19  28.72  29.31  29.07  29.81  30.87    10
```

References
----------

Genz, Alan, and Frank Bretz. 2002. “Comparison of Methods for the Computation of Multivariate T Probabilities.” *Journal of Computational and Graphical Statistics* 11 (4). Taylor & Francis: 950–71. doi:[10.1198/106186002394](https://doi.org/10.1198/106186002394).

Genz, Alan., and John. Monahan. 1998. “Stochastic Integration Rules for Infinite Regions.” *SIAM Journal on Scientific Computing* 19 (2): 426–39. doi:[10.1137/S1064827595286803](https://doi.org/10.1137/S1064827595286803).

Liu, Qing, and Donald A. Pierce. 1994. “A Note on Gauss-Hermite Quadrature.” *Biometrika* 81 (3). \[Oxford University Press, Biometrika Trust\]: 624–29. <http://www.jstor.org/stable/2337136>.

Ochi, Y., and Ross L. Prentice. 1984. “Likelihood Inference in a Correlated Probit Regression Model.” *Biometrika* 71 (3). \[Oxford University Press, Biometrika Trust\]: 531–43. <http://www.jstor.org/stable/2336562>.

Pawitan, Y., M. Reilly, E. Nilsson, S. Cnattingius, and P. Lichtenstein. 2004. “Estimation of Genetic and Environmental Factors for Binary Traits Using Family Data.” *Statistics in Medicine* 23 (3): 449–65. doi:[10.1002/sim.1603](https://doi.org/10.1002/sim.1603).
