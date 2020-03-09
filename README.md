Mixed Models with a Probit Link
===============================

We make a comparison below of making an approximation of a marginal likelihood factor that is typical in many mixed effect models with a probit link function. The particular model we use here is mixed probit model where the observed outcomes are binary. In this model, a marginal factor, ![L](https://latex.codecogs.com/svg.latex?L "L"), for a given cluster is

![\\begin{align\*}
L &= \\int \\phi^{(p)}(\\vec u; \\vec 0, \\Sigma)
  \\prod\_{i = 1}^n 
  \\Phi(\\eta\_i + \\vec z\_i^\\top\\vec u)^{y\_i} 
  \\Phi(-\\eta\_i-\\vec z\_i^\\top\\vec u)^{1 - y\_i}
  d\\vec u \\\\
\\vec y &\\in \\{0,1\\}^n \\\\
\\phi^{(p)}(\\vec u;\\vec \\mu, \\Sigma) &= 
  \\frac 1{(2\\pi)^{p/2}\\lvert\\Sigma\\rvert^{1/2}}
  \\exp\\left(-\\frac 12 (\\vec u - \\vec\\mu)^\\top\\Sigma^{-1}
                      (\\vec u - \\vec\\mu)\\right), 
  \\quad \\vec u \\in\\mathbb{R}^p\\\\
\\Phi(x) &= \\int\_0^x\\phi^{(1)}(z;0,1)dz
\\end{align\*}](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign%2A%7D%0AL%20%26%3D%20%5Cint%20%5Cphi%5E%7B%28p%29%7D%28%5Cvec%20u%3B%20%5Cvec%200%2C%20%5CSigma%29%0A%20%20%5Cprod_%7Bi%20%3D%201%7D%5En%20%0A%20%20%5CPhi%28%5Ceta_i%20%2B%20%5Cvec%20z_i%5E%5Ctop%5Cvec%20u%29%5E%7By_i%7D%20%0A%20%20%5CPhi%28-%5Ceta_i-%5Cvec%20z_i%5E%5Ctop%5Cvec%20u%29%5E%7B1%20-%20y_i%7D%0A%20%20d%5Cvec%20u%20%5C%5C%0A%5Cvec%20y%20%26%5Cin%20%5C%7B0%2C1%5C%7D%5En%20%5C%5C%0A%5Cphi%5E%7B%28p%29%7D%28%5Cvec%20u%3B%5Cvec%20%5Cmu%2C%20%5CSigma%29%20%26%3D%20%0A%20%20%5Cfrac%201%7B%282%5Cpi%29%5E%7Bp%2F2%7D%5Clvert%5CSigma%5Crvert%5E%7B1%2F2%7D%7D%0A%20%20%5Cexp%5Cleft%28-%5Cfrac%2012%20%28%5Cvec%20u%20-%20%5Cvec%5Cmu%29%5E%5Ctop%5CSigma%5E%7B-1%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%28%5Cvec%20u%20-%20%5Cvec%5Cmu%29%5Cright%29%2C%20%0A%20%20%5Cquad%20%5Cvec%20u%20%5Cin%5Cmathbb%7BR%7D%5Ep%5C%5C%0A%5CPhi%28x%29%20%26%3D%20%5Cint_0%5Ex%5Cphi%5E%7B%281%29%7D%28z%3B0%2C1%29dz%0A%5Cend%7Balign%2A%7D "\begin{align*}
L &= \int \phi^{(p)}(\vec u; \vec 0, \Sigma)
  \prod_{i = 1}^n 
  \Phi(\eta_i + \vec z_i^\top\vec u)^{y_i} 
  \Phi(-\eta_i-\vec z_i^\top\vec u)^{1 - y_i}
  d\vec u \\
\vec y &\in \{0,1\}^n \\
\phi^{(p)}(\vec u;\vec \mu, \Sigma) &= 
  \frac 1{(2\pi)^{p/2}\lvert\Sigma\rvert^{1/2}}
  \exp\left(-\frac 12 (\vec u - \vec\mu)^\top\Sigma^{-1}
                      (\vec u - \vec\mu)\right), 
  \quad \vec u \in\mathbb{R}^p\\
\Phi(x) &= \int_0^x\phi^{(1)}(z;0,1)dz
\end{align*}")

where ![\\eta\_i](https://latex.codecogs.com/svg.latex?%5Ceta_i "\eta_i") can be a fixed effect like ![\\vec x\_i^\\top\\vec\\beta](https://latex.codecogs.com/svg.latex?%5Cvec%20x_i%5E%5Ctop%5Cvec%5Cbeta "\vec x_i^\top\vec\beta") for some fixed effect covariate ![\\vec x\_i](https://latex.codecogs.com/svg.latex?%5Cvec%20x_i "\vec x_i") and fixed effect coefficients ![\\vec\\beta](https://latex.codecogs.com/svg.latex?%5Cvec%5Cbeta "\vec\beta") and ![\\vec u](https://latex.codecogs.com/svg.latex?%5Cvec%20u "\vec u") is an unobserved random effect for the cluster.

The [quick comparison](#quick-comparison) section may be skipped unless you want to get a grasp at what is implemented and see the definitions of the functions that is used in this markdown. The [more rigorous comparison](#more-rigorous-comparison) section is the main section of this markdown. It contains an example where we vary the number of observed outcomes, `n`, and the number of random effect, `p`, while considering the computation time of various approximation methods for a fixed relative error. A real data application is provided in [examples/salamander.md](examples/salamander.md).

Quick Comparison
----------------

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
  #   is_adaptive: logical for whether to use adaptive GHQ.
  get_GHQ_cpp <- function(y, eta, Z, S, b, is_adaptive = FALSE){
    mixprobit:::set_GH_rule_cached(b)
    function()
      mixprobit:::aprx_binary_mix_ghq(y = y, eta = eta, Z = Z, Sigma = S,
                                      b = b, is_adaptive = is_adaptive)
  }
  get_AGHQ_cpp <- get_GHQ_cpp
  formals(get_AGHQ_cpp)$is_adaptive <- TRUE
  
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
  #   abseps: absolute error tolerance.
  #   releps: relative error tolerance.
  get_cdf_R <- function(y, eta, Z, S, maxpts, abseps = 1e-5, releps = -1){
    library(compiler)
    library(mvtnorm)
    p <- NROW(Z)
    
    out <- function(){
      dum_vec <- ifelse(y, 1, -1)
      Z_tilde <- Z * rep(dum_vec, each = p)
      SMat <- crossprod(Z_tilde , S %*% Z_tilde)
      diag(SMat) <- diag(SMat) + 1
      pmvnorm(upper = dum_vec * eta, mean = rep(0, n), sigma = SMat,
              algorithm = GenzBretz(maxpts = maxpts, abseps = abseps, 
                                    releps = releps))
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
  #   releps: relative error tolerance.
  get_cdf_cpp <- function(y, eta, Z, S, maxpts, abseps = -1, 
                          releps = 1e-3)
    function()
      mixprobit:::aprx_binary_mix_cdf(
        y = y, eta = eta, Z = Z, Sigma = S, maxpts = maxpts,
        abseps = abseps, releps = releps)
  
  #####
  # returns a function that uses the method from Genz & Monahan (1999).
  #
  # Args:
  #   y: n length logical vector with for whether the observation has an 
  #      event.
  #   eta: n length numeric vector with offset on z-scale.
  #   Z: p by n matrix with random effect covariates. 
  #   S: n by n matrix with random effect covaraites.
  #   maxpts: maximum number of function values as integer. 
  #   abseps: bsolute error tolerance.
  #   releps: relative error tolerance.
  #   is_adaptive: logical for whether to use adaptive method.
  get_sim_mth <- function(y, eta, Z, S, maxpts, abseps = 1e-5, releps = -1, 
                          is_adaptive = FALSE)
    # Args: 
    #   key: integer which determines degree of integration rule.
    function(key)
      mixprobit:::aprx_binary_mix(
        y = y, eta = eta, Z = Z, Sigma = S, maxpts = maxpts, key = key, 
        abseps = abseps, releps = releps, is_adaptive = is_adaptive)
  get_Asim_mth <- get_sim_mth
  formals(get_Asim_mth)$is_adaptive <- TRUE
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
      rbind, c(list(sqrt(1/p)), 
               list(replicate(n, rnorm(p - 1L, sd = sqrt(1/p))))))
    eta <- rnorm(n)                      # fixed offsets/fixed effects
    n <- NCOL(Z)                         # number of individuals
    p <- NROW(Z)                         # number of random effects
    S <- drop(                           # covariance matrix of random effects
      rWishart(1, p, diag(1 / p, p)))
    S_chol <- chol(S)
    u <- drop(rnorm(p) %*% S_chol)       # random effects
    y <- runif(n) < pnorm(drop(u %*% Z)) # observed outcomes
  })
}
```

The variance of the linear predictor given the random effect is independent of the random effect dimension, `p`.

``` r
var(replicate(1000, with(get_sim_dat(10, 2), u %*% Z + eta)))
#> [1] 1.981
var(replicate(1000, with(get_sim_dat(10, 3), u %*% Z + eta)))
#> [1] 1.976
var(replicate(1000, with(get_sim_dat(10, 4), u %*% Z + eta)))
#> [1] 2.004
var(replicate(1000, with(get_sim_dat(10, 5), u %*% Z + eta)))
#> [1] 1.953
var(replicate(1000, with(get_sim_dat(10, 6), u %*% Z + eta)))
#> [1] 1.979
var(replicate(1000, with(get_sim_dat(10, 7), u %*% Z + eta)))
#> [1] 2.025
var(replicate(1000, with(get_sim_dat(10, 8), u %*% Z + eta)))
#> [1] 2.01
```

Next we perform a quick example.

``` r
set.seed(2)

#####
# parameters to change
n <- 10L              # cluster size
p <- 4L               # number of random effects
b <- 15L              # number of nodes to use with GHQ
maxpts <- p * 10000L  # factor to set the (maximum) number of
                      # evaluations of the integrand with
                      # the other methods

#####
# variables used in simulation
dat <- get_sim_dat(n = n, p = p)

# shorter than calling `with(dat, ...)`
wd <- function(expr)
  eval(bquote(with(dat, .(substitute(expr)))), parent.frame())

#####
# get the functions to use
GHQ_R    <- wd(aprx$get_GHQ_R   (y = y, eta = eta, Z = Z, S = S, b = b))
#> Loading required package: Rcpp
GHQ_cpp  <- wd(aprx$get_GHQ_cpp (y = y, eta = eta, Z = Z, S = S, b = b))
AGHQ_cpp <- wd(aprx$get_AGHQ_cpp(y = y, eta = eta, Z = Z, S = S, b = b))

cdf_aprx_R   <- wd(aprx$get_cdf_R  (y = y, eta = eta, Z = Z, S = S, 
                                    maxpts = maxpts))
cdf_aprx_cpp <- wd(aprx$get_cdf_cpp(y = y, eta = eta, Z = Z, S = S, 
                                    maxpts = maxpts))

sim_aprx <-  wd(aprx$get_sim_mth(y = y, eta = eta, Z = Z, S = S, 
                                 maxpts = maxpts))
sim_Aaprx <- wd(aprx$get_Asim_mth(y = y, eta = eta, Z = Z, S = S, 
                                  maxpts = maxpts))

#####
# compare results. Start with the simulation based methods with a lot of
# samples. We take this as the ground truth
truth_maybe1 <- wd( 
  aprx$get_cdf_cpp (y = y, eta = eta, Z = Z, S = S, maxpts = 1e7, 
                    abseps = 1e-11))()
truth_maybe2 <- wd(
  aprx$get_sim_mth (y = y, eta = eta, Z = Z, S = S, maxpts = 1e7, 
                    abseps = 1e-11)(2L))
truth_maybe2_A <- wd(
  aprx$get_Asim_mth(y = y, eta = eta, Z = Z, S = S, maxpts = 1e7, 
                    abseps = 1e-11)(2L))
truth <- wd(
  mixprobit:::aprx_binary_mix_brute(y = y, eta = eta, Z = Z, Sigma = S, 
                                    n_sim = 1e8, n_threads = 6L))

c(Estiamte = truth, SE = attr(truth, "SE"),  
  `Estimate (log)` = log(c(truth)),  
  `SE (log)` = abs(attr(truth, "SE") / truth))
#>       Estiamte             SE Estimate (log)       SE (log) 
#>      4.436e-03      3.413e-08     -5.418e+00      7.694e-06
truth <- c(truth)
all.equal(truth, c(truth_maybe1))
#> [1] "Mean relative difference: 9.68e-05"
all.equal(truth, c(truth_maybe2))
#> [1] "Mean relative difference: 0.0002955"
all.equal(truth, c(truth_maybe2_A))
#> [1] "Mean relative difference: 1.294e-05"

# compare with using fewer samples and GHQ
all.equal(truth,   GHQ_R())
#> [1] "Mean relative difference: 1.193e-05"
all.equal(truth,   GHQ_cpp())
#> [1] "Mean relative difference: 1.193e-05"
all.equal(truth,   AGHQ_cpp())
#> [1] "Mean relative difference: 1.049e-05"
all.equal(truth, c(cdf_aprx_R()))
#> [1] "Mean relative difference: 4.318e-05"
all.equal(truth, c(cdf_aprx_cpp()))
#> [1] "Mean relative difference: 0.0002384"
all.equal(truth, c(sim_aprx(1L)))
#> [1] "Mean relative difference: 0.01504"
all.equal(truth, c(sim_aprx(2L)))
#> [1] "Mean relative difference: 0.005325"
all.equal(truth, c(sim_aprx(3L)))
#> [1] "Mean relative difference: 0.003128"
all.equal(truth, c(sim_aprx(4L)))
#> [1] "Mean relative difference: 0.004472"
all.equal(truth, c(sim_Aaprx(1L)))
#> [1] "Mean relative difference: 2.822e-05"
all.equal(truth, c(sim_Aaprx(2L)))
#> [1] "Mean relative difference: 0.001027"
all.equal(truth, c(sim_Aaprx(3L)))
#> [1] "Mean relative difference: 0.005067"
all.equal(truth, c(sim_Aaprx(4L)))
#> [1] "Mean relative difference: 0.001314"

# compare computations times
system.time(GHQ_R()) # way too slow (seconds!). Use C++ method instead
#>    user  system elapsed 
#>   1.336   0.000   1.335
microbenchmark::microbenchmark(
  `GHQ (C++)` = GHQ_cpp(), `AGHQ (C++)` = AGHQ_cpp(),
  `CDF` = cdf_aprx_R(), `CDF (C++)` = cdf_aprx_cpp(),
  `Genz & Monahan (1)` = sim_aprx(1L), `Genz & Monahan (2)` = sim_aprx(2L),
  `Genz & Monahan (3)` = sim_aprx(3L), `Genz & Monahan (4)` = sim_aprx(4L),
  `Genz & Monahan Adaptive (2)` = sim_Aaprx(2L),
  times = 10)
#> Unit: milliseconds
#>                         expr    min     lq   mean median     uq   max neval
#>                    GHQ (C++) 38.677 38.814 38.958 38.907 39.070 39.34    10
#>                   AGHQ (C++) 41.650 41.864 42.177 42.091 42.334 43.35    10
#>                          CDF 20.716 21.041 21.094 21.069 21.176 21.46    10
#>                    CDF (C++) 11.258 11.482 11.509 11.511 11.570 11.68    10
#>           Genz & Monahan (1) 29.146 29.470 30.074 29.993 30.792 31.04    10
#>           Genz & Monahan (2) 30.162 30.580 31.116 31.160 31.627 32.12    10
#>           Genz & Monahan (3) 29.325 29.946 30.374 30.478 30.843 31.14    10
#>           Genz & Monahan (4) 28.874 29.409 30.022 30.025 30.569 31.00    10
#>  Genz & Monahan Adaptive (2)  5.114  7.326  8.318  8.385  9.538 10.90    10
```

More Rigorous Comparison
------------------------

We are interested in a more rigorous comparison. Therefor, we define a function below which for given number of observation in the cluster, `n`, and given number of random effects, `p`, performs a repeated number of runs with each of the methods and returns the computation time (among other output). To make a fair comparison, we fix the relative error of the methods before hand such that the relative error is below `releps`, ![10^{-4}](https://latex.codecogs.com/svg.latex?10%5E%7B-4%7D "10^{-4}"). Ground truth is computed with brute force MC using `n_brute`, ![10^{7}](https://latex.codecogs.com/svg.latex?10%5E%7B7%7D "10^{7}"), samples.

Since GHQ is deterministic, we use a number of nodes such that this number of nodes or `streak_length`, 4, less value of nodes with GHQ gives a relative error which is below the threshold. We use a minimum of 4 nodes at the time of this writing. The error of the simulation based methods is approximated using `n_reps`, 25, replications.

``` r
# default parameters
ex_params <- list(
  streak_length = 4L, 
  max_b = 25L, 
  max_maxpts = 5000000L, 
  releps = 1e-4,
  min_releps = 1e-6,
  key_use = 3L, 
  n_reps = 25L, 
  n_runs = 5L, 
  n_brute = 1e7, 
  n_brute_max = 1e8, 
  n_brute_sds = 4)
```

``` r
# perform a simulations run for a given number of observations and random 
# effects. First we fix the relative error of each method such that it is
# below a given threshold. Then we run each method a number of times to 
# measure the computation time. 
# 
# Args:
#   n: number of observations in the cluster.
#   p: number of random effects. 
#   releps: required relative error. 
#   key_use: integer which determines degree of integration rule for the 
#            method from Genz and Monahan (1999).
#   n_threads: number of threads to use.
#   n_fail: only used by the function if a brute force estimator cannot
#           get within the precision.
sim_experiment <- function(n, p, releps = ex_params$releps, 
                           key_use = ex_params$key_use, n_threads = 1L, 
                           n_fail = 0L){
  # in some cases we may not want to run the simulation experiment
  do_not_run <- FALSE
  
  # simulate data
  dat <- get_sim_dat(n = n, p = p)
  
  # shorter than calling `with(dat, ...)`
  wd <- function(expr)
    eval(bquote(with(dat, .(substitute(expr)))), parent.frame())
  
  # get the assumed ground truth
  if(do_not_run){
    truth <- SE_truth <- NA_real_
    n_brute <- NA_integer_
    find_brute_failed <- FALSE
    
  } else {
    passed <- FALSE
    n_brute <- NA_integer_
    find_brute_failed <- FALSE
    
    while(!passed){
      if(!is.na(n_brute) && n_brute >= ex_params$n_brute_max){
        n_brute <- NA_integer_
        find_brute_failed <- TRUE
        break
      }
      
      n_brute <- if(is.na(n_brute))
        ex_params$n_brute 
      else 
        min(ex_params$n_brute_max, 
            n_brute * as.integer(ceiling(1.2 * (SE_truth / eps)^2)))
      
      truth <- wd(mixprobit:::aprx_binary_mix_brute(
        y = y, eta = eta, Z = Z, Sigma = S, n_sim = n_brute, 
        n_threads = n_threads))
      
      SE_truth <- abs(attr(truth, "SE") / c(truth))
      eps <- ex_params$releps / ex_params$n_brute_sds * abs(log(c(truth)))
      passed <- SE_truth < eps
    }
      
    truth <- c(truth)
  }
  
  if(find_brute_failed){
    # we failed to find a brute force estimator within the precision. 
    # We repeat with a new data set
    cl <- match.call()
    cl$n_fail <- n_fail + 1L
    return(eval(cl, parent.frame()))
  }
  
  # function to test whether the value is ok
  is_ok_func <- function(vals)
    abs((log(vals) - log(truth)) / log(truth)) < releps
  
  # get function to use with GHQ
  get_b <- function(meth){
    if(do_not_run)
      NA_integer_
    else local({
      apx_func <- function(b)
        wd(meth(y = y, eta = eta, Z = Z, S = S, b = b))()
      
      # length of node values which have a relative error below the threshold
      streak_length <- ex_params$streak_length
      vals <- rep(NA_real_, streak_length)
      
      b <- streak_length
      for(i in 1:(streak_length - 1L))
        vals[i + 1L] <- apx_func(b - streak_length + i)
      repeat {
        vals[1:(streak_length - 1L)] <- vals[-1]
        vals[streak_length] <- apx_func(b)
        
        if(all(is_ok_func(vals)))
          break
        
        b <- b + 1L
        if(b > ex_params$max_b){
          warning("found no node value")
          b <- NA_integer_
          break
        }
      }
      b
    })
  }
  
  is_to_large_for_ghq <- n >= 16L || p >= 5L
  b_use <- if(is_to_large_for_ghq)
    NA_integer_ else get_b(aprx$get_GHQ_cpp)
  ghq_func <- if(!is.na(b_use))
    wd(aprx$get_GHQ_cpp(y = y, eta = eta, Z = Z, S = S, b = b_use))
  else
    NA
  
  # get function to use with AGHQ
  b_use_A <- get_b(aprx$get_AGHQ_cpp)
  aghq_func <- if(!is.na(b_use_A))
    wd(aprx$get_AGHQ_cpp(y = y, eta = eta, Z = Z, S = S, b = b_use_A))
  else
    NA
  
  # get function to use with CDF method
  get_releps <- function(meth){
    if(do_not_run)
      NA_integer_
    else {
      releps_use <- releps * 100
      repeat {
        func <- wd(meth(y = y, eta = eta, Z = Z, S = S, 
                        maxpts = ex_params$max_maxpts, 
                        abseps = -1, releps = releps_use))
        if("key" %in% names(formals(func)))
          formals(func)$key <- ex_params$key_use
        vals <- replicate(ex_params$n_reps, {
          v <- func()
          inivls <- if("inivls" %in% names(attributes(v)))
            attr(v, "inivls") else NA_integer_
          c(value = v, error = attr(v, "error"), inivls = inivls)
        })
        
        inivls_ok <- all(
          is.na(vals["inivls", ]) | 
            vals["inivls", ] / ex_params$max_maxpts < .999)
        
        if(all(is_ok_func(vals["value", ])) && inivls_ok)
          break
        
        releps_use <- if(!inivls_ok) 
          # no point in doing any more computations
          ex_params$min_releps / 10 else 
            releps_use / 2
        if(releps_use < ex_params$min_releps){
          warning("found no releps for CDF method")
          releps_use <- NA_integer_
          break
        }
      }
      releps_use
    }
  }
  
  cdf_releps <- get_releps(aprx$get_cdf_cpp)
  cdf_func <- if(!is.na(cdf_releps))
    wd(aprx$get_cdf_cpp(y = y, eta = eta, Z = Z, S = S, 
                        maxpts = ex_params$max_maxpts, abseps = -1, 
                        releps = cdf_releps))
  else 
    NA
  
  # get function to use with Genz and Monahan method
  sim_releps <- if(is_to_large_for_ghq) 
    NA_integer_ else get_releps(aprx$get_sim_mth)
  sim_func <- if(!is.na(sim_releps))
    wd(aprx$get_sim_mth(y = y, eta = eta, Z = Z, S = S, 
                        maxpts = ex_params$max_maxpts, abseps = -1, 
                        releps = sim_releps))
  else 
    NA
  if(is.function(sim_func))
    formals(sim_func)$key <- key_use
  
  # do the same with the adaptive version
  Asim_releps <- get_releps(aprx$get_Asim_mth)
  Asim_func <- if(!is.na(Asim_releps))
    wd(aprx$get_Asim_mth(y = y, eta = eta, Z = Z, S = S, 
                         maxpts = ex_params$max_maxpts, abseps = -1, 
                         releps = Asim_releps))
  else 
    NA
  if(is.function(Asim_func))
    formals(Asim_func)$key <- key_use
  
  # perform the comparison
  out <- sapply(
    list(GHQ = ghq_func, AGHQ = aghq_func, CDF = cdf_func, 
         GenzMonahan = sim_func, GenzMonahanA = Asim_func), 
    function(func){
      if(!is.function(func) && is.na(func)){
        out <- rep(NA_real_, 6L)
        names(out) <- c("mean", "sd", "mse", "user.self", 
                        "sys.self", "elapsed")
        return(out)
      }
      
      # number of runs used to estimate the computation time, etc.
      n_runs <- ex_params$n_runs
      
      # perform the computations to estimate the computation times
      ti <- system.time(vals <- replicate(n_runs, {
        out <- func()
        if(!is.null(err <- attr(out, "error"))){
          # only of of the two methods needs an adjustment of the sd! 
          # TODO: this is very ad hoc...
          is_genz_mona <- !is.null(environment(func)$is_adaptive)

          sd <- if(is_genz_mona)
            err else err / 2.5
          
          out <- c(value = out, sd = sd)
        }
        out
      }))
      
      # handle computation of sd and mse
      is_ghq <- !is.null(b <- environment(func)$b)
      if(is_ghq){
        # if it is GHQ then we alter the number of nodes to get an sd 
        # estiamte etc.
        sl <- ex_params$streak_length
        other_vs <- sapply((b - sl + 1):b, function(b){
          environment(func)$b <- b
          func()
        })
        
        vs <- c(other_vs, vals[1])
        sd_use <- sd(vs)
        mse <- mean((vs - truth) ^2)
      } else {
        # we combine the variance estimators
        sd_use <- sqrt(mean(vals["sd", ]^2))
        vals <- vals["value", ]
        mse <- mean((vals - truth) ^2)
        
      }
      
      c(mean = mean(vals), sd = sd_use, mse = mse, ti[1:3] / n_runs)            
    })
  
  structure(list(
    b_use = b_use, b_use_A = b_use_A, cdf_releps = cdf_releps, 
    sim_releps = sim_releps, Asim_releps = Asim_releps, 
    ll_truth = log(truth), SE_truth = SE_truth, n_brute = n_brute, 
    n_fail = n_fail, vals_n_comp_time = out), 
    class = "sim_experiment")
}
```

Here is a few quick examples where we use the function we just defined.

``` r
print.sim_experiment <- function(x, ...){
  old <- options()
  on.exit(options(old))
  options(digits = 6, scipen = 999)
  
  cat(
    sprintf("         # brute force samples: %13d", x$n_brute),
    sprintf("                  # nodes  GHQ: %13d", x$b_use),
    sprintf("                  # nodes AGHQ: %13d", x$b_use_A),
    sprintf("                    CDF releps: %13.8f", x$cdf_releps), 
    sprintf("         Genz & Monahan releps: %13.8f", x$sim_releps),
    sprintf("Adaptive Genz & Monahan releps: %13.8f", x$Asim_releps), 
    sprintf("  Log-likelihood estiamte (SE): %13.8f (%.8f)", x$ll_truth, 
            x$SE_truth), 
    "", sep = "\n")
  
  xx <- x$vals_n_comp_time["mean", ]
  print(cbind(`Mean estimate (likelihood)`     = xx, 
              `Mean estimate (log-likelihood)` = log(xx)))
  
  mult <- 1 / ex_params$releps
  cat(sprintf("\nSD & RMSE (/%.2f)\n", mult))
  print(rbind(SD   =      x$vals_n_comp_time["sd", ],  
              RMSE = sqrt(x$vals_n_comp_time[c("mse"), ])) * mult)
  
  cat("\nComputation times\n")
  print(x$vals_n_comp_time["elapsed", ])
}

set.seed(1)
sim_experiment(n = 3L , p = 2L, n_threads = 6L)
#>          # brute force samples:      10000000
#>                   # nodes  GHQ:             9
#>                   # nodes AGHQ:             7
#>                     CDF releps:    0.00015625
#>          Genz & Monahan releps:    0.00007813
#> Adaptive Genz & Monahan releps:    0.00007813
#>   Log-likelihood estiamte (SE):   -0.84934116 (0.00000362)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                            0.427693                      -0.849350
#> AGHQ                           0.427693                      -0.849351
#> CDF                            0.427705                      -0.849322
#> GenzMonahan                    0.427690                      -0.849357
#> GenzMonahanA                   0.427689                      -0.849359
#> 
#> SD & RMSE (/10000.00)
#>            GHQ      AGHQ      CDF GenzMonahan GenzMonahanA
#> SD   0.0866955 0.0192985 0.161242    0.129716     0.129580
#> RMSE 0.1035563 0.0454279 0.140862    0.106769     0.125833
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA 
#>       0.0002       0.0000       0.0002       0.0638       0.0032
sim_experiment(n = 10L, p = 2L, n_threads = 6L)
#>          # brute force samples:      10000000
#>                   # nodes  GHQ:            12
#>                   # nodes AGHQ:             7
#>                     CDF releps:    0.01000000
#>          Genz & Monahan releps:    0.00062500
#> Adaptive Genz & Monahan releps:    0.01000000
#>   Log-likelihood estiamte (SE):   -6.59977058 (0.00000313)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                          0.00136059                       -6.59984
#> AGHQ                         0.00136068                       -6.59977
#> CDF                          0.00136069                       -6.59976
#> GenzMonahan                  0.00136042                       -6.59996
#> GenzMonahanA                 0.00136059                       -6.59983
#> 
#> SD & RMSE (/10000.00)
#>             GHQ        AGHQ         CDF GenzMonahan GenzMonahanA
#> SD   0.00317958 0.000244783 0.000664798  0.00330089   0.00245664
#> RMSE 0.00287675 0.000220605 0.000405248  0.00419424   0.00219275
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA 
#>       0.0002       0.0002       0.0118       0.2544       0.0010

sim_experiment(n = 3L , p = 5L, n_threads = 6L)
#> Warning in get_releps(aprx$get_Asim_mth): found no releps for CDF method
#>          # brute force samples:      10000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             7
#>                     CDF releps:    0.00001953
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:            NA
#>   Log-likelihood estiamte (SE):   -0.69788132 (0.00001560)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                           0.497633                      -0.697893
#> CDF                            0.497638                      -0.697883
#> GenzMonahan                          NA                             NA
#> GenzMonahanA                         NA                             NA
#> 
#> SD & RMSE (/10000.00)
#>      GHQ     AGHQ       CDF GenzMonahan GenzMonahanA
#> SD    NA 0.118737 0.0295792          NA           NA
#> RMSE  NA 0.106942 0.0251312          NA           NA
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA 
#>           NA       0.0056       0.0082           NA           NA
sim_experiment(n = 10L, p = 5L, n_threads = 6L)
#>          # brute force samples:      10000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             7
#>                     CDF releps:    0.01000000
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.00062500
#>   Log-likelihood estiamte (SE):   -9.22899447 (0.00003162)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                       0.0000981574                       -9.22894
#> CDF                        0.0000981678                       -9.22883
#> GenzMonahan                          NA                             NA
#> GenzMonahanA               0.0000981545                       -9.22897
#> 
#> SD & RMSE (/10000.00)
#>      GHQ        AGHQ         CDF GenzMonahan GenzMonahanA
#> SD    NA 0.000124825 0.000268421          NA  0.000238115
#> RMSE  NA 0.000113953 0.000371702          NA  0.000251791
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA 
#>           NA       0.0140       0.0124           NA       0.1518

sim_experiment(n = 3L , p = 7L, n_threads = 6L)
#>          # brute force samples:      10000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             6
#>                     CDF releps:    0.00125000
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.00031250
#>   Log-likelihood estiamte (SE):   -3.58373891 (0.00001481)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                          0.0277718                       -3.58374
#> CDF                           0.0277691                       -3.58383
#> GenzMonahan                          NA                             NA
#> GenzMonahanA                  0.0277703                       -3.58379
#> 
#> SD & RMSE (/10000.00)
#>      GHQ      AGHQ       CDF GenzMonahan GenzMonahanA
#> SD    NA 0.0263532 0.0913941          NA    0.0336394
#> RMSE  NA 0.0274291 0.0680141          NA    0.0526061
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA 
#>           NA       0.0838       0.0002           NA       0.0168
sim_experiment(n = 10L, p = 7L, n_threads = 6L)
#>          # brute force samples:      10000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             7
#>                     CDF releps:    0.01000000
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.00062500
#>   Log-likelihood estiamte (SE):  -10.02500143 (0.00002988)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                       0.0000442762                       -10.0251
#> CDF                        0.0000442737                       -10.0251
#> GenzMonahan                          NA                             NA
#> GenzMonahanA               0.0000442819                       -10.0249
#> 
#> SD & RMSE (/10000.00)
#>      GHQ         AGHQ          CDF GenzMonahan GenzMonahanA
#> SD    NA 0.0000521790 0.0001503558          NA 0.0001074180
#> RMSE  NA 0.0000746156 0.0000978898          NA 0.0000676759
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA 
#>           NA       0.7012       0.0118           NA       0.1634

sim_experiment(n = 20L, p = 7L, n_threads = 6L)
#>          # brute force samples:      10000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             7
#>                     CDF releps:    0.00250000
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.00125000
#>   Log-likelihood estiamte (SE):  -15.92234175 (0.00005487)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                     0.000000121629                       -15.9223
#> CDF                      0.000000121608                       -15.9225
#> GenzMonahan                          NA                             NA
#> GenzMonahanA             0.000000121656                       -15.9221
#> 
#> SD & RMSE (/10000.00)
#>      GHQ           AGHQ            CDF GenzMonahan   GenzMonahanA
#> SD    NA 0.000000271040 0.000000840894          NA 0.000000590181
#> RMSE  NA 0.000000267696 0.000000523387          NA 0.000000719430
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA 
#>           NA       1.3632       0.0820           NA       0.2950
```

Next, we apply the method a number of times for a of combination of number of observations, `n`, and number of random effects, `p`.

``` r
# number of observations in the cluster
n_vals <- 2^(1:5)
# number of random effects
p_vals <- 2:7
# grid with all configurations
gr_vals <- expand.grid(n = n_vals, p = p_vals)
# number of replications per configuration
n_runs <- 100L

ex_output <- (function(){
  # setup directory to store data
  cache_dir <- file.path("README_cache", "experiment")
  if(!dir.exists(cache_dir))
    dir.create(cache_dir)
  
  # setup cluster to use
  library(parallel)
  
  # run the experiment
  mcmapply(function(n, p){
    cache_file <- file.path(cache_dir, sprintf("n-%03d-p-%03d.Rds", n, p))
    if(!file.exists(cache_file)){
      message(sprintf("Running setup with   n %3d and p %3d", n, p))
      
      # create file to write progress to
      prg_file <- file.path(getwd(), 
                            sprintf("progress-n-%03d-p-%03d.txt", n, p))
      file.create(prg_file)
      message(sprintf("Follow progress in %s", sQuote(prg_file)))
      on.exit(unlink(prg_file))
      
      set.seed(71771946)
      sim_out <- lapply(1:n_runs, function(...){
        seed <- .Random.seed
        out <- sim_experiment(n = n, p = p)
        attr(out, "seed") <- seed
        
        cat("-", file = prg_file, append = TRUE)
        out
      })
      
      sim_out[c("n", "p")] <- list(n = n, p = p)
      saveRDS(sim_out, cache_file)
    } else
      message(sprintf ("Loading results with n %3d and p %3d", n, p))
      
    
    readRDS(cache_file)
  }, n = gr_vals$n, p = gr_vals$p, SIMPLIFY = FALSE, 
  mc.cores = 4L, mc.preschedule = FALSE)
})()
```

We create a table where we summarize the results below. First we start with the average computation time, then we show the mean scaled RMSE, and we end by looking at the number of nodes that we need to use with GHQ. The latter shows why GHQ becomes slower as the cluster size, `n`, increases. The computation time is in 1000s of a second, `comp_time_mult`. The mean scaled RMSE is multiplied by ![10^{5}](https://latex.codecogs.com/svg.latex?10%5E%7B5%7D "10^{5}"), `err_mult`.

``` r
comp_time_mult <- 1000 # millisecond
err_mult <- 1e5
```

``` r
#####
# show number of complete cases
local({
  comp_times <- sapply(ex_output, function(x)
    sapply(x[!names(x) %in% c("n", "p")], `[[`, "vals_n_comp_time", 
           simplify = "array"), 
    simplify = "array")
  comp_times <- comp_times["elapsed", , , ]
  n_complete <- apply(!is.na(comp_times), c(1L, 3L), sum)
  
  # flatten the table. Start by getting the row labels
  meths <- rownames(n_complete)
  n_labs <- sprintf("%2d", n_vals)
  rnames <- expand.grid(
    Method = meths, n = n_labs, stringsAsFactors = FALSE)
  rnames[2:1] <- rnames[1:2]
  rnames[[2L]] <- gsub(
    "^GenzMonahan$", "Genz & Monahan (1999)", rnames[[2L]])
  rnames[[2L]] <- gsub(
    "^GenzMonahanA$", "Genz & Monahan (1999) Adaptive", rnames[[2L]])
  # fix stupid typo at one point
  rnames[[2L]] <- gsub(
    "^ADHQ$", "AGHQ", rnames[[2L]])
  
  # then flatten
  n_complete <- matrix(c(n_complete), nrow = NROW(rnames))
  n_complete[] <- sprintf("%4d", n_complete[])
  
  # combine computation times and row labels
  table_out <- cbind(as.matrix(rnames), n_complete)
  
  keep <- apply(
    matrix(as.integer(table_out[, -(1:2), drop = FALSE]), 
           nr = NROW(table_out)) > 0L, 1, any)
  table_out <- table_out[keep, , drop = FALSE]
  
  nvs <- table_out[, 1L]
  table_out[, 1L] <- c(
    nvs[1L], ifelse(nvs[-1L] != head(nvs, -1L), nvs[-1L], NA_integer_))
  
  # add header 
  p_labs <- sprintf("%d", p_vals)
  colnames(table_out) <- c("n", "method/p", p_labs)
  
  cat("**Number of complete cases**\n")
  
  options(knitr.kable.NA = "")
  print(knitr::kable(
    table_out, align = c("l", "l", rep("r", length(p_vals)))))
})
```

**Number of complete cases**

| n   | method/p                       |    2|    3|    4|    5|    6|    7|
|:----|:-------------------------------|----:|----:|----:|----:|----:|----:|
| 2   | GHQ                            |  100|  100|  100|    0|    0|    0|
|     | AGHQ                           |  100|  100|  100|  100|  100|  100|
|     | CDF                            |  100|  100|  100|  100|  100|  100|
|     | Genz & Monahan (1999)          |   78|   60|   60|    0|    0|    0|
|     | Genz & Monahan (1999) Adaptive |   92|   84|   86|   94|   93|   85|
| 4   | GHQ                            |   96|   99|  100|    0|    0|    0|
|     | AGHQ                           |  100|  100|  100|  100|  100|  100|
|     | CDF                            |  100|  100|  100|  100|  100|  100|
|     | Genz & Monahan (1999)          |   65|   56|   54|    0|    0|    0|
|     | Genz & Monahan (1999) Adaptive |   97|   92|   87|   89|   95|   88|
| 8   | GHQ                            |   91|   98|   97|    0|    0|    0|
|     | AGHQ                           |  100|  100|  100|  100|  100|  100|
|     | CDF                            |  100|  100|  100|  100|  100|  100|
|     | Genz & Monahan (1999)          |   45|   39|   27|    0|    0|    0|
|     | Genz & Monahan (1999) Adaptive |   97|   95|   89|   94|   98|   95|
| 16  | AGHQ                           |  100|  100|  100|  100|  100|  100|
|     | CDF                            |  100|  100|  100|  100|  100|  100|
|     | Genz & Monahan (1999) Adaptive |   98|   99|  100|  100|   99|   97|
| 32  | AGHQ                           |  100|  100|  100|  100|  100|  100|
|     | CDF                            |  100|  100|  100|  100|  100|  100|
|     | Genz & Monahan (1999) Adaptive |  100|  100|  100|  100|  100|  100|

``` r

#####
# table with computation times
# util functions
.get_cap <- function(remove_nas, na.rm = FALSE, sufix = ""){
  stopifnot(!(remove_nas && na.rm))
  cap <- if(remove_nas && !na.rm)
    "**Only showing complete cases"
  else if(!remove_nas && na.rm)
    "**NAs have been removed. Cells may not be comparable"
  else 
    "**Blank cells have at least one failure"
  paste0(cap, sufix, "**")
}

.show_n_complete <- function(is_complete, n_labs, p_labs){
  n_complete <- matrix(
    colSums(is_complete), length(n_labs), length(p_labs), 
    dimnames = list(n = n_labs, p = p_labs))
  
  cat("\n**Number of complete cases**")
 print(knitr::kable(n_complete, align = rep("r", ncol(n_complete))))
}

# function to create the computation time table
show_run_times <- function(remove_nas = FALSE, na.rm = FALSE, 
                           meth = rowMeans, suffix = " (means)"){
  # get mean computations time for the methods and the configurations pairs
  comp_times <- sapply(ex_output, function(x)
    sapply(x[!names(x) %in% c("n", "p")], `[[`, "vals_n_comp_time", 
           simplify = "array"), 
    simplify = "array")
  comp_times <- comp_times["elapsed", , , ]
  
  is_complete <- t(apply(comp_times, 2, function(x){
    if(remove_nas)
      apply(!is.na(x), 2, all)
    else 
      rep(TRUE, NCOL(x))
  }))
  dim(is_complete) <- dim(comp_times)[2:3]
  
  comp_times <- lapply(1:dim(comp_times)[3], function(i){
    x <- comp_times[, , i]
    x[, is_complete[, i]]
  })
  comp_times <- sapply(comp_times, meth, na.rm  = na.rm) * 
    comp_time_mult
  comp_times[is.nan(comp_times)] <- NA_real_
  
  # flatten the table. Start by getting the row labels
  meths <- rownames(comp_times)
  n_labs <- sprintf("%2d", n_vals)
  rnames <- expand.grid(
    Method = meths, n = n_labs, stringsAsFactors = FALSE)
  rnames[2:1] <- rnames[1:2]
  rnames[[2L]] <- gsub(
    "^GenzMonahan$", "Genz & Monahan (1999)", rnames[[2L]])
  rnames[[2L]] <- gsub(
    "^GenzMonahanA$", "Genz & Monahan (1999) Adaptive", rnames[[2L]])
  # fix stupid typo at one point
  rnames[[2L]] <- gsub(
    "^ADHQ$", "AGHQ", rnames[[2L]])
  
  # then flatten
  comp_times <- matrix(c(comp_times), nrow = NROW(rnames))
  na_idx <- is.na(comp_times)
  comp_times[] <- sprintf("%.2f", comp_times[])
  comp_times[na_idx] <- NA_character_
  
  # combine computation times and row labels
  table_out <- cbind(as.matrix(rnames), comp_times)
  
  if(na.rm){
    keep <- apply(!is.na(table_out[, -(1:2), drop = FALSE]), 1, any)
    table_out <- table_out[keep, , drop = FALSE]
  }
  
  nvs <- table_out[, 1L]
  table_out[, 1L] <- c(
    nvs[1L], ifelse(nvs[-1L] != head(nvs, -1L), nvs[-1L], NA_integer_))
  
  # add header 
  p_labs <- sprintf("%d", p_vals)
  colnames(table_out) <- c("n", "method/p", p_labs)
  
  cat(.get_cap(remove_nas, na.rm, sufix = suffix))
    
  options(knitr.kable.NA = "")
  print(knitr::kable(
    table_out, align = c("l", "l", rep("r", length(p_vals)))))
  
  if(remove_nas)
    .show_n_complete(is_complete, n_labs, p_labs)
}

show_run_times(FALSE)
```

**Blank cells have at least one failure (means)**

| n   | method/p                       |       2|       3|       4|       5|       6|        7|
|:----|:-------------------------------|-------:|-------:|-------:|-------:|-------:|--------:|
| 2   | GHQ                            |    0.05|    0.27|    1.64|        |        |         |
|     | AGHQ                           |    0.05|    0.13|    0.67|    4.94|   31.25|   241.66|
|     | CDF                            |    0.04|    0.05|    0.05|    0.04|    0.04|     0.04|
|     | Genz & Monahan (1999)          |        |        |        |        |        |         |
|     | Genz & Monahan (1999) Adaptive |        |        |        |        |        |         |
| 4   | GHQ                            |        |        |    5.94|        |        |         |
|     | AGHQ                           |    0.06|    0.21|    1.26|    8.56|   53.89|   364.76|
|     | CDF                            |    1.24|    1.15|    1.26|    1.03|    1.12|     0.93|
|     | Genz & Monahan (1999)          |        |        |        |        |        |         |
|     | Genz & Monahan (1999) Adaptive |        |        |        |        |        |         |
| 8   | GHQ                            |        |        |        |        |        |         |
|     | AGHQ                           |    0.08|    0.31|    2.08|   13.47|   86.35|   638.14|
|     | CDF                            |    5.32|    6.37|    5.60|    5.20|    5.16|     5.63|
|     | Genz & Monahan (1999)          |        |        |        |        |        |         |
|     | Genz & Monahan (1999) Adaptive |        |        |        |        |        |         |
| 16  | GHQ                            |        |        |        |        |        |         |
|     | AGHQ                           |    0.13|    0.48|    3.24|   19.23|  130.25|   981.49|
|     | CDF                            |   34.20|   36.01|   38.96|   37.28|   34.33|    34.84|
|     | Genz & Monahan (1999)          |        |        |        |        |        |         |
|     | Genz & Monahan (1999) Adaptive |        |        |  109.39|   99.70|        |         |
| 32  | GHQ                            |        |        |        |        |        |         |
|     | AGHQ                           |    0.19|    0.66|    3.98|   30.29|  188.14|  1106.73|
|     | CDF                            |  142.39|  169.18|  298.72|  112.39|  114.96|   129.43|
|     | Genz & Monahan (1999)          |        |        |        |        |        |         |
|     | Genz & Monahan (1999) Adaptive |   42.68|   20.55|   24.17|   47.83|   56.54|    79.30|

``` r
show_run_times(na.rm = TRUE)
```

**NAs have been removed. Cells may not be comparable (means)**

| n   | method/p                       |        2|        3|        4|       5|       6|        7|
|:----|:-------------------------------|--------:|--------:|--------:|-------:|-------:|--------:|
| 2   | GHQ                            |     0.05|     0.27|     1.64|        |        |         |
|     | AGHQ                           |     0.05|     0.13|     0.67|    4.94|   31.25|   241.66|
|     | CDF                            |     0.04|     0.05|     0.05|    0.04|    0.04|     0.04|
|     | Genz & Monahan (1999)          |   219.69|   233.30|   207.81|        |        |         |
|     | Genz & Monahan (1999) Adaptive |    64.00|    76.15|    97.42|  118.43|  103.21|   170.21|
| 4   | GHQ                            |     0.09|     0.62|     5.94|        |        |         |
|     | AGHQ                           |     0.06|     0.21|     1.26|    8.56|   53.89|   364.76|
|     | CDF                            |     1.24|     1.15|     1.26|    1.03|    1.12|     0.93|
|     | Genz & Monahan (1999)          |   488.43|   497.66|   500.00|        |        |         |
|     | Genz & Monahan (1999) Adaptive |    59.92|   102.76|   150.17|  144.12|  166.81|   181.96|
| 8   | GHQ                            |     0.18|     2.18|    28.37|        |        |         |
|     | AGHQ                           |     0.08|     0.31|     2.08|   13.47|   86.35|   638.14|
|     | CDF                            |     5.32|     6.37|     5.60|    5.20|    5.16|     5.63|
|     | Genz & Monahan (1999)          |  1135.52|  1124.81|  1293.56|        |        |         |
|     | Genz & Monahan (1999) Adaptive |   101.77|    87.08|   153.86|  200.69|  247.10|   303.41|
| 16  | AGHQ                           |     0.13|     0.48|     3.24|   19.23|  130.25|   981.49|
|     | CDF                            |    34.20|    36.01|    38.96|   37.28|   34.33|    34.84|
|     | Genz & Monahan (1999) Adaptive |    69.09|   109.33|   109.39|   99.70|  134.08|   276.05|
| 32  | AGHQ                           |     0.19|     0.66|     3.98|   30.29|  188.14|  1106.73|
|     | CDF                            |   142.39|   169.18|   298.72|  112.39|  114.96|   129.43|
|     | Genz & Monahan (1999) Adaptive |    42.68|    20.55|    24.17|   47.83|   56.54|    79.30|

``` r
show_run_times(TRUE)
```

**Only showing complete cases (means)**

| n   | method/p                       |        2|        3|        4|    5|    6|    7|
|:----|:-------------------------------|--------:|--------:|--------:|----:|----:|----:|
| 2   | GHQ                            |     0.04|     0.16|     0.94|     |     |     |
|     | AGHQ                           |     0.05|     0.11|     0.47|     |     |     |
|     | CDF                            |     0.04|     0.04|     0.04|     |     |     |
|     | Genz & Monahan (1999)          |   219.69|   213.96|   197.39|     |     |     |
|     | Genz & Monahan (1999) Adaptive |    24.96|    26.52|    22.59|     |     |     |
| 4   | GHQ                            |     0.08|     0.43|     2.49|     |     |     |
|     | AGHQ                           |     0.05|     0.17|     0.88|     |     |     |
|     | CDF                            |     0.81|     1.00|     0.94|     |     |     |
|     | Genz & Monahan (1999)          |   471.03|   497.66|   500.00|     |     |     |
|     | Genz & Monahan (1999) Adaptive |    25.84|    53.33|    74.86|     |     |     |
| 8   | GHQ                            |     0.12|     1.23|    14.33|     |     |     |
|     | AGHQ                           |     0.07|     0.24|     1.41|     |     |     |
|     | CDF                            |     4.75|     5.03|     5.01|     |     |     |
|     | Genz & Monahan (1999)          |  1135.52|  1124.81|  1293.56|     |     |     |
|     | Genz & Monahan (1999) Adaptive |    24.08|    14.08|    19.69|     |     |     |
| 16  | GHQ                            |         |         |         |     |     |     |
|     | AGHQ                           |         |         |         |     |     |     |
|     | CDF                            |         |         |         |     |     |     |
|     | Genz & Monahan (1999)          |         |         |         |     |     |     |
|     | Genz & Monahan (1999) Adaptive |         |         |         |     |     |     |
| 32  | GHQ                            |         |         |         |     |     |     |
|     | AGHQ                           |         |         |         |     |     |     |
|     | CDF                            |         |         |         |     |     |     |
|     | Genz & Monahan (1999)          |         |         |         |     |     |     |
|     | Genz & Monahan (1999) Adaptive |         |         |         |     |     |     |

**Number of complete cases**

|     |    2|    3|    4|    5|    6|    7|
|-----|----:|----:|----:|----:|----:|----:|
| 2   |   78|   58|   58|    0|    0|    0|
| 4   |   64|   56|   54|    0|    0|    0|
| 8   |   45|   39|   27|    0|    0|    0|
| 16  |    0|    0|    0|    0|    0|    0|
| 32  |    0|    0|    0|    0|    0|    0|

``` r

# show medians instead
med_func <- function(x, na.rm)
  apply(x, 1, median, na.rm = na.rm)
show_run_times(meth = med_func, suffix = " (median)", FALSE)
```

**Blank cells have at least one failure (median)**

| n   | method/p                       |      2|      3|      4|      5|       6|       7|
|:----|:-------------------------------|------:|------:|------:|------:|-------:|-------:|
| 2   | GHQ                            |   0.00|   0.20|   1.00|       |        |        |
|     | AGHQ                           |   0.00|   0.20|   0.60|   4.60|   30.40|  236.20|
|     | CDF                            |   0.00|   0.00|   0.00|   0.00|    0.00|    0.00|
|     | Genz & Monahan (1999)          |       |       |       |       |        |        |
|     | Genz & Monahan (1999) Adaptive |       |       |       |       |        |        |
| 4   | GHQ                            |       |       |   2.70|       |        |        |
|     | AGHQ                           |   0.00|   0.20|   1.00|   7.40|   50.00|  376.00|
|     | CDF                            |   0.60|   0.60|   0.60|   0.60|    0.60|    0.40|
|     | Genz & Monahan (1999)          |       |       |       |       |        |        |
|     | Genz & Monahan (1999) Adaptive |       |       |       |       |        |        |
| 8   | GHQ                            |       |       |       |       |        |        |
|     | AGHQ                           |   0.00|   0.20|   1.80|  13.00|   90.40|  678.10|
|     | CDF                            |   4.80|   5.00|   5.00|   4.80|    4.80|    5.00|
|     | Genz & Monahan (1999)          |       |       |       |       |        |        |
|     | Genz & Monahan (1999) Adaptive |       |       |       |       |        |        |
| 16  | GHQ                            |       |       |       |       |        |        |
|     | AGHQ                           |   0.20|   0.40|   3.30|  24.20|   72.30|  854.40|
|     | CDF                            |  33.70|  33.80|  35.20|  33.70|   34.00|   33.80|
|     | Genz & Monahan (1999)          |       |       |       |       |        |        |
|     | Genz & Monahan (1999) Adaptive |       |       |  11.80|  26.20|        |        |
| 32  | GHQ                            |       |       |       |       |        |        |
|     | AGHQ                           |   0.20|   0.60|   3.80|  23.00|  138.40|  827.60|
|     | CDF                            |  71.00|  72.80|  71.10|  72.40|   71.80|   73.70|
|     | Genz & Monahan (1999)          |       |       |       |       |        |        |
|     | Genz & Monahan (1999) Adaptive |   2.80|   2.80|   3.00|   4.70|    6.30|    6.80|

``` r
show_run_times(meth = med_func, suffix = " (median)", na.rm = TRUE)
```

**NAs have been removed. Cells may not be comparable (median)**

| n   | method/p                       |       2|       3|        4|      5|       6|       7|
|:----|:-------------------------------|-------:|-------:|--------:|------:|-------:|-------:|
| 2   | GHQ                            |    0.00|    0.20|     1.00|       |        |        |
|     | AGHQ                           |    0.00|    0.20|     0.60|   4.60|   30.40|  236.20|
|     | CDF                            |    0.00|    0.00|     0.00|   0.00|    0.00|    0.00|
|     | Genz & Monahan (1999)          |   14.70|   62.80|    89.80|       |        |        |
|     | Genz & Monahan (1999) Adaptive |    0.40|    6.00|     6.80|   8.20|    9.00|   28.40|
| 4   | GHQ                            |    0.00|    0.40|     2.70|       |        |        |
|     | AGHQ                           |    0.00|    0.20|     1.00|   7.40|   50.00|  376.00|
|     | CDF                            |    0.60|    0.60|     0.60|   0.60|    0.60|    0.40|
|     | Genz & Monahan (1999)          |  162.60|  232.20|   294.10|       |        |        |
|     | Genz & Monahan (1999) Adaptive |    0.60|   13.70|    16.40|  28.80|   29.80|   59.40|
| 8   | GHQ                            |    0.20|    1.60|    10.60|       |        |        |
|     | AGHQ                           |    0.00|    0.20|     1.80|  13.00|   90.40|  678.10|
|     | CDF                            |    4.80|    5.00|     5.00|   4.80|    4.80|    5.00|
|     | Genz & Monahan (1999)          |  689.00|  766.40|  1025.60|       |        |        |
|     | Genz & Monahan (1999) Adaptive |    1.60|    7.60|    12.40|  36.20|   85.00|   68.80|
| 16  | AGHQ                           |    0.20|    0.40|     3.30|  24.20|   72.30|  854.40|
|     | CDF                            |   33.70|   33.80|    35.20|  33.70|   34.00|   33.80|
|     | Genz & Monahan (1999) Adaptive |    1.40|    1.60|    11.80|  26.20|   32.00|   45.40|
| 32  | AGHQ                           |    0.20|    0.60|     3.80|  23.00|  138.40|  827.60|
|     | CDF                            |   71.00|   72.80|    71.10|  72.40|   71.80|   73.70|
|     | Genz & Monahan (1999) Adaptive |    2.80|    2.80|     3.00|   4.70|    6.30|    6.80|

``` r
show_run_times(meth = med_func, suffix = " (median)", TRUE)
```

**Only showing complete cases (median)**

| n   | method/p                       |       2|       3|        4|    5|    6|    7|
|:----|:-------------------------------|-------:|-------:|--------:|----:|----:|----:|
| 2   | GHQ                            |    0.00|    0.20|     0.80|     |     |     |
|     | AGHQ                           |    0.00|    0.20|     0.40|     |     |     |
|     | CDF                            |    0.00|    0.00|     0.00|     |     |     |
|     | Genz & Monahan (1999)          |   14.70|   52.40|    85.20|     |     |     |
|     | Genz & Monahan (1999) Adaptive |    0.40|    0.40|     1.20|     |     |     |
| 4   | GHQ                            |    0.00|    0.40|     2.30|     |     |     |
|     | AGHQ                           |    0.00|    0.20|     1.00|     |     |     |
|     | CDF                            |    0.40|    0.60|     0.40|     |     |     |
|     | Genz & Monahan (1999)          |  157.90|  232.20|   294.10|     |     |     |
|     | Genz & Monahan (1999) Adaptive |    0.60|    1.20|     6.20|     |     |     |
| 8   | GHQ                            |    0.20|    0.80|     5.00|     |     |     |
|     | AGHQ                           |    0.00|    0.20|     1.00|     |     |     |
|     | CDF                            |    4.80|    5.00|     5.00|     |     |     |
|     | Genz & Monahan (1999)          |  689.00|  766.40|  1025.60|     |     |     |
|     | Genz & Monahan (1999) Adaptive |    0.80|    0.80|     2.00|     |     |     |
| 16  | GHQ                            |        |        |         |     |     |     |
|     | AGHQ                           |        |        |         |     |     |     |
|     | CDF                            |        |        |         |     |     |     |
|     | Genz & Monahan (1999)          |        |        |         |     |     |     |
|     | Genz & Monahan (1999) Adaptive |        |        |         |     |     |     |
| 32  | GHQ                            |        |        |         |     |     |     |
|     | AGHQ                           |        |        |         |     |     |     |
|     | CDF                            |        |        |         |     |     |     |
|     | Genz & Monahan (1999)          |        |        |         |     |     |     |
|     | Genz & Monahan (1999) Adaptive |        |        |         |     |     |     |

**Number of complete cases**

|     |    2|    3|    4|    5|    6|    7|
|-----|----:|----:|----:|----:|----:|----:|
| 2   |   78|   58|   58|    0|    0|    0|
| 4   |   64|   56|   54|    0|    0|    0|
| 8   |   45|   39|   27|    0|    0|    0|
| 16  |    0|    0|    0|    0|    0|    0|
| 32  |    0|    0|    0|    0|    0|    0|

``` r

# show quantiles instead
med_func <- function(x, prob = .75, ...)
  apply(x, 1, function(z) quantile(na.omit(z), probs = prob))
show_run_times(meth = med_func, suffix = " (75% quantile)", na.rm = TRUE)
```

**NAs have been removed. Cells may not be comparable (75% quantile)**

| n   | method/p                       |        2|        3|        4|       5|       6|        7|
|:----|:-------------------------------|--------:|--------:|--------:|-------:|-------:|--------:|
| 2   | GHQ                            |     0.00|     0.40|     1.60|        |        |         |
|     | AGHQ                           |     0.00|     0.20|     0.80|    4.80|   31.60|   250.15|
|     | CDF                            |     0.00|     0.00|     0.00|    0.00|    0.00|     0.00|
|     | Genz & Monahan (1999)          |   295.50|   323.60|   317.90|        |        |         |
|     | Genz & Monahan (1999) Adaptive |    13.40|    60.25|    64.40|   55.85|   78.00|   151.80|
| 4   | GHQ                            |     0.20|     0.80|     5.80|        |        |         |
|     | AGHQ                           |     0.20|     0.20|     1.20|    7.80|   52.55|   394.10|
|     | CDF                            |     1.60|     1.25|     1.60|    1.00|    1.40|     1.20|
|     | Genz & Monahan (1999)          |   856.00|   780.00|   805.55|        |        |         |
|     | Genz & Monahan (1999) Adaptive |    28.20|    88.75|   128.40|  139.40|  126.50|   189.20|
| 8   | GHQ                            |     0.20|     2.60|    28.20|        |        |         |
|     | AGHQ                           |     0.20|     0.40|     2.00|   13.80|   94.85|   711.50|
|     | CDF                            |     5.20|     5.20|     5.20|    5.05|    5.00|     5.20|
|     | Genz & Monahan (1999)          |  2245.40|  1929.80|  2150.30|        |        |         |
|     | Genz & Monahan (1999) Adaptive |    29.60|    53.70|    72.00|  172.25|  272.40|   338.60|
| 16  | AGHQ                           |     0.20|     0.60|     3.80|   26.20|  181.20|  1323.85|
|     | CDF                            |    34.85|    34.80|    36.65|   35.00|   35.20|    35.20|
|     | Genz & Monahan (1999) Adaptive |     3.95|    26.20|    67.80|   80.35|  117.20|   177.60|
| 32  | AGHQ                           |     0.20|     1.00|     4.05|   49.80|  322.65|   889.10|
|     | CDF                            |    73.25|    76.20|    74.55|   74.60|   74.20|    75.45|
|     | Genz & Monahan (1999) Adaptive |     3.00|     3.20|    10.00|   57.55|   55.85|    58.85|

``` r
show_run_times(meth = med_func, suffix = " (75% quantile)", TRUE)
```

**Only showing complete cases (75% quantile)**

| n   | method/p                       |        2|        3|        4|    5|    6|    7|
|:----|:-------------------------------|--------:|--------:|--------:|----:|----:|----:|
| 2   | GHQ                            |     0.00|     0.20|     1.15|     |     |     |
|     | AGHQ                           |     0.00|     0.20|     0.60|     |     |     |
|     | CDF                            |     0.00|     0.00|     0.00|     |     |     |
|     | Genz & Monahan (1999)          |   295.50|   269.90|   273.60|     |     |     |
|     | Genz & Monahan (1999) Adaptive |     4.60|    12.45|    12.40|     |     |     |
| 4   | GHQ                            |     0.20|     0.60|     2.60|     |     |     |
|     | AGHQ                           |     0.00|     0.20|     1.00|     |     |     |
|     | CDF                            |     0.80|     0.65|     0.60|     |     |     |
|     | Genz & Monahan (1999)          |   778.15|   780.00|   805.55|     |     |     |
|     | Genz & Monahan (1999) Adaptive |    16.05|    31.65|    69.30|     |     |     |
| 8   | GHQ                            |     0.20|     1.70|     7.20|     |     |     |
|     | AGHQ                           |     0.20|     0.20|     1.80|     |     |     |
|     | CDF                            |     5.00|     5.20|     5.20|     |     |     |
|     | Genz & Monahan (1999)          |  2245.40|  1929.80|  2150.30|     |     |     |
|     | Genz & Monahan (1999) Adaptive |     5.20|     8.40|    17.60|     |     |     |
| 16  | GHQ                            |         |         |         |     |     |     |
|     | AGHQ                           |         |         |         |     |     |     |
|     | CDF                            |         |         |         |     |     |     |
|     | Genz & Monahan (1999)          |         |         |         |     |     |     |
|     | Genz & Monahan (1999) Adaptive |         |         |         |     |     |     |
| 32  | GHQ                            |         |         |         |     |     |     |
|     | AGHQ                           |         |         |         |     |     |     |
|     | CDF                            |         |         |         |     |     |     |
|     | Genz & Monahan (1999)          |         |         |         |     |     |     |
|     | Genz & Monahan (1999) Adaptive |         |         |         |     |     |     |

**Number of complete cases**

|     |    2|    3|    4|    5|    6|    7|
|-----|----:|----:|----:|----:|----:|----:|
| 2   |   78|   58|   58|    0|    0|    0|
| 4   |   64|   56|   54|    0|    0|    0|
| 8   |   45|   39|   27|    0|    0|    0|
| 16  |    0|    0|    0|    0|    0|    0|
| 32  |    0|    0|    0|    0|    0|    0|

``` r

#####
# mean scaled RMSE table
show_scaled_mean_rmse <- function(remove_nas = FALSE, na.rm = FALSE){
  # get mean scaled RMSE for the methods and the configurations pairs
  res <- sapply(ex_output, function(x)
    sapply(x[!names(x) %in% c("n", "p")], `[[`, "vals_n_comp_time", 
           simplify = "array"), 
    simplify = "array")
  err <- sqrt(res["mse", , , ])
  
  # scale by mean integral value
  mean_integral <- apply(res["mean", , , ], 2:3, mean, na.rm = TRUE)
  n_meth <- dim(err)[1]
  err <- err / rep(mean_integral, each = n_meth)
  
  is_complete <- t(apply(err, 2, function(x){
    if(remove_nas)
      apply(!is.na(x), 2, all)
    else 
      rep(TRUE, NCOL(x))
  }))
  dim(is_complete) <- dim(err)[2:3]
  
  err <- lapply(1:dim(err)[3], function(i){
    x <- err[, , i]
    x[, is_complete[, i]]
  })
  
  err <- sapply(err, rowMeans, na.rm = na.rm) * err_mult
  err[is.nan(err)] <- NA_real_
  
  # flatten the table. Start by getting the row labels
  meths <- rownames(err)
  n_labs <- sprintf("%2d", n_vals)
  rnames <- expand.grid(
    Method = meths, n = n_labs, stringsAsFactors = FALSE)
  rnames[2:1] <- rnames[1:2]
  rnames[[2L]] <- gsub(
    "^GenzMonahan$", "Genz & Monahan (1999)", rnames[[2L]])
  rnames[[2L]] <- gsub(
    "^GenzMonahanA$", "Genz & Monahan (1999) Adaptive", rnames[[2L]])
  # fix stupid typo at one point
  rnames[[2L]] <- gsub(
    "^ADHQ$", "AGHQ", rnames[[2L]])
  
  # then flatten
  err <- matrix(c(err), nrow = NROW(rnames))
  na_idx <- is.na(err)
  err[] <- sprintf("%.2f", err[])
  err[na_idx] <- NA_character_
  
  # combine mean mse and row labels
  table_out <- cbind(as.matrix(rnames), err)
  
  if(na.rm){
    keep <- apply(!is.na(table_out[, -(1:2), drop = FALSE]), 1, any)
    table_out <- table_out[keep, , drop = FALSE]
  }
  
  nvs <- table_out[, 1L]
  table_out[, 1L] <- c(
    nvs[1L], ifelse(nvs[-1L] != head(nvs, -1L), nvs[-1L], NA_integer_))
  
  # add header 
  p_labs <- sprintf("%d", p_vals)
  colnames(table_out) <- c("n", "method/p", p_labs)
  
  cat(.get_cap(remove_nas, na.rm))
  
  options(knitr.kable.NA = "")
  print(knitr::kable(
    table_out, align = c("l", "l", rep("r", length(p_vals)))))
  
  if(remove_nas)
    .show_n_complete(is_complete, n_labs, p_labs)
}

show_scaled_mean_rmse(FALSE)
```

**Blank cells have at least one failure**

| n   | method/p                       |      2|      3|      4|      5|      6|      7|
|:----|:-------------------------------|------:|------:|------:|------:|------:|------:|
| 2   | GHQ                            |   4.50|   4.51|   4.14|       |       |       |
|     | AGHQ                           |   3.39|   3.46|   2.84|   3.23|   3.07|   2.82|
|     | CDF                            |   0.59|   1.00|   1.01|   0.70|   0.80|   0.90|
|     | Genz & Monahan (1999)          |       |       |       |       |       |       |
|     | Genz & Monahan (1999) Adaptive |       |       |       |       |       |       |
| 4   | GHQ                            |       |       |   8.92|       |       |       |
|     | AGHQ                           |   5.99|   6.20|   5.74|   6.13|   5.63|   6.19|
|     | CDF                            |  10.81|  12.24|   9.23|  12.44|  10.83|  10.37|
|     | Genz & Monahan (1999)          |       |       |       |       |       |       |
|     | Genz & Monahan (1999) Adaptive |       |       |       |       |       |       |
| 8   | GHQ                            |       |       |       |       |       |       |
|     | AGHQ                           |  14.02|  13.55|  12.59|  11.43|  11.18|  10.61|
|     | CDF                            |  12.45|  13.09|  12.25|  12.46|  13.03|  12.22|
|     | Genz & Monahan (1999)          |       |       |       |       |       |       |
|     | Genz & Monahan (1999) Adaptive |       |       |       |       |       |       |
| 16  | GHQ                            |       |       |       |       |       |       |
|     | AGHQ                           |  27.30|  22.16|  21.38|  19.01|  21.73|  23.48|
|     | CDF                            |  15.98|  17.29|  17.63|  17.86|  17.99|  18.26|
|     | Genz & Monahan (1999)          |       |       |       |       |       |       |
|     | Genz & Monahan (1999) Adaptive |       |       |  39.70|  37.17|       |       |
| 32  | GHQ                            |       |       |       |       |       |       |
|     | AGHQ                           |  62.73|  66.97|  55.04|  40.61|  41.45|  41.00|
|     | CDF                            |  36.59|  46.14|  53.98|  62.96|  61.91|  64.95|
|     | Genz & Monahan (1999)          |       |       |       |       |       |       |
|     | Genz & Monahan (1999) Adaptive |  32.27|  46.21|  57.86|  62.26|  81.08|  79.52|

``` r
show_scaled_mean_rmse(na.rm = TRUE)
```

**NAs have been removed. Cells may not be comparable**

| n   | method/p                       |      2|      3|      4|      5|      6|      7|
|:----|:-------------------------------|------:|------:|------:|------:|------:|------:|
| 2   | GHQ                            |   4.50|   4.51|   4.14|       |       |       |
|     | AGHQ                           |   3.39|   3.46|   2.84|   3.23|   3.07|   2.82|
|     | CDF                            |   0.59|   1.00|   1.01|   0.70|   0.80|   0.90|
|     | Genz & Monahan (1999)          |   5.60|   5.05|   6.64|       |       |       |
|     | Genz & Monahan (1999) Adaptive |   3.21|   4.50|   4.65|   5.32|   5.97|   5.78|
| 4   | GHQ                            |  11.26|  10.39|   8.92|       |       |       |
|     | AGHQ                           |   5.99|   6.20|   5.74|   6.13|   5.63|   6.19|
|     | CDF                            |  10.81|  12.24|   9.23|  12.44|  10.83|  10.37|
|     | Genz & Monahan (1999)          |  12.26|  12.20|  10.42|       |       |       |
|     | Genz & Monahan (1999) Adaptive |   5.77|   9.12|   8.69|  11.05|  11.22|  10.81|
| 8   | GHQ                            |  23.72|  24.64|  22.30|       |       |       |
|     | AGHQ                           |  14.02|  13.55|  12.59|  11.43|  11.18|  10.61|
|     | CDF                            |  12.45|  13.09|  12.25|  12.46|  13.03|  12.22|
|     | Genz & Monahan (1999)          |  26.67|  26.07|  25.89|       |       |       |
|     | Genz & Monahan (1999) Adaptive |  14.39|  19.91|  20.27|  23.52|  19.22|  20.16|
| 16  | AGHQ                           |  27.30|  22.16|  21.38|  19.01|  21.73|  23.48|
|     | CDF                            |  15.98|  17.29|  17.63|  17.86|  17.99|  18.26|
|     | Genz & Monahan (1999) Adaptive |  19.35|  28.90|  39.70|  37.17|  42.14|  45.42|
| 32  | AGHQ                           |  62.73|  66.97|  55.04|  40.61|  41.45|  41.00|
|     | CDF                            |  36.59|  46.14|  53.98|  62.96|  61.91|  64.95|
|     | Genz & Monahan (1999) Adaptive |  32.27|  46.21|  57.86|  62.26|  81.08|  79.52|

``` r
show_scaled_mean_rmse(TRUE)
```

**Only showing complete cases**

| n   | method/p                       |      2|      3|      4|    5|    6|    7|
|:----|:-------------------------------|------:|------:|------:|----:|----:|----:|
| 2   | GHQ                            |   4.41|   3.91|   4.22|     |     |     |
|     | AGHQ                           |   3.42|   2.75|   2.33|     |     |     |
|     | CDF                            |   0.37|   0.45|   0.32|     |     |     |
|     | Genz & Monahan (1999)          |   5.60|   5.13|   6.73|     |     |     |
|     | Genz & Monahan (1999) Adaptive |   2.92|   3.79|   4.63|     |     |     |
| 4   | GHQ                            |   9.92|   9.70|   8.45|     |     |     |
|     | AGHQ                           |   5.67|   5.91|   5.17|     |     |     |
|     | CDF                            |  10.66|  12.71|   9.05|     |     |     |
|     | Genz & Monahan (1999)          |  12.37|  12.20|  10.42|     |     |     |
|     | Genz & Monahan (1999) Adaptive |   5.41|   8.53|   8.18|     |     |     |
| 8   | GHQ                            |  23.43|  24.72|  22.61|     |     |     |
|     | AGHQ                           |  14.64|  14.02|  15.37|     |     |     |
|     | CDF                            |   7.78|  10.57|   6.84|     |     |     |
|     | Genz & Monahan (1999)          |  26.67|  26.07|  25.89|     |     |     |
|     | Genz & Monahan (1999) Adaptive |  11.91|  16.62|  20.57|     |     |     |
| 16  | GHQ                            |       |       |       |     |     |     |
|     | AGHQ                           |       |       |       |     |     |     |
|     | CDF                            |       |       |       |     |     |     |
|     | Genz & Monahan (1999)          |       |       |       |     |     |     |
|     | Genz & Monahan (1999) Adaptive |       |       |       |     |     |     |
| 32  | GHQ                            |       |       |       |     |     |     |
|     | AGHQ                           |       |       |       |     |     |     |
|     | CDF                            |       |       |       |     |     |     |
|     | Genz & Monahan (1999)          |       |       |       |     |     |     |
|     | Genz & Monahan (1999) Adaptive |       |       |       |     |     |     |

**Number of complete cases**

|     |    2|    3|    4|    5|    6|    7|
|-----|----:|----:|----:|----:|----:|----:|
| 2   |   78|   58|   58|    0|    0|    0|
| 4   |   64|   56|   54|    0|    0|    0|
| 8   |   45|   39|   27|    0|    0|    0|
| 16  |    0|    0|    0|    0|    0|    0|
| 32  |    0|    0|    0|    0|    0|    0|

``` r

#####
# (A)GHQ node table
show_n_nodes <- function(adaptive){
  b_use_name <- if(adaptive) "b_use_A" else "b_use"
  
  # get the number of nodes that we use
  res <- sapply(ex_output, function(x)
    sapply(x[!names(x) %in% c("n", "p")], `[[`, b_use_name))
  
  # compute the quantiles
  probs <- seq(0, 1, length.out = 5)
  is_ok <- !is.na(res)
  qs <- lapply(1:dim(res)[2], function(i) res[is_ok[, i], i])
  qs <- sapply(qs, quantile, prob = probs)
  
  # flatten the table. Start by getting the row labels
  meths <- rownames(qs)
  n_labs <- sprintf("%2d", n_vals)
  rnames <- expand.grid(
    Method = meths, n = n_labs, stringsAsFactors = FALSE)
  rnames[2:1] <- rnames[1:2]
  
  # then flatten
  qs <- matrix(c(qs), nrow = NROW(rnames))
  na_idx <- is.na(qs)
  qs[] <- sprintf("%.2f", qs[])
  qs[na_idx] <- NA_character_
  
  # combine mean mse and row labels
  table_out <- cbind(as.matrix(rnames), qs)
  
  keep <- apply(!is.na(table_out[, -(1:2), drop = FALSE]), 1, any)
    table_out <- table_out[keep, , drop = FALSE]
  nvs <- table_out[, 1L]
  table_out[, 1L] <- c(
    nvs[1L], ifelse(nvs[-1L] != head(nvs, -1L), nvs[-1L], NA_integer_))
  
  # add header 
  p_labs <- sprintf("%d", p_vals)
  colnames(table_out) <- c("n", "quantile/p", p_labs)
  
  cat(.get_cap(TRUE, FALSE, if(adaptive) " (Adaptive GHQ)" else " (GHQ)"))
  
  options(knitr.kable.NA = "")
  print(knitr::kable(
    table_out, align = c("l", "l", rep("r", length(p_vals)))))
  
  .show_n_complete(is_ok, n_labs, p_labs)
}

show_n_nodes(FALSE)
```

**Only showing complete cases (GHQ)**

| n   | quantile/p |      2|      3|      4|    5|    6|    7|
|:----|:-----------|------:|------:|------:|----:|----:|----:|
| 2   | 0%         |   5.00|   5.00|   5.00|     |     |     |
|     | 25%        |   7.00|   8.00|   8.00|     |     |     |
|     | 50%        |   9.00|   9.00|   8.00|     |     |     |
|     | 75%        |  11.00|  11.00|   9.00|     |     |     |
|     | 100%       |  20.00|  24.00|  19.00|     |     |     |
| 4   | 0%         |   6.00|   6.00|   6.00|     |     |     |
|     | 25%        |   8.75|   9.00|   8.00|     |     |     |
|     | 50%        |  11.00|  10.00|   9.00|     |     |     |
|     | 75%        |  15.00|  12.00|  11.00|     |     |     |
|     | 100%       |  23.00|  23.00|  22.00|     |     |     |
| 8   | 0%         |   7.00|   7.00|   7.00|     |     |     |
|     | 25%        |  12.00|  10.00|  10.00|     |     |     |
|     | 50%        |  14.00|  13.00|  11.00|     |     |     |
|     | 75%        |  18.00|  15.00|  14.00|     |     |     |
|     | 100%       |  25.00|  24.00|  24.00|     |     |     |

**Number of complete cases**

|     |    2|    3|    4|    5|    6|    7|
|-----|----:|----:|----:|----:|----:|----:|
| 2   |  100|  100|  100|    0|    0|    0|
| 4   |   96|   99|  100|    0|    0|    0|
| 8   |   91|   98|   97|    0|    0|    0|
| 16  |    0|    0|    0|    0|    0|    0|
| 32  |    0|    0|    0|    0|    0|    0|

``` r
show_n_nodes(TRUE)
```

**Only showing complete cases (Adaptive GHQ)**

| n   | quantile/p |      2|      3|      4|      5|      6|     7|
|:----|:-----------|------:|------:|------:|------:|------:|-----:|
| 2   | 0%         |   4.00|   4.00|   6.00|   4.00|   4.00|  4.00|
|     | 25%        |   6.00|   6.00|   6.00|   6.00|   6.00|  6.00|
|     | 50%        |   7.00|   7.00|   7.00|   7.00|   7.00|  7.00|
|     | 75%        |   8.00|   8.00|   7.00|   7.00|   7.00|  7.00|
|     | 100%       |  10.00|  11.00|  10.00|  10.00|  10.00|  8.00|
| 4   | 0%         |   4.00|   4.00|   5.00|   6.00|   6.00|  5.00|
|     | 25%        |   6.00|   6.00|   6.00|   6.75|   6.00|  6.00|
|     | 50%        |   7.00|   7.00|   7.00|   7.00|   7.00|  7.00|
|     | 75%        |   7.00|   8.00|   7.00|   7.00|   7.00|  7.00|
|     | 100%       |  12.00|  10.00|  11.00|  10.00|   9.00|  9.00|
| 8   | 0%         |   4.00|   4.00|   4.00|   5.00|   6.00|  5.00|
|     | 25%        |   6.00|   6.00|   6.00|   6.00|   6.00|  6.00|
|     | 50%        |   7.00|   7.00|   7.00|   7.00|   7.00|  7.00|
|     | 75%        |   7.00|   7.00|   7.00|   7.00|   7.00|  7.00|
|     | 100%       |  14.00|  10.00|  10.00|  10.00|   9.00|  9.00|
| 16  | 0%         |   4.00|   4.00|   4.00|   5.00|   6.00|  5.00|
|     | 25%        |   5.75|   6.00|   6.00|   6.00|   6.00|  6.00|
|     | 50%        |   6.00|   7.00|   7.00|   7.00|   6.00|  6.50|
|     | 75%        |   7.00|   7.00|   7.00|   7.00|   7.00|  7.00|
|     | 100%       |   9.00|   9.00|  10.00|   8.00|   9.00|  9.00|
| 32  | 0%         |   4.00|   4.00|   4.00|   4.00|   5.00|  4.00|
|     | 25%        |   4.00|   5.00|   5.75|   6.00|   6.00|  6.00|
|     | 50%        |   5.00|   6.00|   6.00|   6.00|   6.00|  6.00|
|     | 75%        |   7.00|   6.25|   6.00|   7.00|   7.00|  6.00|
|     | 100%       |  10.00|   8.00|   7.00|   7.00|   7.00|  7.00|

**Number of complete cases**

|     |    2|    3|    4|    5|    6|    7|
|-----|----:|----:|----:|----:|----:|----:|
| 2   |  100|  100|  100|  100|  100|  100|
| 4   |  100|  100|  100|  100|  100|  100|
| 8   |  100|  100|  100|  100|  100|  100|
| 16  |  100|  100|  100|  100|  100|  100|
| 32  |  100|  100|  100|  100|  100|  100|

References
----------

Barrett, Jessica, Peter Diggle, Robin Henderson, and David Taylor-Robinson. 2015. “Joint Modelling of Repeated Measurements and Time-to-Event Outcomes: Flexible Model Specification and Exact Likelihood Inference.” *Journal of the Royal Statistical Society: Series B (Statistical Methodology)* 77 (1): 131–48. doi:[10.1111/rssb.12060](https://doi.org/10.1111/rssb.12060).

Genz, Alan, and Frank Bretz. 2002. “Comparison of Methods for the Computation of Multivariate T Probabilities.” *Journal of Computational and Graphical Statistics* 11 (4). Taylor & Francis: 950–71. doi:[10.1198/106186002394](https://doi.org/10.1198/106186002394).

Genz, Alan, and John Monahan. 1999. “A Stochastic Algorithm for High-Dimensional Integrals over Unbounded Regions with Gaussian Weight.” *Journal of Computational and Applied Mathematics* 112 (1): 71–81. doi:[https://doi.org/10.1016/S0377-0427(99)00214-9](https://doi.org/https://doi.org/10.1016/S0377-0427(99)00214-9).

Genz, Alan., and John. Monahan. 1998. “Stochastic Integration Rules for Infinite Regions.” *SIAM Journal on Scientific Computing* 19 (2): 426–39. doi:[10.1137/S1064827595286803](https://doi.org/10.1137/S1064827595286803).

Hajivassiliou, Vassilis, Daniel McFadden, and Paul Ruud. 1996. “Simulation of Multivariate Normal Rectangle Probabilities and Their Derivatives Theoretical and Computational Results.” *Journal of Econometrics* 72 (1): 85–134. doi:[https://doi.org/10.1016/0304-4076(94)01716-6](https://doi.org/https://doi.org/10.1016/0304-4076(94)01716-6).

Liu, Qing, and Donald A. Pierce. 1994. “A Note on Gauss-Hermite Quadrature.” *Biometrika* 81 (3). \[Oxford University Press, Biometrika Trust\]: 624–29. <http://www.jstor.org/stable/2337136>.

Ochi, Y., and Ross L. Prentice. 1984. “Likelihood Inference in a Correlated Probit Regression Model.” *Biometrika* 71 (3). \[Oxford University Press, Biometrika Trust\]: 531–43. <http://www.jstor.org/stable/2336562>.

Pawitan, Y., M. Reilly, E. Nilsson, S. Cnattingius, and P. Lichtenstein. 2004. “Estimation of Genetic and Environmental Factors for Binary Traits Using Family Data.” *Statistics in Medicine* 23 (3): 449–65. doi:[10.1002/sim.1603](https://doi.org/10.1002/sim.1603).
