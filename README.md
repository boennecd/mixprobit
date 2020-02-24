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

The [quick comparison](#quick-comparison) section may be skipped unless you want to get a grasp at what is implemented and see the definitions of the functions that is used in this markdown. The [more rigorous comparison](#more-rigorous-comparison) section is the main section of this markdown. It contains an example where we vary the number of observed outcomes, `n`, and the number of random effect, `p`, while considering the computation time of various approximation methods for a fixed relative error.

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
#> [1] 2.056
var(replicate(1000, with(get_sim_dat(10, 3), u %*% Z + eta)))
#> [1] 2.009
var(replicate(1000, with(get_sim_dat(10, 4), u %*% Z + eta)))
#> [1] 2.043
var(replicate(1000, with(get_sim_dat(10, 5), u %*% Z + eta)))
#> [1] 1.989
var(replicate(1000, with(get_sim_dat(10, 6), u %*% Z + eta)))
#> [1] 1.947
var(replicate(1000, with(get_sim_dat(10, 7), u %*% Z + eta)))
#> [1] 2.045
var(replicate(1000, with(get_sim_dat(10, 8), u %*% Z + eta)))
#> [1] 1.937
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
#> [1] "Mean relative difference: 4.462e-05"
all.equal(truth, c(sim_Aaprx(2L)))
#> [1] "Mean relative difference: 0.0008423"
all.equal(truth, c(sim_Aaprx(3L)))
#> [1] "Mean relative difference: 0.0003868"
all.equal(truth, c(sim_Aaprx(4L)))
#> [1] "Mean relative difference: 0.001326"

# compare computations times
system.time(GHQ_R()) # way too slow (seconds!). Use C++ method instead
#>    user  system elapsed 
#>   1.374   0.000   1.375
microbenchmark::microbenchmark(
  `GHQ (C++)` = GHQ_cpp(), `AGHQ (C++)` = AGHQ_cpp(),
  `CDF` = cdf_aprx_R(), `CDF (C++)` = cdf_aprx_cpp(),
  `Genz & Monahan (1)` = sim_aprx(1L), `Genz & Monahan (2)` = sim_aprx(2L),
  `Genz & Monahan (3)` = sim_aprx(3L), `Genz & Monahan (4)` = sim_aprx(4L),
  `Genz & Monahan Adaptive (2)` = sim_Aaprx(2L),
  times = 10)
#> Unit: milliseconds
#>                         expr   min    lq  mean median    uq   max neval
#>                    GHQ (C++) 38.95 39.53 39.82  39.79 40.03 40.60    10
#>                   AGHQ (C++) 41.70 42.92 43.06  43.01 43.17 44.83    10
#>                          CDF 20.53 20.59 20.78  20.72 20.80 21.26    10
#>                    CDF (C++) 11.14 11.42 11.51  11.51 11.63 11.76    10
#>           Genz & Monahan (1) 29.36 29.65 30.43  30.09 30.15 33.61    10
#>           Genz & Monahan (2) 30.99 31.39 31.95  31.56 32.23 34.96    10
#>           Genz & Monahan (3) 29.97 30.06 30.43  30.46 30.60 30.93    10
#>           Genz & Monahan (4) 28.88 29.52 29.81  29.66 29.91 31.33    10
#>  Genz & Monahan Adaptive (2) 35.10 35.48 35.71  35.68 35.77 36.48    10
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
  max_maxpts = 2000000L, 
  releps = 1e-4,
  min_releps = 1e-6,
  key_use = 3L, 
  n_reps = 25L, 
  n_runs = 5L, 
  n_brute = 1e7, 
  n_brute_max = 2e8, 
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
  cdf_releps <- if(do_not_run)
    NA_integer_
  else local({
    releps_use <- releps * 10
    repeat {
      func <- wd(aprx$get_cdf_cpp(y = y, eta = eta, Z = Z, S = S, 
                                  maxpts = ex_params$max_maxpts, 
                                  abseps = -1, releps = releps_use))
      vals <- replicate(ex_params$n_reps, func())
      if(all(is_ok_func(vals)))
        break
      
      releps_use <- releps_use / 2
      if(releps_use < ex_params$min_releps){
        warning("found no releps for CDF method")
        releps_use <- NA_integer_
        break
      }
    }
    releps_use
  })
  
  cdf_func <- if(!is.na(cdf_releps))
    wd(aprx$get_cdf_cpp(y = y, eta = eta, Z = Z, S = S, 
                        maxpts = ex_params$max_maxpts, abseps = -1, 
                        releps = cdf_releps))
  else 
    NA
  
  # get function to use with Genz and Monahan method
  get_sim_maxpts <- function(meth){
    if(do_not_run)
      NA_integer_
    else local({
      maxpts <- 100L
      repeat {
        func <- wd(meth(y = y, eta = eta, Z = Z, S = S, maxpts = maxpts, 
                        abseps = -1, releps = releps / 10))
        vals <- replicate(ex_params$n_reps, func(key_use))
        if(all(is_ok_func(vals)))
          break
        
        maxpts <- maxpts * 4L
        if(maxpts > ex_params$max_maxpts){
          warning("found no maxpts for sim method")
          maxpts <- NA_integer_
          break
        }
      }
      maxpts
    })
  }
  
  sim_maxpts_use <- if(is_to_large_for_ghq) 
    NA_integer_ else get_sim_maxpts(aprx$get_sim_mth)
  sim_func <- if(!is.na(sim_maxpts_use))
    wd(aprx$get_sim_mth(y = y, eta = eta, Z = Z, S = S, 
                        maxpts = sim_maxpts_use, abseps = -1, 
                        releps = releps / 10))
  else 
    NA
  if(is.function(sim_func))
    formals(sim_func)$key <- key_use
  
  # do the same with the adaptive version
  Asim_maxpts_use <- get_sim_maxpts(aprx$get_Asim_mth)
  Asim_func <- if(!is.na(Asim_maxpts_use))
    wd(aprx$get_Asim_mth(y = y, eta = eta, Z = Z, S = S, 
                         maxpts = Asim_maxpts_use, abseps = -1, 
                         releps = releps / 10))
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
    sim_maxpts_use = sim_maxpts_use, Asim_maxpts_use = Asim_maxpts_use, 
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
    sprintf("         Genz & Monahan maxpts: %13d", x$sim_maxpts_use),
    sprintf("Adaptive Genz & Monahan maxpts: %13d", x$Asim_maxpts_use), 
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
#>                     CDF releps:    0.00012500
#>          Genz & Monahan maxpts:        409600
#> Adaptive Genz & Monahan maxpts:          6400
#>   Log-likelihood estiamte (SE):   -0.84934116 (0.00000362)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                            0.427693                      -0.849350
#> AGHQ                           0.427693                      -0.849351
#> CDF                            0.427696                      -0.849342
#> GenzMonahan                    0.427690                      -0.849356
#> GenzMonahanA                   0.427694                      -0.849347
#> 
#> SD & RMSE (/10000.00)
#>            GHQ      AGHQ      CDF GenzMonahan GenzMonahanA
#> SD   0.0866955 0.0192985 0.167973   0.0877849     0.146107
#> RMSE 0.1035563 0.0454279 0.220646   0.1089928     0.119360
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA 
#>       0.0000       0.0000       0.0004       0.1310       0.0022
sim_experiment(n = 10L, p = 2L, n_threads = 6L)
#> Warning in (function() {: found no node value
#> Warning in (function() {: found no maxpts for sim method
#>          # brute force samples:      10000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             7
#>                     CDF releps:    0.00100000
#>          Genz & Monahan maxpts:            NA
#> Adaptive Genz & Monahan maxpts:           400
#>   Log-likelihood estiamte (SE):  -13.08652877 (0.00000817)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                      0.00000207298                       -13.0865
#> CDF                       0.00000207295                       -13.0865
#> GenzMonahan                          NA                             NA
#> GenzMonahanA              0.00000207290                       -13.0866
#> 
#> SD & RMSE (/10000.00)
#>      GHQ           AGHQ           CDF GenzMonahan GenzMonahanA
#> SD    NA 0.000000503960 0.00000294602          NA 0.0000156992
#> RMSE  NA 0.000000585509 0.00000256665          NA 0.0000134619
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA 
#>           NA       0.0000       0.0124           NA       0.0004

sim_experiment(n = 3L , p = 5L, n_threads = 6L)
#>          # brute force samples:      10000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             7
#>                     CDF releps:    0.00050000
#>          Genz & Monahan maxpts:            NA
#> Adaptive Genz & Monahan maxpts:         25600
#>   Log-likelihood estiamte (SE):   -5.02513648 (0.00000972)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                         0.00657076                       -5.02513
#> CDF                          0.00657027                       -5.02520
#> GenzMonahan                          NA                             NA
#> GenzMonahanA                 0.00657067                       -5.02514
#> 
#> SD & RMSE (/10000.00)
#>      GHQ       AGHQ       CDF GenzMonahan GenzMonahanA
#> SD    NA 0.00180505 0.0105089          NA    0.0084589
#> RMSE  NA 0.00163584 0.0126665          NA    0.0103855
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA 
#>           NA       0.0050       0.0008           NA       0.0074
sim_experiment(n = 10L, p = 5L, n_threads = 6L)
#>          # brute force samples:      10000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             7
#>                     CDF releps:    0.00100000
#>          Genz & Monahan maxpts:            NA
#> Adaptive Genz & Monahan maxpts:        102400
#>   Log-likelihood estiamte (SE):   -9.75429434 (0.00002388)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                       0.0000580460                       -9.75428
#> CDF                        0.0000580426                       -9.75433
#> GenzMonahan                          NA                             NA
#> GenzMonahanA               0.0000580456                       -9.75428
#> 
#> SD & RMSE (/10000.00)
#>      GHQ         AGHQ          CDF GenzMonahan GenzMonahanA
#> SD    NA 0.0000297975 0.0000650707          NA  0.000142566
#> RMSE  NA 0.0000277245 0.0000376950          NA  0.000101115
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA 
#>           NA       0.0134       0.0118           NA       0.0796

sim_experiment(n = 3L , p = 7L, n_threads = 6L)
#>          # brute force samples:      10000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             6
#>                     CDF releps:    0.00100000
#>          Genz & Monahan maxpts:            NA
#> Adaptive Genz & Monahan maxpts:        409600
#>   Log-likelihood estiamte (SE):   -4.57831980 (0.00002514)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                          0.0102719                       -4.57834
#> CDF                           0.0102722                       -4.57832
#> GenzMonahan                          NA                             NA
#> GenzMonahanA                  0.0102722                       -4.57831
#> 
#> SD & RMSE (/10000.00)
#>      GHQ      AGHQ        CDF GenzMonahan GenzMonahanA
#> SD    NA 0.0182811 0.01579277          NA    0.0108150
#> RMSE  NA 0.0198978 0.00893431          NA    0.0164839
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA 
#>           NA       0.0838       0.0002           NA       0.1238
sim_experiment(n = 10L, p = 7L, n_threads = 6L)
#>          # brute force samples:      10000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             7
#>                     CDF releps:    0.00100000
#>          Genz & Monahan maxpts:            NA
#> Adaptive Genz & Monahan maxpts:        409600
#>   Log-likelihood estiamte (SE):   -9.18407002 (0.00003312)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                        0.000102658                       -9.18411
#> CDF                         0.000102670                       -9.18399
#> GenzMonahan                          NA                             NA
#> GenzMonahanA                0.000102656                       -9.18413
#> 
#> SD & RMSE (/10000.00)
#>      GHQ        AGHQ         CDF GenzMonahan GenzMonahanA
#> SD    NA 0.000127363 0.000201003          NA 0.0001753384
#> RMSE  NA 0.000160830 0.000150239          NA 0.0000975285
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA 
#>           NA       0.6802       0.0124           NA       0.3394
```

Next, we apply the method a number of times for a of combination of number of observations, `n`, and number of random effects, `p`.

``` r
# number of observations in the cluster
n_vals <- 2^(1:4)
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
  cl <- makeCluster(4L)
  on.exit(stopCluster(cl))
  clusterExport(cl, c("aprx", "get_sim_dat", "sim_experiment", "ex_params"))
  
  # run the experiment
  mapply(function(n, p){
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
      clusterExport(cl, c("n", "p", "prg_file"), envir = environment())    
      clusterSetRNGStream(cl)
      
      sim_out <- parLapply(cl, 1:n_runs, function(...){
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
  }, n = gr_vals$n, p = gr_vals$p, SIMPLIFY = FALSE)
})()
```

We create a table where we summarize the results below. First we start with the average computation time, then we show the mean scaled RMSE, and we end by looking at the number of nodes that we need to use with GHQ. The latter shows why GHQ becomes slower as the cluster size, `n`, increases. The computation time is in 1000s of a second, `comp_time_mult`. The mean scaled RMSE is multiplied by ![10^{5}](https://latex.codecogs.com/svg.latex?10%5E%7B5%7D "10^{5}"), `err_mult`.

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
  nvs <- rnames[[1L]]
  rnames[[1L]] <- c(
    nvs[1L], ifelse(nvs[-1L] != head(nvs, -1L), nvs[-1L], NA_integer_))
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
|     | Genz & Monahan (1999)          |   67|   59|   56|    0|    0|    0|
|     | Genz & Monahan (1999) Adaptive |   89|   91|   83|   79|   91|   86|
| 4   | GHQ                            |   98|   99|   99|    0|    0|    0|
|     | AGHQ                           |  100|  100|  100|  100|  100|  100|
|     | CDF                            |  100|  100|  100|  100|  100|  100|
|     | Genz & Monahan (1999)          |   58|   48|   43|    0|    0|    0|
|     | Genz & Monahan (1999) Adaptive |   95|   94|   86|   90|   93|   89|
| 8   | GHQ                            |   91|   98|   99|    0|    0|    0|
|     | AGHQ                           |  100|  100|  100|  100|  100|  100|
|     | CDF                            |  100|  100|  100|  100|  100|  100|
|     | Genz & Monahan (1999)          |   52|   29|   22|    0|    0|    0|
|     | Genz & Monahan (1999) Adaptive |   97|   94|   97|   92|   87|   97|
| 16  | GHQ                            |    0|    0|    0|    0|    0|    0|
|     | AGHQ                           |  100|  100|  100|  100|  100|  100|
|     | CDF                            |  100|  100|  100|  100|  100|  100|
|     | Genz & Monahan (1999)          |    0|    0|    0|    0|    0|    0|
|     | Genz & Monahan (1999) Adaptive |   98|   99|   99|   99|  100|   98|

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
  nvs <- rnames[[1L]]
  rnames[[1L]] <- c(
    nvs[1L], ifelse(nvs[-1L] != head(nvs, -1L), nvs[-1L], NA_integer_))
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

| n   | method/p                       |      2|      3|      4|      5|       6|       7|
|:----|:-------------------------------|------:|------:|------:|------:|-------:|-------:|
| 2   | GHQ                            |   0.05|   0.25|   2.12|       |        |        |
|     | AGHQ                           |   0.04|   0.12|   0.83|   5.57|   36.07|  328.28|
|     | CDF                            |   0.05|   0.04|   0.05|   0.04|    0.03|    0.04|
|     | Genz & Monahan (1999)          |       |       |       |       |        |        |
|     | Genz & Monahan (1999) Adaptive |       |       |       |       |        |        |
| 4   | GHQ                            |       |       |       |       |        |        |
|     | AGHQ                           |   0.05|   0.18|   1.17|   8.92|   59.74|  425.38|
|     | CDF                            |   1.23|   1.30|   1.24|   1.09|    1.33|    1.08|
|     | Genz & Monahan (1999)          |       |       |       |       |        |        |
|     | Genz & Monahan (1999) Adaptive |       |       |       |       |        |        |
| 8   | GHQ                            |       |       |       |       |        |        |
|     | AGHQ                           |   0.07|   0.31|   2.00|  15.06|  104.59|  588.03|
|     | CDF                            |   5.83|   6.07|   5.95|   7.11|    6.56|    5.81|
|     | Genz & Monahan (1999)          |       |       |       |       |        |        |
|     | Genz & Monahan (1999) Adaptive |       |       |       |       |        |        |
| 16  | GHQ                            |       |       |       |       |        |        |
|     | AGHQ                           |   0.10|   0.44|   2.83|  20.79|  135.16|  979.51|
|     | CDF                            |  46.83|  50.94|  44.61|  53.48|   55.63|   53.33|
|     | Genz & Monahan (1999)          |       |       |       |       |        |        |
|     | Genz & Monahan (1999) Adaptive |       |       |       |       |  157.04|        |

``` r
show_run_times(na.rm = TRUE)
```

**NAs have been removed. Cells may not be comparable (means)**

| n   | method/p                       |       2|       3|       4|       5|       6|       7|
|:----|:-------------------------------|-------:|-------:|-------:|-------:|-------:|-------:|
| 2   | GHQ                            |    0.05|    0.25|    2.12|        |        |        |
|     | AGHQ                           |    0.04|    0.12|    0.83|    5.57|   36.07|  328.28|
|     | CDF                            |    0.05|    0.04|    0.05|    0.04|    0.03|    0.04|
|     | Genz & Monahan (1999)          |  139.77|  152.38|  149.80|        |        |        |
|     | Genz & Monahan (1999) Adaptive |   72.05|   68.98|   86.22|   81.21|  109.23|  135.05|
| 4   | GHQ                            |    0.06|    0.65|    6.03|        |        |        |
|     | AGHQ                           |    0.05|    0.18|    1.17|    8.92|   59.74|  425.38|
|     | CDF                            |    1.23|    1.30|    1.24|    1.09|    1.33|    1.08|
|     | Genz & Monahan (1999)          |  215.18|  289.30|  289.29|        |        |        |
|     | Genz & Monahan (1999) Adaptive |   65.93|  107.77|  104.09|  155.61|  207.49|  242.63|
| 8   | GHQ                            |    0.16|    2.19|   25.48|        |        |        |
|     | AGHQ                           |    0.07|    0.31|    2.00|   15.06|  104.59|  588.03|
|     | CDF                            |    5.83|    6.07|    5.95|    7.11|    6.56|    5.81|
|     | Genz & Monahan (1999)          |  661.77|  594.18|  662.29|        |        |        |
|     | Genz & Monahan (1999) Adaptive |   87.24|  124.04|  217.67|  190.45|  243.58|  280.91|
| 16  | GHQ                            |        |        |        |        |        |        |
|     | AGHQ                           |    0.10|    0.44|    2.83|   20.79|  135.16|  979.51|
|     | CDF                            |   46.83|   50.94|   44.61|   53.48|   55.63|   53.33|
|     | Genz & Monahan (1999)          |        |        |        |        |        |        |
|     | Genz & Monahan (1999) Adaptive |   16.21|  107.44|  151.06|  106.17|  157.04|  279.11|

``` r
show_run_times(TRUE)
```

**Only showing complete cases (means)**

| n   | method/p                       |       2|       3|       4|    5|    6|    7|
|:----|:-------------------------------|-------:|-------:|-------:|----:|----:|----:|
| 2   | GHQ                            |    0.05|    0.16|    0.91|     |     |     |
|     | AGHQ                           |    0.04|    0.09|    0.52|     |     |     |
|     | CDF                            |    0.04|    0.04|    0.04|     |     |     |
|     | Genz & Monahan (1999)          |  139.77|  148.96|  145.31|     |     |     |
|     | Genz & Monahan (1999) Adaptive |   23.77|   43.47|   20.75|     |     |     |
| 4   | GHQ                            |    0.03|    0.34|    2.23|     |     |     |
|     | AGHQ                           |    0.04|    0.13|    0.78|     |     |     |
|     | CDF                            |    0.98|    0.89|    0.80|     |     |     |
|     | Genz & Monahan (1999)          |  215.18|  282.10|  280.95|     |     |     |
|     | Genz & Monahan (1999) Adaptive |   14.24|   30.66|   38.17|     |     |     |
| 8   | GHQ                            |    0.13|    0.90|    6.02|     |     |     |
|     | AGHQ                           |    0.06|    0.26|    1.54|     |     |     |
|     | CDF                            |    5.05|    5.09|    5.99|     |     |     |
|     | Genz & Monahan (1999)          |  661.77|  594.18|  662.29|     |     |     |
|     | Genz & Monahan (1999) Adaptive |   19.97|   25.56|   10.96|     |     |     |
| 16  | GHQ                            |        |        |        |     |     |     |
|     | AGHQ                           |        |        |        |     |     |     |
|     | CDF                            |        |        |        |     |     |     |
|     | Genz & Monahan (1999)          |        |        |        |     |     |     |
|     | Genz & Monahan (1999) Adaptive |        |        |        |     |     |     |

**Number of complete cases**

|     |    2|    3|    4|    5|    6|    7|
|-----|----:|----:|----:|----:|----:|----:|
| 2   |   67|   58|   55|    0|    0|    0|
| 4   |   58|   47|   42|    0|    0|    0|
| 8   |   52|   29|   22|    0|    0|    0|
| 16  |    0|    0|    0|    0|    0|    0|

``` r

# show medians instead
med_func <- function(x, na.rm)
  apply(x, 1, median, na.rm = na.rm)
show_run_times(meth = med_func, suffix = " (median)", FALSE)
```

**Blank cells have at least one failure (median)**

| n   | method/p                       |      2|      3|      4|      5|       6|        7|
|:----|:-------------------------------|------:|------:|------:|------:|-------:|--------:|
| 2   | GHQ                            |   0.00|   0.20|   1.20|       |        |         |
|     | AGHQ                           |   0.00|   0.20|   0.60|   4.60|   30.70|   239.60|
|     | CDF                            |   0.00|   0.00|   0.00|   0.00|    0.00|     0.00|
|     | Genz & Monahan (1999)          |       |       |       |       |        |         |
|     | Genz & Monahan (1999) Adaptive |       |       |       |       |        |         |
| 4   | GHQ                            |       |       |       |       |        |         |
|     | AGHQ                           |   0.00|   0.20|   1.00|   7.40|   52.40|   407.70|
|     | CDF                            |   0.80|   0.80|   0.80|   0.80|    0.70|     0.60|
|     | Genz & Monahan (1999)          |       |       |       |       |        |         |
|     | Genz & Monahan (1999) Adaptive |       |       |       |       |        |         |
| 8   | GHQ                            |       |       |       |       |        |         |
|     | AGHQ                           |   0.00|   0.20|   1.80|  14.00|   92.60|   632.90|
|     | CDF                            |   5.00|   4.80|   5.00|   5.40|    5.00|     5.20|
|     | Genz & Monahan (1999)          |       |       |       |       |        |         |
|     | Genz & Monahan (1999) Adaptive |       |       |       |       |        |         |
| 16  | GHQ                            |       |       |       |       |        |         |
|     | AGHQ                           |   0.10|   0.40|   3.20|  24.00|  165.70|  1321.10|
|     | CDF                            |  33.40|  33.40|  33.70|  35.20|   36.60|    39.00|
|     | Genz & Monahan (1999)          |       |       |       |       |        |         |
|     | Genz & Monahan (1999) Adaptive |       |       |       |       |   37.60|         |

``` r
show_run_times(meth = med_func, suffix = " (median)", na.rm = TRUE)
```

**NAs have been removed. Cells may not be comparable (median)**

| n   | method/p                       |       2|       3|       4|      5|       6|        7|
|:----|:-------------------------------|-------:|-------:|-------:|------:|-------:|--------:|
| 2   | GHQ                            |    0.00|    0.20|    1.20|       |        |         |
|     | AGHQ                           |    0.00|    0.20|    0.60|   4.60|   30.70|   239.60|
|     | CDF                            |    0.00|    0.00|    0.00|   0.00|    0.00|     0.00|
|     | Genz & Monahan (1999)          |   25.60|   80.60|   88.50|       |        |         |
|     | Genz & Monahan (1999) Adaptive |    1.80|    6.00|    7.20|   7.20|   25.20|    32.20|
| 4   | GHQ                            |    0.00|    0.40|    3.60|       |        |         |
|     | AGHQ                           |    0.00|    0.20|    1.00|   7.40|   52.40|   407.70|
|     | CDF                            |    0.80|    0.80|    0.80|   0.80|    0.70|     0.60|
|     | Genz & Monahan (1999)          |   43.90|  152.60|  156.60|       |        |         |
|     | Genz & Monahan (1999) Adaptive |    2.40|    2.80|   10.50|  45.00|   45.00|    75.20|
| 8   | GHQ                            |    0.20|    1.60|   10.80|       |        |         |
|     | AGHQ                           |    0.00|    0.20|    1.80|  14.00|   92.60|   632.90|
|     | CDF                            |    5.00|    4.80|    5.00|   5.40|    5.00|     5.20|
|     | Genz & Monahan (1999)          |  978.00|  284.00|  667.00|       |        |         |
|     | Genz & Monahan (1999) Adaptive |    0.40|   10.20|   19.20|  21.70|   70.20|    82.00|
| 16  | GHQ                            |        |        |        |       |        |         |
|     | AGHQ                           |    0.10|    0.40|    3.20|  24.00|  165.70|  1321.10|
|     | CDF                            |   33.40|   33.40|   33.70|  35.20|   36.60|    39.00|
|     | Genz & Monahan (1999)          |        |        |        |       |        |         |
|     | Genz & Monahan (1999) Adaptive |    0.60|    2.20|    9.00|  11.60|   37.60|    44.30|

``` r
show_run_times(meth = med_func, suffix = " (median)", TRUE)
```

**Only showing complete cases (median)**

| n   | method/p                       |       2|       3|       4|    5|    6|    7|
|:----|:-------------------------------|-------:|-------:|-------:|----:|----:|----:|
| 2   | GHQ                            |    0.00|    0.20|    0.80|     |     |     |
|     | AGHQ                           |    0.00|    0.00|    0.60|     |     |     |
|     | CDF                            |    0.00|    0.00|    0.00|     |     |     |
|     | Genz & Monahan (1999)          |   25.60|   79.40|   87.40|     |     |     |
|     | Genz & Monahan (1999) Adaptive |    0.40|    1.00|    2.00|     |     |     |
| 4   | GHQ                            |    0.00|    0.20|    1.90|     |     |     |
|     | AGHQ                           |    0.00|    0.20|    0.60|     |     |     |
|     | CDF                            |    0.60|    0.40|    0.60|     |     |     |
|     | Genz & Monahan (1999)          |   43.90|  150.60|  155.40|     |     |     |
|     | Genz & Monahan (1999) Adaptive |    0.20|    2.40|    0.80|     |     |     |
| 8   | GHQ                            |    0.20|    0.60|    5.30|     |     |     |
|     | AGHQ                           |    0.00|    0.20|    1.20|     |     |     |
|     | CDF                            |    5.00|    4.60|    5.00|     |     |     |
|     | Genz & Monahan (1999)          |  978.00|  284.00|  667.00|     |     |     |
|     | Genz & Monahan (1999) Adaptive |    0.20|    1.20|    4.70|     |     |     |
| 16  | GHQ                            |        |        |        |     |     |     |
|     | AGHQ                           |        |        |        |     |     |     |
|     | CDF                            |        |        |        |     |     |     |
|     | Genz & Monahan (1999)          |        |        |        |     |     |     |
|     | Genz & Monahan (1999) Adaptive |        |        |        |     |     |     |

**Number of complete cases**

|     |    2|    3|    4|    5|    6|    7|
|-----|----:|----:|----:|----:|----:|----:|
| 2   |   67|   58|   55|    0|    0|    0|
| 4   |   58|   47|   42|    0|    0|    0|
| 8   |   52|   29|   22|    0|    0|    0|
| 16  |    0|    0|    0|    0|    0|    0|

``` r

# show quantiles instead
med_func <- function(x, prob = .75, ...)
  apply(x, 1, function(z) quantile(na.omit(z), probs = prob))
show_run_times(meth = med_func, suffix = " (75% quantile)", na.rm = TRUE)
```

**NAs have been removed. Cells may not be comparable (75% quantile)**

| n   | method/p                       |        2|        3|        4|       5|       6|        7|
|:----|:-------------------------------|--------:|--------:|--------:|-------:|-------:|--------:|
| 2   | GHQ                            |     0.00|     0.25|     2.45|        |        |         |
|     | AGHQ                           |     0.00|     0.20|     1.00|    5.00|   32.85|   269.55|
|     | CDF                            |     0.20|     0.00|     0.20|    0.00|    0.00|     0.00|
|     | Genz & Monahan (1999)          |   388.50|   356.60|   328.25|        |        |         |
|     | Genz & Monahan (1999) Adaptive |    29.20|    30.40|   115.70|  111.70|  114.20|   127.20|
| 4   | GHQ                            |     0.20|     0.80|     7.60|        |        |         |
|     | AGHQ                           |     0.05|     0.20|     1.20|    7.80|   55.70|   429.20|
|     | CDF                            |     1.60|     1.85|     1.60|    1.20|    1.40|     1.20|
|     | Genz & Monahan (1999)          |   572.15|   593.35|   576.80|        |        |         |
|     | Genz & Monahan (1999) Adaptive |    45.60|   163.25|   126.25|  183.25|  189.40|   226.80|
| 8   | GHQ                            |     0.20|     2.60|    32.00|        |        |         |
|     | AGHQ                           |     0.20|     0.40|     2.00|   15.20|   97.85|   750.85|
|     | CDF                            |     5.20|     5.20|     5.20|    6.00|    5.40|     5.60|
|     | Genz & Monahan (1999)          |  1154.70|  1101.00|  1134.65|        |        |         |
|     | Genz & Monahan (1999) Adaptive |    18.20|    71.25|    81.20|   93.65|  201.50|   332.40|
| 16  | GHQ                            |         |         |         |        |        |         |
|     | AGHQ                           |     0.20|     0.60|     3.60|   26.20|  182.90|  1391.70|
|     | CDF                            |    37.30|    41.60|    37.75|   46.30|   52.10|    51.50|
|     | Genz & Monahan (1999)          |         |         |         |        |        |         |
|     | Genz & Monahan (1999) Adaptive |     8.90|    34.20|    91.30|   63.10|  155.35|   179.85|

``` r
show_run_times(meth = med_func, suffix = " (75% quantile)", TRUE)
```

**Only showing complete cases (75% quantile)**

| n   | method/p                       |        2|        3|        4|    5|    6|    7|
|:----|:-------------------------------|--------:|--------:|--------:|----:|----:|----:|
| 2   | GHQ                            |     0.00|     0.20|     1.20|     |     |     |
|     | AGHQ                           |     0.00|     0.20|     0.60|     |     |     |
|     | CDF                            |     0.00|     0.00|     0.00|     |     |     |
|     | Genz & Monahan (1999)          |   388.50|   356.90|   325.90|     |     |     |
|     | Genz & Monahan (1999) Adaptive |     7.40|    18.85|    17.10|     |     |     |
| 4   | GHQ                            |     0.00|     0.40|     3.40|     |     |     |
|     | AGHQ                           |     0.00|     0.20|     1.00|     |     |     |
|     | CDF                            |     1.15|     1.00|     1.00|     |     |     |
|     | Genz & Monahan (1999)          |   572.15|   589.90|   574.80|     |     |     |
|     | Genz & Monahan (1999) Adaptive |     2.80|    24.00|    10.70|     |     |     |
| 8   | GHQ                            |     0.20|     1.00|     7.00|     |     |     |
|     | AGHQ                           |     0.20|     0.20|     1.80|     |     |     |
|     | CDF                            |     5.20|     4.80|     5.15|     |     |     |
|     | Genz & Monahan (1999)          |  1154.70|  1101.00|  1134.65|     |     |     |
|     | Genz & Monahan (1999) Adaptive |     4.40|    18.80|    18.55|     |     |     |
| 16  | GHQ                            |         |         |         |     |     |     |
|     | AGHQ                           |         |         |         |     |     |     |
|     | CDF                            |         |         |         |     |     |     |
|     | Genz & Monahan (1999)          |         |         |         |     |     |     |
|     | Genz & Monahan (1999) Adaptive |         |         |         |     |     |     |

**Number of complete cases**

|     |    2|    3|    4|    5|    6|    7|
|-----|----:|----:|----:|----:|----:|----:|
| 2   |   67|   58|   55|    0|    0|    0|
| 4   |   58|   47|   42|    0|    0|    0|
| 8   |   52|   29|   22|    0|    0|    0|
| 16  |    0|    0|    0|    0|    0|    0|

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
  nvs <- rnames[[1L]]
  rnames[[1L]] <- c(
    nvs[1L], ifelse(nvs[-1L] != head(nvs, -1L), nvs[-1L], NA_integer_))
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
| 2   | GHQ                            |   5.31|   4.37|   4.58|       |       |       |
|     | AGHQ                           |   2.72|   3.22|   3.08|   2.66|   2.90|   3.08|
|     | CDF                            |   0.90|   0.80|   1.14|   0.76|   0.77|   0.91|
|     | Genz & Monahan (1999)          |       |       |       |       |       |       |
|     | Genz & Monahan (1999) Adaptive |       |       |       |       |       |       |
| 4   | GHQ                            |       |       |       |       |       |       |
|     | AGHQ                           |   7.19|   6.00|   6.91|   6.91|   6.09|   6.24|
|     | CDF                            |   9.33|   9.41|   9.75|  11.53|  10.63|  10.74|
|     | Genz & Monahan (1999)          |       |       |       |       |       |       |
|     | Genz & Monahan (1999) Adaptive |       |       |       |       |       |       |
| 8   | GHQ                            |       |       |       |       |       |       |
|     | AGHQ                           |  14.08|  11.83|  14.05|  11.90|  10.21|  12.68|
|     | CDF                            |  10.47|  12.09|  13.67|  12.33|  13.10|  12.04|
|     | Genz & Monahan (1999)          |       |       |       |       |       |       |
|     | Genz & Monahan (1999) Adaptive |       |       |       |       |       |       |
| 16  | GHQ                            |       |       |       |       |       |       |
|     | AGHQ                           |  27.69|  22.97|  20.03|  21.23|  22.35|  22.46|
|     | CDF                            |  11.89|  15.93|  15.24|  15.92|  13.89|  16.60|
|     | Genz & Monahan (1999)          |       |       |       |       |       |       |
|     | Genz & Monahan (1999) Adaptive |       |       |       |       |  41.89|       |

``` r
show_scaled_mean_rmse(na.rm = TRUE)
```

**NAs have been removed. Cells may not be comparable**

| n   | method/p                       |      2|      3|      4|      5|      6|      7|
|:----|:-------------------------------|------:|------:|------:|------:|------:|------:|
| 2   | GHQ                            |   5.31|   4.37|   4.58|       |       |       |
|     | AGHQ                           |   2.72|   3.22|   3.08|   2.66|   2.90|   3.08|
|     | CDF                            |   0.90|   0.80|   1.14|   0.76|   0.77|   0.91|
|     | Genz & Monahan (1999)          |   4.84|   5.37|   6.33|       |       |       |
|     | Genz & Monahan (1999) Adaptive |   4.26|   4.88|   5.16|   4.52|   5.53|   5.42|
| 4   | GHQ                            |  11.88|   9.89|  10.50|       |       |       |
|     | AGHQ                           |   7.19|   6.00|   6.91|   6.91|   6.09|   6.24|
|     | CDF                            |   9.33|   9.41|   9.75|  11.53|  10.63|  10.74|
|     | Genz & Monahan (1999)          |  14.02|  11.73|  13.68|       |       |       |
|     | Genz & Monahan (1999) Adaptive |   8.98|   9.28|  11.58|  10.89|  11.61|  11.41|
| 8   | GHQ                            |  27.76|  24.51|  25.35|       |       |       |
|     | AGHQ                           |  14.08|  11.83|  14.05|  11.90|  10.21|  12.68|
|     | CDF                            |  10.47|  12.09|  13.67|  12.33|  13.10|  12.04|
|     | Genz & Monahan (1999)          |  29.38|  21.44|  26.38|       |       |       |
|     | Genz & Monahan (1999) Adaptive |  18.29|  17.62|  20.94|  20.44|  20.85|  20.02|
| 16  | GHQ                            |       |       |       |       |       |       |
|     | AGHQ                           |  27.69|  22.97|  20.03|  21.23|  22.35|  22.46|
|     | CDF                            |  11.89|  15.93|  15.24|  15.92|  13.89|  16.60|
|     | Genz & Monahan (1999)          |       |       |       |       |       |       |
|     | Genz & Monahan (1999) Adaptive |  27.54|  36.65|  35.62|  40.72|  41.89|  42.80|

``` r
show_scaled_mean_rmse(TRUE)
```

**Only showing complete cases**

| n   | method/p                       |      2|      3|      4|    5|    6|    7|
|:----|:-------------------------------|------:|------:|------:|----:|----:|----:|
| 2   | GHQ                            |   4.72|   3.97|   3.81|     |     |     |
|     | AGHQ                           |   2.07|   2.72|   2.29|     |     |     |
|     | CDF                            |   0.39|   0.53|   0.25|     |     |     |
|     | Genz & Monahan (1999)          |   4.84|   5.43|   6.42|     |     |     |
|     | Genz & Monahan (1999) Adaptive |   3.80|   4.50|   4.44|     |     |     |
| 4   | GHQ                            |  11.38|   9.47|   9.99|     |     |     |
|     | AGHQ                           |   6.55|   5.61|   5.66|     |     |     |
|     | CDF                            |   9.32|   9.17|   9.59|     |     |     |
|     | Genz & Monahan (1999)          |  14.02|  11.78|  13.80|     |     |     |
|     | Genz & Monahan (1999) Adaptive |   8.02|   8.55|  10.00|     |     |     |
| 8   | GHQ                            |  32.15|  19.90|  25.45|     |     |     |
|     | AGHQ                           |  17.12|  10.62|  15.84|     |     |     |
|     | CDF                            |   7.88|   8.05|  10.36|     |     |     |
|     | Genz & Monahan (1999)          |  29.38|  21.44|  26.38|     |     |     |
|     | Genz & Monahan (1999) Adaptive |  20.93|  17.05|  19.41|     |     |     |
| 16  | GHQ                            |       |       |       |     |     |     |
|     | AGHQ                           |       |       |       |     |     |     |
|     | CDF                            |       |       |       |     |     |     |
|     | Genz & Monahan (1999)          |       |       |       |     |     |     |
|     | Genz & Monahan (1999) Adaptive |       |       |       |     |     |     |

**Number of complete cases**

|     |    2|    3|    4|    5|    6|    7|
|-----|----:|----:|----:|----:|----:|----:|
| 2   |   67|   58|   55|    0|    0|    0|
| 4   |   58|   47|   42|    0|    0|    0|
| 8   |   52|   29|   22|    0|    0|    0|
| 16  |    0|    0|    0|    0|    0|    0|

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
  nvs <- rnames[[1L]]
  rnames[[1L]] <- c(
    nvs[1L], ifelse(nvs[-1L] != head(nvs, -1L), nvs[-1L], NA_integer_))
  
  # then flatten
  qs <- matrix(c(qs), nrow = NROW(rnames))
  na_idx <- is.na(qs)
  qs[] <- sprintf("%.2f", qs[])
  qs[na_idx] <- NA_character_
  
  # combine mean mse and row labels
  table_out <- cbind(as.matrix(rnames), qs)
  
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
| 2   | 0%         |   5.00|   5.00|   6.00|     |     |     |
|     | 25%        |   8.00|   8.00|   7.00|     |     |     |
|     | 50%        |  10.00|   9.00|   9.00|     |     |     |
|     | 75%        |  12.00|  11.00|  10.25|     |     |     |
|     | 100%       |  21.00|  17.00|  18.00|     |     |     |
| 4   | 0%         |   6.00|   6.00|   6.00|     |     |     |
|     | 25%        |   9.00|   9.00|   8.50|     |     |     |
|     | 50%        |  11.00|  10.00|  10.00|     |     |     |
|     | 75%        |  13.75|  13.00|  12.00|     |     |     |
|     | 100%       |  25.00|  23.00|  18.00|     |     |     |
| 8   | 0%         |   6.00|   8.00|   7.00|     |     |     |
|     | 25%        |  10.00|  10.00|  10.00|     |     |     |
|     | 50%        |  13.00|  13.00|  11.00|     |     |     |
|     | 75%        |  16.00|  16.00|  14.50|     |     |     |
|     | 100%       |  24.00|  25.00|  23.00|     |     |     |
| 16  | 0%         |       |       |       |     |     |     |
|     | 25%        |       |       |       |     |     |     |
|     | 50%        |       |       |       |     |     |     |
|     | 75%        |       |       |       |     |     |     |
|     | 100%       |       |       |       |     |     |     |

**Number of complete cases**

|     |    2|    3|    4|    5|    6|    7|
|-----|----:|----:|----:|----:|----:|----:|
| 2   |  100|  100|  100|    0|    0|    0|
| 4   |   98|   99|   99|    0|    0|    0|
| 8   |   91|   98|   99|    0|    0|    0|
| 16  |    0|    0|    0|    0|    0|    0|

``` r
show_n_nodes(TRUE)
```

**Only showing complete cases (Adaptive GHQ)**

| n   | quantile/p |      2|      3|      4|      5|      6|     7|
|:----|:-----------|------:|------:|------:|------:|------:|-----:|
| 2   | 0%         |   4.00|   4.00|   4.00|   5.00|   4.00|  4.00|
|     | 25%        |   6.00|   6.00|   6.00|   6.00|   6.00|  6.00|
|     | 50%        |   7.00|   7.00|   7.00|   7.00|   7.00|  7.00|
|     | 75%        |   8.00|   7.25|   8.00|   7.00|   7.00|  7.00|
|     | 100%       |  14.00|  11.00|  10.00|  10.00|  10.00|  9.00|
| 4   | 0%         |   4.00|   4.00|   5.00|   6.00|   5.00|  6.00|
|     | 25%        |   6.00|   6.00|   6.00|   6.75|   6.00|  6.75|
|     | 50%        |   7.00|   7.00|   7.00|   7.00|   7.00|  7.00|
|     | 75%        |   8.00|   8.00|   7.00|   7.00|   7.00|  7.00|
|     | 100%       |  14.00|  10.00|  11.00|  11.00|  10.00|  9.00|
| 8   | 0%         |   4.00|   4.00|   4.00|   5.00|   6.00|  6.00|
|     | 25%        |   6.00|   6.00|   6.00|   6.00|   6.00|  6.00|
|     | 50%        |   6.00|   7.00|   7.00|   7.00|   7.00|  7.00|
|     | 75%        |   7.00|   7.00|   7.00|   7.00|   7.00|  7.00|
|     | 100%       |  11.00|  10.00|   9.00|  10.00|   9.00|  9.00|
| 16  | 0%         |   4.00|   4.00|   5.00|   5.00|   5.00|  5.00|
|     | 25%        |   5.00|   6.00|   6.00|   6.00|   6.00|  6.00|
|     | 50%        |   6.00|   6.00|   7.00|   7.00|   7.00|  7.00|
|     | 75%        |   7.00|   7.00|   7.00|   7.00|   7.00|  7.00|
|     | 100%       |  12.00|   9.00|   9.00|   9.00|   8.00|  7.00|

**Number of complete cases**

|     |    2|    3|    4|    5|    6|    7|
|-----|----:|----:|----:|----:|----:|----:|
| 2   |  100|  100|  100|  100|  100|  100|
| 4   |  100|  100|  100|  100|  100|  100|
| 8   |  100|  100|  100|  100|  100|  100|
| 16  |  100|  100|  100|  100|  100|  100|

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
