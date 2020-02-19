Mixed Models with Probit Link
=============================

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
#> [1] 1.928
var(replicate(1000, with(get_sim_dat(10, 3), u %*% Z + eta)))
#> [1] 1.973
var(replicate(1000, with(get_sim_dat(10, 4), u %*% Z + eta)))
#> [1] 2
var(replicate(1000, with(get_sim_dat(10, 5), u %*% Z + eta)))
#> [1] 2.001
var(replicate(1000, with(get_sim_dat(10, 6), u %*% Z + eta)))
#> [1] 1.951
var(replicate(1000, with(get_sim_dat(10, 7), u %*% Z + eta)))
#> [1] 1.975
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
b <- 30L              # number of nodes to use with GHQ
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

all.equal(truth, c(truth_maybe1))
#> [1] "Mean relative difference: 0.0002567"
all.equal(truth, c(truth_maybe2))
#> [1] "Mean relative difference: 5.793e-05"
all.equal(truth, c(truth_maybe2_A))
#> [1] "Mean relative difference: 0.0003406"

# compare with using fewer samples and GHQ
all.equal(truth,   GHQ_R())
#> [1] "Mean relative difference: 0.000343"
all.equal(truth,   GHQ_cpp())
#> [1] "Mean relative difference: 0.000343"
all.equal(truth,   AGHQ_cpp())
#> [1] "Mean relative difference: 0.000343"
all.equal(truth, c(cdf_aprx_R()))
#> [1] "Mean relative difference: 0.0003967"
all.equal(truth, c(cdf_aprx_cpp()))
#> [1] "Mean relative difference: 0.000592"
all.equal(truth, c(sim_aprx(1L)))
#> [1] "Mean relative difference: 0.0154"
all.equal(truth, c(sim_aprx(2L)))
#> [1] "Mean relative difference: 0.004974"
all.equal(truth, c(sim_aprx(3L)))
#> [1] "Mean relative difference: 0.002775"
all.equal(truth, c(sim_aprx(4L)))
#> [1] "Mean relative difference: 0.00412"
all.equal(truth, c(sim_Aaprx(1L)))
#> [1] "Mean relative difference: 0.0003982"
all.equal(truth, c(sim_Aaprx(2L)))
#> [1] "Mean relative difference: 0.001196"
all.equal(truth, c(sim_Aaprx(3L)))
#> [1] "Mean relative difference: 0.0007405"
all.equal(truth, c(sim_Aaprx(4L)))
#> [1] "Mean relative difference: 0.00168"

# compare computations times
system.time(GHQ_R()) # way too slow (seconds!). Use C++ method instead
#>    user  system elapsed 
#>   21.36    0.00   21.36
microbenchmark::microbenchmark(
  `GHQ (C++)` = GHQ_cpp(), `AGHQ (C++)` = AGHQ_cpp(),
  `CDF` = cdf_aprx_R(), `CDF (C++)` = cdf_aprx_cpp(),
  `Genz & Monahan (1)` = sim_aprx(1L), `Genz & Monahan (2)` = sim_aprx(2L),
  `Genz & Monahan (3)` = sim_aprx(3L), `Genz & Monahan (4)` = sim_aprx(4L),
  `Genz & Monahan Adaptive (2)` = sim_Aaprx(2L),
  times = 10)
#> Unit: milliseconds
#>                         expr    min     lq   mean median     uq    max neval
#>                    GHQ (C++) 611.95 612.40 613.04 612.62 612.84 617.18    10
#>                   AGHQ (C++) 648.15 649.23 650.97 649.65 650.11 662.02    10
#>                          CDF  20.46  20.59  20.76  20.62  21.00  21.14    10
#>                    CDF (C++)  11.13  11.13  11.15  11.14  11.15  11.29    10
#>           Genz & Monahan (1)  28.35  28.56  29.04  28.98  29.43  29.88    10
#>           Genz & Monahan (2)  29.56  29.97  30.15  30.14  30.33  30.92    10
#>           Genz & Monahan (3)  28.77  28.82  29.16  29.04  29.49  29.77    10
#>           Genz & Monahan (4)  28.39  28.45  28.71  28.69  28.93  29.20    10
#>  Genz & Monahan Adaptive (2)  33.86  33.90  34.32  34.40  34.49  35.16    10
```

More Rigorous Comparison
------------------------

We are interested in a more rigorous comparison. Therefor, we define a function below which for given number of observation in the cluster, `n`, and given number of random effects, `p`, performs a repeated number of runs with each of the methods and returns the computation time (among other output). To make a fair comparison, we fix the relative error of the methods before hand such that the relative error is below `releps`, ![5\\times 10^{-4}](https://latex.codecogs.com/svg.latex?5%5Ctimes%2010%5E%7B-4%7D "5\times 10^{-4}"). Ground truth is computed with brute force MC using `n_brute`, ![10^{8}](https://latex.codecogs.com/svg.latex?10%5E%7B8%7D "10^{8}"), samples.

Since GHQ is deterministic, we use a number of nodes such that this number of nodes or `streak_length`, 4, less value of nodes with GHQ gives a relative error which is below the threshold. We use a minimum of 4 nodes at the time of this writing. The error of the simulation based methods is approximated using `n_reps`, 5, replications.

``` r
# default parameters
ex_params <- list(
  streak_length = 4L, 
  max_b = 30L, 
  max_maxpts = 1000000L, 
  releps = 5e-4,
  min_releps = 5e-6,
  key_use = 2L, 
  n_reps = 5L, 
  n_runs = 10L, 
  n_brute = 1e8)
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
sim_experiment <- function(n, p, releps = ex_params$releps, 
                           key_use = ex_params$key_use){
  # in some cases we may not want to run the simulation experiment
  do_not_run <- FALSE
  
  # simulate data
  dat <- get_sim_dat(n = n, p = p)
  
  # shorter than calling `with(dat, ...)`
  wd <- function(expr)
    eval(bquote(with(dat, .(substitute(expr)))), parent.frame())
  
  # get the assumed ground truth
  truth <- if(do_not_run)
    NA
  else wd(mixprobit:::aprx_binary_mix_brute(
    y = y, eta = eta, Z = Z, Sigma = S, n_sim = ex_params$n_brute))
  
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
      repeat({
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
      })
      b
    })
  }
  
  b_use <- get_b(aprx$get_GHQ_cpp)
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
    releps_use <- releps * 100
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
        
        maxpts <- maxpts * 2L
        if(maxpts > ex_params$max_maxpts){
          warning("found no maxpts for sim method")
          maxpts <- NA_integer_
          break
        }
      }
      maxpts
    })
  }
  
  sim_maxpts_use <- get_sim_maxpts(aprx$get_sim_mth)
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
      
      # perform the computations
      ti <- system.time(vals <- replicate(n_runs, func()))
      
      c(mean = mean(vals), sd = sd(vals), mse = mean((vals - truth)^2), 
        ti[1:3] / n_runs)            
    })
  
  list(b_use = b_use, b_use_A = b_use_A, cdf_releps = cdf_releps, 
       sim_maxpts_use = sim_maxpts_use, Asim_maxpts_use = Asim_maxpts_use, 
       vals_n_comp_time = out)
}
```

Here is a few quick examples where we use the function we just defined.

``` r
set.seed(1)
sim_experiment(n = 3L , p = 2L)
#> $b_use
#> [1] 8
#> 
#> $b_use_A
#> [1] 7
#> 
#> $cdf_releps
#> [1] 0.05
#> 
#> $sim_maxpts_use
#> [1] 25600
#> 
#> $Asim_maxpts_use
#> [1] 800
#> 
#> $vals_n_comp_time
#>                 GHQ      AGHQ       CDF GenzMonahan GenzMonahanA
#> mean      4.277e-01 4.277e-01 4.277e-01   4.276e-01    4.277e-01
#> sd        0.000e+00 0.000e+00 1.570e-05   1.444e-04    1.168e-04
#> mse       3.608e-10 4.003e-10 9.010e-10   2.540e-08    1.249e-08
#> user.self 1.000e-04 1.000e-04 2.000e-04   7.900e-03    3.000e-04
#> sys.self  0.000e+00 0.000e+00 0.000e+00   0.000e+00    0.000e+00
#> elapsed   0.000e+00 0.000e+00 2.000e-04   8.000e-03    3.000e-04
sim_experiment(n = 10L, p = 2L)
#> $b_use
#> [1] 14
#> 
#> $b_use_A
#> [1] 6
#> 
#> $cdf_releps
#> [1] 0.05
#> 
#> $sim_maxpts_use
#> [1] 819200
#> 
#> $Asim_maxpts_use
#> [1] 100
#> 
#> $vals_n_comp_time
#>                 GHQ      AGHQ       CDF GenzMonahan GenzMonahanA
#> mean      6.760e-05 6.752e-05 6.753e-05   6.747e-05    6.748e-05
#> sd        0.000e+00 0.000e+00 6.486e-09   2.204e-07    1.626e-07
#> mse       5.103e-15 3.736e-18 3.788e-17   4.700e-14    2.636e-14
#> user.self 2.000e-04 1.000e-04 1.190e-02   6.195e-01    1.000e-04
#> sys.self  0.000e+00 0.000e+00 0.000e+00   0.000e+00    0.000e+00
#> elapsed   2.000e-04 1.000e-04 1.190e-02   6.196e-01    1.000e-04
sim_experiment(n = 3L , p = 5L)
#> Warning in (function() {: found no maxpts for sim method
#> $b_use
#> [1] 8
#> 
#> $b_use_A
#> [1] 7
#> 
#> $cdf_releps
#> [1] 0.05
#> 
#> $sim_maxpts_use
#> [1] NA
#> 
#> $Asim_maxpts_use
#> [1] 409600
#> 
#> $vals_n_comp_time
#>                 GHQ      AGHQ       CDF GenzMonahan GenzMonahanA
#> mean      4.831e-01 4.831e-01 4.831e-01          NA    4.831e-01
#> sd        0.000e+00 0.000e+00 1.270e-05          NA    1.384e-04
#> mse       8.973e-12 7.000e-14 1.896e-10          NA    1.830e-08
#> user.self 8.700e-03 5.400e-03 1.000e-04          NA    1.569e-01
#> sys.self  0.000e+00 0.000e+00 0.000e+00          NA    0.000e+00
#> elapsed   8.700e-03 5.400e-03 1.000e-04          NA    1.569e-01
sim_experiment(n = 8L , p = 5L)
#> $b_use
#> [1] 10
#> 
#> $b_use_A
#> [1] 6
#> 
#> $cdf_releps
#> [1] 0.05
#> 
#> $sim_maxpts_use
#> [1] 409600
#> 
#> $Asim_maxpts_use
#> [1] 200
#> 
#> $vals_n_comp_time
#>                 GHQ      AGHQ       CDF GenzMonahan GenzMonahanA
#> mean      2.849e-05 2.849e-05 2.849e-05   2.845e-05    2.843e-05
#> sd        0.000e+00 0.000e+00 4.266e-09   9.996e-08    8.924e-08
#> mse       5.247e-18 3.958e-17 3.289e-17   1.064e-14    1.042e-14
#> user.self 6.300e-02 5.200e-03 4.500e-03   2.601e-01    2.000e-04
#> sys.self  0.000e+00 0.000e+00 0.000e+00   0.000e+00    0.000e+00
#> elapsed   6.300e-02 5.100e-03 4.500e-03   2.601e-01    2.000e-04
sim_experiment(n = 3L , p = 6L)
#> $b_use
#> [1] 12
#> 
#> $b_use_A
#> [1] 7
#> 
#> $cdf_releps
#> [1] 0.05
#> 
#> $sim_maxpts_use
#> [1] 409600
#> 
#> $Asim_maxpts_use
#> [1] 800
#> 
#> $vals_n_comp_time
#>                 GHQ      AGHQ       CDF GenzMonahan GenzMonahanA
#> mean      1.775e-03 1.775e-03 1.775e-03   1.775e-03    1.775e-03
#> sd        0.000e+00 0.000e+00 6.692e-07   2.468e-06    3.289e-06
#> mse       1.306e-12 1.002e-12 1.438e-12   6.227e-12    1.174e-11
#> user.self 7.265e-01 3.170e-02 2.000e-04   1.102e-01    2.000e-04
#> sys.self  0.000e+00 0.000e+00 0.000e+00   0.000e+00    0.000e+00
#> elapsed   7.265e-01 3.170e-02 3.000e-04   1.103e-01    2.000e-04
sim_experiment(n = 8L , p = 6L)
#> $b_use
#> [1] 10
#> 
#> $b_use_A
#> [1] 6
#> 
#> $cdf_releps
#> [1] 0.05
#> 
#> $sim_maxpts_use
#> [1] 409600
#> 
#> $Asim_maxpts_use
#> [1] 3200
#> 
#> $vals_n_comp_time
#>                 GHQ      AGHQ       CDF GenzMonahan GenzMonahanA
#> mean      4.187e-04 4.188e-04 4.188e-04   4.186e-04    4.189e-04
#> sd        0.000e+00 0.000e+00 8.770e-08   6.374e-07    6.339e-07
#> mse       7.912e-15 8.279e-16 7.786e-15   4.248e-13    3.706e-13
#> user.self 6.200e-01 2.960e-02 4.000e-03   2.640e-01    2.200e-03
#> sys.self  0.000e+00 0.000e+00 0.000e+00   0.000e+00    0.000e+00
#> elapsed   6.200e-01 2.950e-02 4.100e-03   2.641e-01    2.300e-03
```

Next, we apply the method a number of times for a of combination of number of observations, `n`, and number of random effects, `p`.

``` r
# number of observations in the cluster
n_vals <- 2^(1:5)
# number of random effects
p_vals <- 2:6
# grid with all configurations
gr_vals <- expand.grid(n = n_vals, p = p_vals)
# number of replications per configuration
n_runs <- 20L

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
      
      set.seed(71771946)
      clusterExport(cl, c("n", "p"), envir = environment())    
      clusterSetRNGStream(cl)
      
      sim_out <- parLapply(cl, 1:n_runs, function(...){
        seed <- .Random.seed
        out <- sim_experiment(n = n, p = p)
        attr(out, "seed") <- seed
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
# table with computation times
# util functions
.get_cap <- function(remove_nas, sufix = ""){
  cap <- if(remove_nas)
    "**Only showing complete cases"
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
show_run_times <- function(remove_nas = FALSE){
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
  comp_times <- sapply(comp_times, rowMeans) * comp_time_mult
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
  
  cat(.get_cap(remove_nas))
    
  options(knitr.kable.NA = "")
  print(knitr::kable(
    table_out, align = c("l", "l", rep("r", length(p_vals)))))
  
  if(remove_nas)
    .show_n_complete(is_complete, n_labs, p_labs)
}

show_run_times(FALSE)
```

**Blank cells have at least one failure**

| n   | method/p                       |      2|      3|       4|        5|         6|
|:----|:-------------------------------|------:|------:|-------:|--------:|---------:|
| 2   | GHQ                            |   0.02|   0.12|    1.06|     6.93|     69.27|
|     | AGHQ                           |   0.02|   0.09|    0.47|     3.00|     32.33|
|     | CDF                            |   0.03|   0.03|    0.00|     0.02|      0.02|
|     | Genz & Monahan (1999)          |       |  49.94|        |    74.98|          |
|     | Genz & Monahan (1999) Adaptive |  18.60|   9.77|    6.89|    23.51|          |
| 4   | GHQ                            |   0.07|   0.41|    3.95|    24.96|    206.69|
|     | AGHQ                           |   0.02|   0.14|    0.60|     3.59|     35.34|
|     | CDF                            |   0.42|   0.45|    0.42|     0.42|      0.46|
|     | Genz & Monahan (1999)          |       |  85.60|  108.23|    91.98|          |
|     | Genz & Monahan (1999) Adaptive |  14.73|  12.54|   26.44|     4.89|          |
| 8   | GHQ                            |       |   1.49|    8.44|   142.16|    680.75|
|     | AGHQ                           |   0.04|   0.19|    1.48|     7.91|     37.23|
|     | CDF                            |   4.75|   4.62|    4.53|     4.69|      4.56|
|     | Genz & Monahan (1999)          |       |       |        |         |          |
|     | Genz & Monahan (1999) Adaptive |   1.64|   4.07|        |     4.17|      4.13|
| 16  | GHQ                            |       |   4.84|   93.00|   542.17|   2610.50|
|     | AGHQ                           |   0.06|   0.27|    1.73|     8.93|     62.87|
|     | CDF                            |  31.90|  32.38|   32.23|    33.98|     33.66|
|     | Genz & Monahan (1999)          |       |       |        |         |          |
|     | Genz & Monahan (1999) Adaptive |   0.25|   1.59|    1.19|     1.29|      2.76|
| 32  | GHQ                            |       |       |  260.15|  4491.88|  33577.08|
|     | AGHQ                           |   0.16|   0.35|    1.66|     6.20|     59.00|
|     | CDF                            |  75.88|  74.31|   73.43|    76.19|     73.61|
|     | Genz & Monahan (1999)          |       |       |        |         |          |
|     | Genz & Monahan (1999) Adaptive |   4.88|   0.45|    0.84|     0.63|      1.06|

``` r
show_run_times(TRUE)
```

**Only showing complete cases**

| n   | method/p                       |       2|       3|       4|        5|         6|
|:----|:-------------------------------|-------:|-------:|-------:|--------:|---------:|
| 2   | GHQ                            |    0.02|    0.12|    0.79|     6.93|     40.46|
|     | AGHQ                           |    0.03|    0.09|    0.43|     3.00|     26.33|
|     | CDF                            |    0.04|    0.03|    0.00|     0.02|      0.02|
|     | Genz & Monahan (1999)          |   13.88|   49.94|   43.96|    74.98|     74.73|
|     | Genz & Monahan (1999) Adaptive |   13.68|    9.77|    6.18|    23.51|     22.44|
| 4   | GHQ                            |    0.07|    0.41|    3.95|    24.96|    153.06|
|     | AGHQ                           |    0.03|    0.14|    0.60|     3.59|     30.64|
|     | CDF                            |    0.43|    0.45|    0.42|     0.42|      0.47|
|     | Genz & Monahan (1999)          |   77.28|   85.60|  108.23|    91.98|    123.41|
|     | Genz & Monahan (1999) Adaptive |    5.30|   12.54|   26.44|     4.89|     14.21|
| 8   | GHQ                            |    0.10|    0.74|    7.41|    89.98|    331.24|
|     | AGHQ                           |    0.05|    0.17|    1.36|     7.80|     34.50|
|     | CDF                            |    4.74|    4.61|    4.55|     4.74|      4.59|
|     | Genz & Monahan (1999)          |  103.63|  238.29|  244.12|   215.58|    193.56|
|     | Genz & Monahan (1999) Adaptive |    1.91|    0.82|    1.20|     4.21|      3.34|
| 16  | GHQ                            |    0.28|    4.42|  110.59|   538.85|   2793.15|
|     | AGHQ                           |    0.06|    0.26|    1.74|     7.81|     60.53|
|     | CDF                            |   31.71|   32.35|   32.07|    33.58|     33.39|
|     | Genz & Monahan (1999)          |  141.72|  383.82|  439.39|   575.03|    561.41|
|     | Genz & Monahan (1999) Adaptive |    0.24|    0.69|    1.26|     0.71|      2.99|
| 32  | GHQ                            |    0.98|   19.06|  172.64|  5023.05|  26964.95|
|     | AGHQ                           |    0.14|    0.31|    0.84|     5.27|     47.69|
|     | CDF                            |   75.26|   75.18|   74.61|    75.18|     74.35|
|     | Genz & Monahan (1999)          |  465.74|  685.87|  702.84|   846.62|   1570.17|
|     | Genz & Monahan (1999) Adaptive |    0.42|    0.45|    0.38|     0.54|      0.81|

**Number of complete cases**

|     |    2|    3|    4|    5|    6|
|-----|----:|----:|----:|----:|----:|
| 2   |   19|   20|   16|   20|   18|
| 4   |   19|   20|   20|   20|   18|
| 8   |   17|   16|   18|   17|   18|
| 16  |   17|   17|   16|   15|   16|
| 32  |   17|   17|   14|   17|   15|

``` r

#####
# mean scaled RMSE table
show_scaled_mean_rmse <- function(remove_nas){
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
  
  err <- sapply(err, rowMeans) * err_mult
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
  
  cat(.get_cap(remove_nas))
  
  options(knitr.kable.NA = "")
  print(knitr::kable(
    table_out, align = c("l", "l", rep("r", length(p_vals)))))
  
  if(remove_nas)
    .show_n_complete(is_complete, n_labs, p_labs)
}

show_scaled_mean_rmse(FALSE)
```

**Blank cells have at least one failure**

| n   | method/p                       |       2|       3|       4|       5|       6|
|:----|:-------------------------------|-------:|-------:|-------:|-------:|-------:|
| 2   | GHQ                            |    9.19|    6.75|    9.42|    6.47|    6.01|
|     | AGHQ                           |    7.13|    5.67|    9.04|    5.55|    5.64|
|     | CDF                            |    7.26|    5.72|    8.92|    5.67|    5.62|
|     | Genz & Monahan (1999)          |        |   38.73|        |   58.60|        |
|     | Genz & Monahan (1999) Adaptive |   30.97|   47.24|   47.39|   64.27|        |
| 4   | GHQ                            |   15.74|   12.51|   17.72|    9.58|   12.73|
|     | AGHQ                           |    9.17|    9.58|   12.43|    8.67|   14.25|
|     | CDF                            |   19.35|   18.70|   22.91|   24.32|   26.99|
|     | Genz & Monahan (1999)          |        |  131.94|  139.43|  127.33|        |
|     | Genz & Monahan (1999) Adaptive |   55.25|  162.08|   89.73|  132.42|        |
| 8   | GHQ                            |        |   38.84|   39.84|   39.26|   21.65|
|     | AGHQ                           |   15.62|   17.50|   19.83|   23.77|   18.08|
|     | CDF                            |   21.87|   23.61|   25.28|   31.87|   22.29|
|     | Genz & Monahan (1999)          |        |        |        |        |        |
|     | Genz & Monahan (1999) Adaptive |  123.66|  168.21|        |  210.24|  208.82|
| 16  | GHQ                            |        |  167.28|  106.41|  127.49|   92.19|
|     | AGHQ                           |   23.23|   17.62|   23.33|   37.00|   61.23|
|     | CDF                            |   30.15|   27.96|   28.30|   46.09|   68.04|
|     | Genz & Monahan (1999)          |        |        |        |        |        |
|     | Genz & Monahan (1999) Adaptive |  127.51|  286.55|  312.72|  353.72|  378.96|
| 32  | GHQ                            |        |        |  282.43|  312.92|  266.08|
|     | AGHQ                           |   34.23|   49.35|   99.43|   57.01|   83.81|
|     | CDF                            |   76.19|   82.15|  125.72|   94.36|  125.81|
|     | Genz & Monahan (1999)          |        |        |        |        |        |
|     | Genz & Monahan (1999) Adaptive |  244.21|  274.19|  409.09|  387.13|  463.11|

``` r
show_scaled_mean_rmse(TRUE)
```

**Only showing complete cases**

| n   | method/p                       |       2|       3|        4|       5|       6|
|:----|:-------------------------------|-------:|-------:|--------:|-------:|-------:|
| 2   | GHQ                            |    8.06|    6.75|     8.26|    6.47|    5.67|
|     | AGHQ                           |    6.22|    5.67|     8.84|    5.55|    5.34|
|     | CDF                            |    6.15|    5.72|     8.88|    5.67|    5.36|
|     | Genz & Monahan (1999)          |   37.81|   38.73|    67.95|   58.60|   57.10|
|     | Genz & Monahan (1999) Adaptive |   29.09|   47.24|    46.43|   64.27|   46.52|
| 4   | GHQ                            |   16.32|   12.51|    17.72|    9.58|   12.83|
|     | AGHQ                           |    9.43|    9.58|    12.43|    8.67|   15.18|
|     | CDF                            |   19.69|   18.70|    22.91|   24.32|   26.48|
|     | Genz & Monahan (1999)          |  162.67|  131.94|   139.43|  127.33|  172.40|
|     | Genz & Monahan (1999) Adaptive |   56.07|  162.08|    89.73|  132.42|  138.80|
| 8   | GHQ                            |   56.11|   36.70|    42.10|   31.64|   21.65|
|     | AGHQ                           |   14.96|   19.47|    20.94|   17.23|   18.97|
|     | CDF                            |   17.83|   23.58|    25.84|   25.91|   22.78|
|     | Genz & Monahan (1999)          |  236.28|  185.91|   200.49|  272.13|  252.81|
|     | Genz & Monahan (1999) Adaptive |  119.25|  179.82|   175.57|  202.02|  212.55|
| 16  | GHQ                            |  134.64|  160.54|   126.85|   93.83|   85.55|
|     | AGHQ                           |   19.60|   16.63|    25.47|   34.65|   37.27|
|     | CDF                            |   25.99|   27.36|    29.86|   43.33|   44.78|
|     | Genz & Monahan (1999)          |  457.59|  486.99|   408.75|  569.23|  445.92|
|     | Genz & Monahan (1999) Adaptive |  127.07|  286.81|   298.32|  370.65|  360.66|
| 32  | GHQ                            |  388.58|  407.13|   272.85|  331.03|  237.07|
|     | AGHQ                           |   31.78|   47.17|    69.40|   52.50|   54.38|
|     | CDF                            |   76.19|   74.42|   105.98|   91.77|  101.08|
|     | Genz & Monahan (1999)          |  859.31|  903.12|  1052.99|  941.96|  840.60|
|     | Genz & Monahan (1999) Adaptive |  243.57|  238.78|   362.14|  381.14|  398.06|

**Number of complete cases**

|     |    2|    3|    4|    5|    6|
|-----|----:|----:|----:|----:|----:|
| 2   |   19|   20|   16|   20|   18|
| 4   |   19|   20|   20|   20|   18|
| 8   |   17|   16|   18|   17|   18|
| 16  |   17|   17|   16|   15|   16|
| 32  |   17|   17|   14|   17|   15|

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
  
  cat(.get_cap(TRUE, if(adaptive) " (Adaptive GHQ)" else "GHQ"))
  
  options(knitr.kable.NA = "")
  print(knitr::kable(
    table_out, align = c("l", "l", rep("r", length(p_vals)))))
  
  .show_n_complete(is_ok, n_labs, p_labs)
}

show_n_nodes(FALSE)
```

**Only showing complete casesGHQ**

| n   | quantile/p |      2|      3|      4|      5|      6|
|:----|:-----------|------:|------:|------:|------:|------:|
| 2   | 0%         |   5.00|   6.00|   6.00|   6.00|   5.00|
|     | 25%        |   5.75|   7.00|   7.00|   7.00|   7.00|
|     | 50%        |   7.00|   8.00|   7.50|   7.00|   7.00|
|     | 75%        |   9.00|   8.25|   9.00|   8.00|   8.00|
|     | 100%       |  13.00|  10.00|  12.00|  11.00|  11.00|
| 4   | 0%         |   6.00|   7.00|   6.00|   7.00|   5.00|
|     | 25%        |   7.75|   8.00|   8.00|   7.75|   7.00|
|     | 50%        |   9.00|   9.00|   9.00|   8.00|   8.00|
|     | 75%        |  11.25|  11.25|  12.00|   9.00|  10.00|
|     | 100%       |  27.00|  16.00|  13.00|  14.00|  12.00|
| 8   | 0%         |   6.00|   7.00|   7.00|   7.00|   7.00|
|     | 25%        |   7.50|   9.00|   8.00|   8.00|   8.00|
|     | 50%        |  12.00|  10.00|  10.00|   9.50|   9.00|
|     | 75%        |  14.50|  13.25|  12.00|  11.00|   9.00|
|     | 100%       |  20.00|  21.00|  13.00|  17.00|  14.00|
| 16  | 0%         |   6.00|   8.00|   8.00|   8.00|   8.00|
|     | 25%        |   9.00|  10.75|   9.75|  10.00|  10.00|
|     | 50%        |  12.00|  13.00|  11.00|  11.50|  11.00|
|     | 75%        |  16.50|  16.00|  12.25|  14.00|  11.00|
|     | 100%       |  22.00|  26.00|  30.00|  17.00|  14.00|
| 32  | 0%         |  11.00|   9.00|  11.00|   8.00|   8.00|
|     | 25%        |  14.25|  12.50|  12.00|  11.75|  10.75|
|     | 50%        |  17.50|  17.00|  14.00|  13.00|  12.50|
|     | 75%        |  19.75|  23.00|  17.00|  15.75|  14.00|
|     | 100%       |  28.00|  25.00|  30.00|  27.00|  21.00|

**Number of complete cases**

|     |    2|    3|    4|    5|    6|
|-----|----:|----:|----:|----:|----:|
| 2   |   20|   20|   20|   20|   20|
| 4   |   20|   20|   20|   20|   20|
| 8   |   19|   20|   20|   20|   20|
| 16  |   18|   20|   20|   20|   20|
| 32  |   18|   19|   20|   20|   20|

``` r
show_n_nodes(TRUE)
```

**Only showing complete cases (Adaptive GHQ)**

| n   | quantile/p |      2|     3|     4|     5|     6|
|:----|:-----------|------:|-----:|-----:|-----:|-----:|
| 2   | 0%         |   4.00|  4.00|  4.00|  4.00|  5.00|
|     | 25%        |   4.00|  6.00|  5.75|  6.00|  6.00|
|     | 50%        |   6.00|  6.50|  7.00|  6.00|  6.00|
|     | 75%        |   7.00|  7.00|  7.00|  7.00|  7.00|
|     | 100%       |   9.00|  8.00|  8.00|  8.00|  9.00|
| 4   | 0%         |   4.00|  6.00|  4.00|  5.00|  4.00|
|     | 25%        |   4.00|  6.00|  5.75|  6.00|  5.00|
|     | 50%        |   7.00|  7.00|  6.00|  6.00|  6.00|
|     | 75%        |   7.25|  7.00|  7.00|  6.00|  7.00|
|     | 100%       |  12.00|  9.00|  7.00|  7.00|  8.00|
| 8   | 0%         |   4.00|  4.00|  4.00|  4.00|  5.00|
|     | 25%        |   4.00|  5.75|  6.00|  5.00|  6.00|
|     | 50%        |   6.00|  6.00|  6.00|  6.00|  6.00|
|     | 75%        |   6.00|  6.00|  7.00|  7.00|  6.00|
|     | 100%       |  10.00|  8.00|  9.00|  8.00|  7.00|
| 16  | 0%         |   4.00|  4.00|  4.00|  4.00|  4.00|
|     | 25%        |   4.00|  4.00|  4.75|  5.00|  5.00|
|     | 50%        |   4.00|  5.50|  6.00|  6.00|  6.00|
|     | 75%        |   4.00|  6.00|  6.00|  6.00|  6.00|
|     | 100%       |   7.00|  7.00|  7.00|  6.00|  7.00|
| 32  | 0%         |   4.00|  4.00|  4.00|  4.00|  4.00|
|     | 25%        |   4.00|  4.00|  4.00|  4.00|  4.00|
|     | 50%        |   4.00|  4.00|  4.00|  4.00|  5.00|
|     | 75%        |   4.00|  4.00|  4.00|  5.00|  6.00|
|     | 100%       |   9.00|  6.00|  7.00|  6.00|  6.00|

**Number of complete cases**

|     |    2|    3|    4|    5|    6|
|-----|----:|----:|----:|----:|----:|
| 2   |   20|   20|   20|   20|   20|
| 4   |   20|   20|   20|   20|   20|
| 8   |   20|   20|   20|   20|   20|
| 16  |   20|   20|   20|   20|   20|
| 32  |   20|   20|   20|   20|   20|

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
