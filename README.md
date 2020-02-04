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
  get_cdf_cpp <- function(y, eta, Z, S, maxpts, abseps = 1e-5, 
                          releps = -1)
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
  get_sim_mth <- function(y, eta, Z, S, maxpts, abseps = 1e-5, releps = -1)
    # Args: 
    #   key: integer which determines degree of integration rule.
    function(key)
      mixprobit:::aprx_binary_mix(
        y = y, eta = eta, Z = Z, Sigma = S, maxpts = maxpts, key = key, 
        abseps = abseps, releps = releps)
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
GHQ_R   <- wd(aprx$get_GHQ_R  (y = y, eta = eta, Z = Z, S = S, b = b))
#> Loading required package: Rcpp
GHQ_cpp <- wd(aprx$get_GHQ_cpp(y = y, eta = eta, Z = Z, S = S, b = b))

cdf_aprx_R   <- wd(aprx$get_cdf_R  (y = y, eta = eta, Z = Z, S = S, 
                                    maxpts = maxpts))
cdf_aprx_cpp <- wd(aprx$get_cdf_cpp(y = y, eta = eta, Z = Z, S = S, 
                                    maxpts = maxpts))

sim_aprx <- wd(aprx$get_sim_mth(y = y, eta = eta, Z = Z, S = S, 
                                maxpts = maxpts))

#####
# compare results. Start with the simulation based methods with a lot of
# samples. We take this as the ground truth
truth_maybe1 <- wd(
  aprx$get_cdf_cpp(y = y, eta = eta, Z = Z, S = S, maxpts = 1e7, 
                   abseps = 1e-11))()
truth_maybe2 <- wd(
  aprx$get_sim_mth(y = y, eta = eta, Z = Z, S = S, maxpts = 1e7, 
                   abseps = 1e-11)(2L))
truth <- wd(
  mixprobit:::aprx_binary_mix_brute(y = y, eta = eta, Z = Z, Sigma = S, 
                                    n_sim = 1e8, n_threads = 6L))

all.equal(truth, c(truth_maybe1))
#> [1] "Mean relative difference: 0.0001078"
all.equal(truth, c(truth_maybe2))
#> [1] "Mean relative difference: 6.412e-05"

# compare with using fewer samples and GHQ
all.equal(truth,   GHQ_R())
#> [1] "Mean relative difference: 0.0006208"
all.equal(truth,   GHQ_cpp())
#> [1] "Mean relative difference: 0.0006208"
all.equal(truth, c(cdf_aprx_R()))
#> [1] "Mean relative difference: 0.0001428"
all.equal(truth, c(cdf_aprx_cpp()))
#> [1] "Mean relative difference: 0.0001866"
all.equal(truth, c(sim_aprx(1L)))
#> [1] "Mean relative difference: 0.001772"
all.equal(truth, c(sim_aprx(2L)))
#> [1] "Mean relative difference: 0.0005996"
all.equal(truth, c(sim_aprx(3L)))
#> [1] "Mean relative difference: 0.002807"
all.equal(truth, c(sim_aprx(4L)))
#> [1] "Mean relative difference: 0.01063"

# compare computations times
system.time(GHQ_R()) # way too slow (seconds!). Use C++ method instead
#>    user  system elapsed 
#>   21.17    0.00   21.17
microbenchmark::microbenchmark(
  `GHQ (C++)` = GHQ_cpp(),
  `CDF` = cdf_aprx_R(), `CDF (C++)` = cdf_aprx_cpp(),
  `Genz & Monahan (1)` = sim_aprx(1L), `Genz & Monahan (2)` = sim_aprx(2L),
  `Genz & Monahan (3)` = sim_aprx(3L), `Genz & Monahan (4)` = sim_aprx(4L),
  times = 10)
#> Unit: milliseconds
#>                expr    min     lq   mean median     uq    max neval
#>           GHQ (C++) 604.83 605.30 615.35 613.76 625.89 631.09    10
#>                 CDF  20.42  20.56  20.89  21.01  21.11  21.23    10
#>           CDF (C++)  11.55  11.65  12.05  11.87  12.10  13.36    10
#>  Genz & Monahan (1)  28.83  28.86  29.29  29.11  29.79  29.89    10
#>  Genz & Monahan (2)  29.60  29.66  30.61  30.55  31.45  31.60    10
#>  Genz & Monahan (3)  28.62  28.93  29.55  29.50  29.78  30.94    10
#>  Genz & Monahan (4)  28.31  29.18  30.37  29.48  29.85  38.62    10
```

More Rigorous Comparison
------------------------

We are interested in a more rigorous comparison. Therefor, we define a function below which for given number of observation in the cluster, `n`, and given number of random effects, `p`, performs a repeated number of runs with each of the methods and returns the computation time (among other output). To make a fair comparison, we fix the relative error of the methods before hand such that the relative error is below `releps`. Since GHQ is deterministic, we use a number of nodes such that this number of nodes or `streak_length` less (which is set to three at the time of this writing) value of nodes with GHQ gives a relative error which is below the threshold. We use a minimum of 10 nodes at the time of this writing.

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
sim_experiment <- function(n, p, releps = 5e-3, key_use = 2L){
  # in some cases we do not want to run anything
  do_not_run <- p >= 5L && n >= 10L
  
  # simulate data
  dat <- get_sim_dat(n = n, p = p)
  
  # shorter than calling `with(dat, ...)`
  wd <- function(expr)
    eval(bquote(with(dat, .(substitute(expr)))), parent.frame())
  
  # get the assumed ground truth
  truth <- if(do_not_run)
    NA
  else wd(mixprobit:::aprx_binary_mix_brute(
    y = y, eta = eta, Z = Z, Sigma = S, n_sim = 1e7))
  
  # function to test whether the value is ok
  is_ok_func <- function(vals)
    abs((log(vals) - log(truth)) / log(truth)) < releps
  
  # get function to use with GHQ
  b_use <- if(do_not_run)
    NA_integer_
  else local({
    apx_func <- function(b)
      wd(aprx$get_GHQ_cpp(y = y, eta = eta, Z = Z, S = S, b = b))()
    
    # length of node values which have a relative error below the threshold
    streak_length <- 3L
    vals <- rep(NA_real_, streak_length)
    
    b <- 10L
    for(i in 1:(streak_length - 1L))
      vals[i + 1L] <- apx_func(b - streak_length + i)
    repeat({
      vals[1:(streak_length - 1L)] <- vals[-1]
      vals[streak_length] <- apx_func(b)
      
      if(all(is_ok_func(vals)))
        break
      
      b <- b + 1L
      if(b > 50L){
        warning("found no node value")
        b <- NA_integer_
        break
      }
    })
    b
  })
  
  ghq_func <- if(!is.na(b_use))
    wd(aprx$get_GHQ_cpp(y = y, eta = eta, Z = Z, S = S, b = b_use))
  else
    NA
  
  # get function to use with CDF method
  cdf_maxpts_use <- if(do_not_run)
    NA_integer_
  else local({
    maxpts <- 100L
    repeat {
      func <- wd(aprx$get_cdf_cpp(y = y, eta = eta, Z = Z, S = S, 
                                  maxpts = maxpts, abseps = -1, 
                                  releps = releps / 10))
      vals <- replicate(10, func())
      if(all(is_ok_func(vals)))
        break
      
      maxpts <- maxpts * 2L
      if(maxpts > 1000000L){
        warning("found no maxpts for CDF method")
        maxpts <- NA_integer_
        break
      }
    }
    maxpts
  })
  
  cdf_func <- if(!is.na(cdf_maxpts_use))
    wd(aprx$get_cdf_cpp(y = y, eta = eta, Z = Z, S = S, 
                        maxpts = cdf_maxpts_use, abseps = -1, 
                        releps = releps / 10))
  else 
    NA
  
  # get function to use with Genz and Monahan method
  sim_maxpts_use <- if(do_not_run)
    NA_integer_
  else local({
    maxpts <- 100L
    repeat {
      func <- wd(aprx$get_sim_mth(y = y, eta = eta, Z = Z, S = S, 
                                  maxpts = maxpts, abseps = -1, 
                                  releps = releps / 10))
      vals <- replicate(10, func(key_use))
      if(all(is_ok_func(vals)))
        break
      
      maxpts <- maxpts * 2L
      if(maxpts > 1000000L){
        warning("found no maxpts for sim method")
        maxpts <- NA_integer_
        break
      }
    }
    maxpts
  })
  
  sim_func <- if(!is.na(sim_maxpts_use))
    wd(aprx$get_sim_mth(y = y, eta = eta, Z = Z, S = S, 
                        maxpts = sim_maxpts_use, abseps = -1, 
                        releps = releps / 10))
  else 
    NA
  
  if(is.function(sim_func))
    formals(sim_func)$key <- key_use
  
  # perform the comparison
  out <- sapply(
    list(GHQ = ghq_func, CDF = cdf_func, GenzMonahan = sim_func), 
    function(func){
      if(!is.function(func) && is.na(func)){
        out <- rep(NA_real_, 6L)
        names(out) <- c("mean", "sd", "mse", "user.self", 
                        "sys.self", "elapsed")
        return(out)
      }
      
      # number of runs used to estimate the computation time, etc.
      n_runs <- 20L
      
      # perform the computations
      ti <- system.time(vals <- replicate(n_runs, func()))
      
      c(mean = mean(vals), sd = sd(vals), mse = mean((vals - truth)^2), 
        ti[1:3] / n_runs)            
    })
  
  list(b_use = b_use, cdf_maxpts_use = cdf_maxpts_use, 
       sim_maxpts_use = sim_maxpts_use, vals_n_comp_time = out)
}
```

Here is a few quick examples where we use the function we just defined.

``` r
set.seed(1)
sim_experiment(n = 3L , p = 2L)
#> $b_use
#> [1] 14
#> 
#> $cdf_maxpts_use
#> [1] 100
#> 
#> $sim_maxpts_use
#> [1] 3200
#> 
#> $vals_n_comp_time
#>                 GHQ       CDF GenzMonahan
#> mean      1.037e-01 1.033e-01   1.031e-01
#> sd        0.000e+00 4.327e-05   8.469e-04
#> mse       1.117e-07 3.675e-09   7.575e-07
#> user.self 1.000e-04 2.000e-04   9.500e-04
#> sys.self  0.000e+00 0.000e+00   0.000e+00
#> elapsed   5.000e-05 2.000e-04   9.500e-04
sim_experiment(n = 10L, p = 2L)
#> $b_use
#> [1] 10
#> 
#> $cdf_maxpts_use
#> [1] 100
#> 
#> $sim_maxpts_use
#> [1] 800
#> 
#> $vals_n_comp_time
#>                 GHQ       CDF GenzMonahan
#> mean      4.839e-05 4.836e-05   4.855e-05
#> sd        0.000e+00 3.420e-09   8.255e-07
#> mse       1.241e-15 1.360e-16   6.858e-13
#> user.self 5.000e-05 1.135e-02   6.000e-04
#> sys.self  0.000e+00 0.000e+00   0.000e+00
#> elapsed   5.000e-05 1.135e-02   6.000e-04
sim_experiment(n = 3L , p = 4L)
#> $b_use
#> [1] 10
#> 
#> $cdf_maxpts_use
#> [1] 100
#> 
#> $sim_maxpts_use
#> [1] 400
#> 
#> $vals_n_comp_time
#>                 GHQ       CDF GenzMonahan
#> mean      2.870e-02 2.870e-02   2.877e-02
#> sd        0.000e+00 6.402e-06   1.781e-04
#> mse       5.949e-12 6.064e-11   3.486e-08
#> user.self 2.400e-03 1.500e-04   1.000e-04
#> sys.self  0.000e+00 0.000e+00   0.000e+00
#> elapsed   2.450e-03 2.000e-04   1.000e-04
sim_experiment(n = 10L, p = 4L)
#> $b_use
#> [1] 25
#> 
#> $cdf_maxpts_use
#> [1] 100
#> 
#> $sim_maxpts_use
#> [1] 6400
#> 
#> $vals_n_comp_time
#>                 GHQ       CDF GenzMonahan
#> mean      3.504e-04 3.418e-04   3.429e-04
#> sd        0.000e+00 5.080e-08   9.691e-06
#> mse       7.276e-11 2.566e-15   9.027e-11
#> user.self 3.177e-01 1.175e-02   5.050e-03
#> sys.self  0.000e+00 0.000e+00   0.000e+00
#> elapsed   3.177e-01 1.175e-02   5.050e-03
```

Next, we apply the method a number of times for a fixed of combination of number of observations, `n`, and number of random effects, `p`.

``` r
# number of observations in the cluster
n_vals <- c(2:5, 7L, 10L, 15L, 20L)
# number of random effects
p_vals <- 2:5
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
  clusterExport(cl, c("aprx", "get_sim_dat", "sim_experiment"))
  
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

We create a table where we summarize the results below. First we start with the average computation time, then we show the mean MSE, and we end by looking at the number of nodes that we need to use with GHQ. The latter shows why GHQ becomes slower as the cluster size, `n`, increases.

``` r
#####
# table with computation times
comp_time_mult <- 1000 # millisecond

local({
  # get mean computations time for the methods and the configurations pairs
  comp_times <- sapply(ex_output, function(x)
    sapply(x[!names(x) %in% c("n", "p")], `[[`, "vals_n_comp_time", 
           simplify = "array"), 
    simplify = "array")
  comp_times <- comp_times["elapsed", , , ]
  comp_times <- apply(comp_times, c(1, 3), mean) * comp_time_mult
  
  # flatten the table. Start by getting the row labels
  meths <- rownames(comp_times)
  rnames <- expand.grid(
    Method = meths, n = sprintf("%2d", n_vals), stringsAsFactors = FALSE)
  rnames[2:1] <- rnames[1:2]
  nvs <- rnames[[1L]]
  rnames[[1L]] <- c(
    nvs[1L], ifelse(nvs[-1L] != head(nvs, -1L), nvs[-1L], NA_integer_))
  rnames[[2L]] <- gsub(
    "^GenzMonahan$", "Genz & Monahan (1999)", rnames[[2L]])
  
  # then flatten
  comp_times <- matrix(c(comp_times), nrow = NROW(rnames))
  na_idx <- is.na(comp_times)
  comp_times[] <- sprintf("%.2f", comp_times[])
  comp_times[na_idx] <- NA_character_
  
  # combine computation times and row labels
  table_out <- cbind(as.matrix(rnames), comp_times)
  
  # add header 
  colnames(table_out) <- c("n", "method/p", sprintf("%d", p_vals))
  
  options(knitr.kable.NA = "")
  knitr::kable(table_out, align = c("l", "l", rep("r", length(p_vals))))
})
```

| n   | method/p              |      2|      3|       4|       5|
|:----|:----------------------|------:|------:|-------:|-------:|
| 2   | GHQ                   |   0.03|   0.36|    2.52|   20.05|
|     | CDF                   |   0.01|   0.02|    0.02|    0.02|
|     | Genz & Monahan (1999) |   3.57|   6.96|   11.36|   11.77|
| 3   | GHQ                   |   0.05|   0.38|    4.18|   53.78|
|     | CDF                   |   0.17|   0.19|    0.19|    0.18|
|     | Genz & Monahan (1999) |   2.22|  14.46|    8.83|   20.97|
| 4   | GHQ                   |   0.05|   0.55|    6.68|   81.73|
|     | CDF                   |   0.37|   0.40|    0.42|    0.38|
|     | Genz & Monahan (1999) |   8.10|  10.71|   15.74|   14.26|
| 5   | GHQ                   |   0.06|   1.19|   15.67|   92.01|
|     | CDF                   |   0.75|   0.75|    0.79|    0.77|
|     | Genz & Monahan (1999) |   4.49|  17.73|   23.24|   37.94|
| 7   | GHQ                   |   0.10|   1.37|   31.76|  339.64|
|     | CDF                   |   2.46|   2.51|    2.60|    2.46|
|     | Genz & Monahan (1999) |  13.10|  20.75|   18.46|   53.02|
| 10  | GHQ                   |   0.18|   2.33|   29.55|        |
|     | CDF                   |  12.23|  12.81|   12.67|        |
|     | Genz & Monahan (1999) |  18.80|  24.15|   24.10|        |
| 15  | GHQ                   |   0.28|       |  154.91|        |
|     | CDF                   |  28.41|  29.38|   28.81|        |
|     | Genz & Monahan (1999) |  14.65|  31.97|   30.80|        |
| 20  | GHQ                   |   0.36|   9.73|  215.64|        |
|     | CDF                   |  38.56|  40.67|   38.16|        |
|     | Genz & Monahan (1999) |  16.96|  35.80|   50.72|        |

``` r

#####
# mean MSE table
err_mult <- 1e6
local({
  # get mean mse for the methods and the configurations pairs
  res <- sapply(ex_output, function(x)
    sapply(x[!names(x) %in% c("n", "p")], `[[`, "vals_n_comp_time", 
           simplify = "array"), 
    simplify = "array")
  err <- res["mse", , , ]
  err <- apply(err, c(1, 3), mean) * err_mult
  
  # flatten the table. Start by getting the row labels
  meths <- rownames(err)
  rnames <- expand.grid(
    Method = meths, n = sprintf("%2d", n_vals), stringsAsFactors = FALSE)
  rnames[2:1] <- rnames[1:2]
  nvs <- rnames[[1L]]
  rnames[[1L]] <- c(
    nvs[1L], ifelse(nvs[-1L] != head(nvs, -1L), nvs[-1L], NA_integer_))
  rnames[[2L]] <- gsub(
    "^GenzMonahan$", "Genz & Monahan (1999)", rnames[[2L]])
  
  # then flatten
  err <- matrix(c(err), nrow = NROW(rnames))
  na_idx <- is.na(err)
  err[] <- sprintf("%.2f", err[])
  err[na_idx] <- NA_character_
  
  # combine mean mse and row labels
  table_out <- cbind(as.matrix(rnames), err)
  
  # add header 
  colnames(table_out) <- c("n", "method/p", sprintf("%d", p_vals))
  
  options(knitr.kable.NA = "")
  knitr::kable(table_out, align = c("l", "l", rep("r", length(p_vals))))
})
```

| n   | method/p              |     2|     3|     4|     5|
|:----|:----------------------|-----:|-----:|-----:|-----:|
| 2   | GHQ                   |  0.06|  0.10|  0.06|  0.03|
|     | CDF                   |  0.01|  0.00|  0.01|  0.01|
|     | Genz & Monahan (1999) |  0.79|  0.80|  0.90|  0.79|
| 3   | GHQ                   |  0.10|  0.08|  0.05|  0.10|
|     | CDF                   |  0.00|  0.00|  0.00|  0.00|
|     | Genz & Monahan (1999) |  0.46|  0.66|  0.33|  0.44|
| 4   | GHQ                   |  0.16|  0.09|  0.14|  0.08|
|     | CDF                   |  0.00|  0.00|  0.00|  0.01|
|     | Genz & Monahan (1999) |  0.26|  0.35|  0.47|  0.35|
| 5   | GHQ                   |  0.08|  0.13|  0.23|  0.07|
|     | CDF                   |  0.00|  0.00|  0.01|  0.00|
|     | Genz & Monahan (1999) |  0.24|  0.30|  0.30|  0.19|
| 7   | GHQ                   |  0.01|  0.09|  0.08|  0.05|
|     | CDF                   |  0.00|  0.00|  0.00|  0.00|
|     | Genz & Monahan (1999) |  0.04|  0.10|  0.15|  0.08|
| 10  | GHQ                   |  0.02|  0.02|  0.01|      |
|     | CDF                   |  0.00|  0.00|  0.00|      |
|     | Genz & Monahan (1999) |  0.05|  0.02|  0.02|      |
| 15  | GHQ                   |  0.00|      |  0.00|      |
|     | CDF                   |  0.00|  0.00|  0.00|      |
|     | Genz & Monahan (1999) |  0.01|  0.01|  0.03|      |
| 20  | GHQ                   |  0.01|  0.00|  0.00|      |
|     | CDF                   |  0.00|  0.00|  0.00|      |
|     | Genz & Monahan (1999) |  0.00|  0.00|  0.02|      |

``` r

#####
# GHQ node table
local({
  # get the number of nodes that we use
  res <- sapply(ex_output, function(x)
    sapply(x[!names(x) %in% c("n", "p")], `[[`, "b_use"))
  
  # compute the quantiles
  probs <- seq(0, 1, length.out = 5)
  qs <- matrix(NA_real_, nr = length(probs), nc = NCOL(res))
  is_ok <- apply(!is.na(res), 2L, all)
  qs_ok <- apply(res[, is_ok, drop = FALSE], 2L, quantile, prob = probs)
  
  qs[, is_ok] <- qs_ok
  rownames(qs) <- rownames(qs_ok)
  
  # flatten the table. Start by getting the row labels
  meths <- rownames(qs)
  rnames <- expand.grid(
    Method = meths, n = sprintf("%2d", n_vals), stringsAsFactors = FALSE)
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
  colnames(table_out) <- c("n", "quantile/p", sprintf("%d", p_vals))
  
  options(knitr.kable.NA = "")
  knitr::kable(table_out, align = c("l", "l", rep("r", length(p_vals))))
})
```

| n   | quantile/p |      2|      3|      4|      5|
|:----|:-----------|------:|------:|------:|------:|
| 2   | 0%         |  10.00|  10.00|  10.00|  10.00|
|     | 25%        |  10.00|  10.00|  10.00|  10.00|
|     | 50%        |  10.00|  10.00|  10.00|  10.00|
|     | 75%        |  10.00|  10.00|  10.00|  10.00|
|     | 100%       |  12.00|  19.00|  15.00|  10.00|
| 3   | 0%         |  10.00|  10.00|  10.00|  10.00|
|     | 25%        |  10.00|  10.00|  10.00|  10.00|
|     | 50%        |  10.00|  10.00|  10.00|  10.00|
|     | 75%        |  10.00|  11.00|  10.00|  11.00|
|     | 100%       |  13.00|  16.00|  16.00|  16.00|
| 4   | 0%         |  10.00|  10.00|  10.00|  10.00|
|     | 25%        |  10.00|  10.00|  10.00|  10.00|
|     | 50%        |  10.00|  10.00|  10.00|  10.00|
|     | 75%        |  10.00|  10.25|  11.00|  12.00|
|     | 100%       |  15.00|  18.00|  18.00|  15.00|
| 5   | 0%         |  10.00|  10.00|  10.00|  10.00|
|     | 25%        |  10.00|  10.00|  10.00|  10.00|
|     | 50%        |  10.00|  10.00|  10.00|  10.00|
|     | 75%        |  11.00|  15.75|  11.50|  12.00|
|     | 100%       |  17.00|  20.00|  22.00|  15.00|
| 7   | 0%         |  10.00|  10.00|  10.00|  10.00|
|     | 25%        |  10.00|  10.00|  10.00|  10.00|
|     | 50%        |  10.00|  11.50|  11.50|  11.00|
|     | 75%        |  12.25|  14.00|  16.00|  13.00|
|     | 100%       |  19.00|  18.00|  22.00|  20.00|
| 10  | 0%         |  10.00|  10.00|  10.00|       |
|     | 25%        |  10.00|  10.00|  10.00|       |
|     | 50%        |  10.00|  12.00|  11.50|       |
|     | 75%        |  14.75|  14.50|  14.25|       |
|     | 100%       |  27.00|  21.00|  19.00|       |
| 15  | 0%         |  10.00|       |  10.00|       |
|     | 25%        |  10.00|       |  10.00|       |
|     | 50%        |  11.00|       |  11.50|       |
|     | 75%        |  16.25|       |  13.25|       |
|     | 100%       |  29.00|       |  34.00|       |
| 20  | 0%         |  10.00|  10.00|  10.00|       |
|     | 25%        |  10.00|  12.75|  10.75|       |
|     | 50%        |  11.50|  15.00|  14.50|       |
|     | 75%        |  17.00|  19.50|  17.50|       |
|     | 100%       |  26.00|  28.00|  31.00|       |

The computation time is in 1000s of a second. The mean MSE is multiplied by ![10^{6}](https://latex.codecogs.com/svg.latex?10%5E%7B6%7D "10^{6}").

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
