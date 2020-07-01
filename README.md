Mixed Models with a Probit Link
===============================

We make a comparison below of making an approximation of a marginal
likelihood factor that is typical in many mixed effect models with a
probit link function. The particular model we use here is mixed probit
model where the observed outcomes are binary. In this model, a marginal
factor, ![L](https://latex.codecogs.com/svg.latex?L "L"), for a given
cluster is

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

where
![\\eta\_i](https://latex.codecogs.com/svg.latex?%5Ceta_i "\eta_i") can
be a fixed effect like
![\\vec x\_i^\\top\\vec\\beta](https://latex.codecogs.com/svg.latex?%5Cvec%20x_i%5E%5Ctop%5Cvec%5Cbeta "\vec x_i^\top\vec\beta")
for some fixed effect covariate
![\\vec x\_i](https://latex.codecogs.com/svg.latex?%5Cvec%20x_i "\vec x_i")
and fixed effect coefficients
![\\vec\\beta](https://latex.codecogs.com/svg.latex?%5Cvec%5Cbeta "\vec\beta")
and ![\\vec u](https://latex.codecogs.com/svg.latex?%5Cvec%20u "\vec u")
is an unobserved random effect for the cluster.

The [quick comparison](#quick-comparison) section may be skipped unless
you want to get a grasp at what is implemented and see the definitions
of the functions that is used in this markdown. The [more rigorous
comparison](#more-rigorous-comparison) section is the main section of
this markdown. It contains an example where we vary the number of
observed outcomes, `n`, and the number of random effect, `p`, while
considering the computation time of various approximation methods for a
fixed relative error. A real data application is provided in
[examples/salamander.md](examples/salamander.md).

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
  
  #####
  # returns a function that uses Quasi-monte carlo integration to 
  # approximate the integrals. 
  # 
  # Args:
  #   y: n length logical vector with for whether the observation has an 
  #      event.
  #   eta: n length numeric vector with offset on z-scale.
  #   Z: p by n matrix with random effect covariates. 
  #   S: n by n matrix with random effect covaraites.
  #   maxpts: integer with maximum number of points to use. 
  #   is_adaptive: logical for whether to use an adaptive method.
  #   releps: relative error tolerance.
  #   n_seqs: number of randomized sobol sequences.
  #   abseps: unused.
  get_qmc <- function(y, eta, Z, S, maxpts, is_adaptive = FALSE, 
                      releps = 1e-4, n_seqs = 15L, abseps)
    function(){
      seeds <- sample.int(2147483646L, n_seqs)
      mixprobit:::aprx_binary_mix_qmc(
        y = y, eta = eta, Z = Z, Sigma = S, n_max = maxpts, 
        is_adaptive = is_adaptive, seeds = seeds, releps = releps)
    }
  get_Aqmc <- get_qmc
  formals(get_Aqmc)$is_adaptive <- TRUE
})
```

Then we assign a function to get a simulated data set for a single
cluster within a mixed probit model with binary outcomes.

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

The variance of the linear predictor given the random effect is
independent of the random effect dimension, `p`.

``` r
var(replicate(1000, with(get_sim_dat(10, 2), u %*% Z + eta)))
#> [1] 1.971
var(replicate(1000, with(get_sim_dat(10, 3), u %*% Z + eta)))
#> [1] 1.977
var(replicate(1000, with(get_sim_dat(10, 4), u %*% Z + eta)))
#> [1] 1.953
var(replicate(1000, with(get_sim_dat(10, 5), u %*% Z + eta)))
#> [1] 1.989
var(replicate(1000, with(get_sim_dat(10, 6), u %*% Z + eta)))
#> [1] 2.014
var(replicate(1000, with(get_sim_dat(10, 7), u %*% Z + eta)))
#> [1] 1.986
var(replicate(1000, with(get_sim_dat(10, 8), u %*% Z + eta)))
#> [1] 2.03
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

qmc_aprx <- wd(
  aprx$get_qmc(y = y, eta = eta, Z = Z, S = S, maxpts = maxpts))
qmc_Aaprx <- wd(
  aprx$get_Aqmc(y = y, eta = eta, Z = Z, S = S, maxpts = maxpts))

sim_aprx <-  wd(aprx$get_sim_mth(y = y, eta = eta, Z = Z, S = S, 
                                 maxpts = maxpts))
sim_Aaprx <- wd(aprx$get_Asim_mth(y = y, eta = eta, Z = Z, S = S, 
                                  maxpts = maxpts))

#####
# compare results. Start with the simulation based methods with a lot of
# samples. We take this as the ground truth
truth_maybe_cdf <- wd( 
  aprx$get_cdf_cpp (y = y, eta = eta, Z = Z, S = S, maxpts = 1e7, 
                    abseps = 1e-11))()

truth_maybe_qmc <- wd(
  aprx$get_qmc(y = y, eta = eta, Z = Z, S = S, maxpts = 1e7, 
               releps = 1e-11)())

truth_maybe_Aqmc <- wd(
  aprx$get_Aqmc(y = y, eta = eta, Z = Z, S = S, maxpts = 1e7, 
                releps = 1e-11)())

truth_maybe_mc <- wd(
  aprx$get_sim_mth (y = y, eta = eta, Z = Z, S = S, maxpts = 1e7, 
                    abseps = 1e-11)(2L))
truth_maybe_Amc <- wd(
  aprx$get_Asim_mth(y = y, eta = eta, Z = Z, S = S, maxpts = 1e7, 
                    abseps = 1e-11)(2L))

truth <- wd(
  mixprobit:::aprx_binary_mix_brute(y = y, eta = eta, Z = Z, Sigma = S, 
                                    n_sim = 1e8, n_threads = 6L))

c(Estiamte = truth, SE = attr(truth, "SE"),  
  `Estimate (log)` = log(c(truth)),  
  `SE (log)` = abs(attr(truth, "SE") / truth))
#>       Estiamte             SE Estimate (log)       SE (log) 
#>      4.436e-03      3.214e-08     -5.418e+00      7.246e-06
truth <- c(truth)
all.equal(truth, c(truth_maybe_cdf))
#> [1] "Mean relative difference: 8.141e-05"
all.equal(truth, c(truth_maybe_qmc))
#> [1] "Mean relative difference: 5.678e-06"
all.equal(truth, c(truth_maybe_Aqmc))
#> [1] "Mean relative difference: 9.317e-06"
all.equal(truth, c(truth_maybe_mc))
#> [1] "Mean relative difference: 0.0005177"
all.equal(truth, c(truth_maybe_Amc))
#> [1] "Mean relative difference: 3.532e-05"

# compare with using fewer samples and GHQ
all.equal(truth,   GHQ_R())
#> [1] "Mean relative difference: 3.46e-06"
all.equal(truth,   GHQ_cpp())
#> [1] "Mean relative difference: 3.46e-06"
all.equal(truth,   AGHQ_cpp())
#> [1] "Mean relative difference: 4.897e-06"
all.equal(truth, c(cdf_aprx_R()))
#> [1] "Mean relative difference: 3.656e-05"
all.equal(truth, c(qmc_aprx()))
#> [1] "Mean relative difference: 0.004652"
all.equal(truth, c(qmc_Aaprx()))
#> [1] "Mean relative difference: 0.000562"
all.equal(truth, c(cdf_aprx_cpp()))
#> [1] "Mean relative difference: 4.006e-06"
all.equal(truth, c(sim_aprx(1L)))
#> [1] "Mean relative difference: 0.01306"
all.equal(truth, c(sim_aprx(2L)))
#> [1] "Mean relative difference: 3.302e-05"
all.equal(truth, c(sim_aprx(3L)))
#> [1] "Mean relative difference: 0.003255"
all.equal(truth, c(sim_aprx(4L)))
#> [1] "Mean relative difference: 0.0006256"
all.equal(truth, c(sim_Aaprx(1L)))
#> [1] "Mean relative difference: 0.0006038"
all.equal(truth, c(sim_Aaprx(2L)))
#> [1] "Mean relative difference: 0.001045"
all.equal(truth, c(sim_Aaprx(3L)))
#> [1] "Mean relative difference: 0.0005731"
all.equal(truth, c(sim_Aaprx(4L)))
#> [1] "Mean relative difference: 0.0001428"

# compare computations times
system.time(GHQ_R()) # way too slow (seconds!). Use C++ method instead
#>    user  system elapsed 
#>   1.483   0.000   1.482
microbenchmark::microbenchmark(
  `GHQ (C++)` = GHQ_cpp(), `AGHQ (C++)` = AGHQ_cpp(),
  `CDF` = cdf_aprx_R(), `CDF (C++)` = cdf_aprx_cpp(),
  QMC = qmc_aprx(), `QMC Adaptive` = qmc_Aaprx(),
  `Genz & Monahan (1)` = sim_aprx(1L), `Genz & Monahan (2)` = sim_aprx(2L),
  `Genz & Monahan (3)` = sim_aprx(3L), `Genz & Monahan (4)` = sim_aprx(4L),
  `Genz & Monahan Adaptive (2)` = sim_Aaprx(2L),
  times = 10)
#> Unit: milliseconds
#>                         expr    min     lq   mean median    uq   max neval
#>                    GHQ (C++) 39.703 39.926 40.672 40.651 41.09 42.51    10
#>                   AGHQ (C++) 43.028 43.587 44.519 44.112 45.41 47.68    10
#>                          CDF 21.198 21.304 21.762 21.702 22.00 22.70    10
#>                    CDF (C++) 11.735 12.148 12.506 12.366 12.91 13.51    10
#>                          QMC 35.477 35.709 37.076 36.350 38.56 40.16    10
#>                 QMC Adaptive 40.521 41.070 42.620 42.511 43.98 45.77    10
#>           Genz & Monahan (1) 30.806 31.566 32.326 32.489 33.12 33.63    10
#>           Genz & Monahan (2) 31.824 32.098 33.698 33.604 33.86 36.74    10
#>           Genz & Monahan (3) 31.201 31.858 33.397 33.226 35.03 36.06    10
#>           Genz & Monahan (4) 30.265 31.103 32.154 32.267 32.43 34.33    10
#>  Genz & Monahan Adaptive (2)  6.861  7.415  8.664  7.836  9.39 12.07    10
```

More Rigorous Comparison
------------------------

We are interested in a more rigorous comparison. Therefor, we define a
function below which for given number of observation in the cluster,
`n`, and given number of random effects, `p`, performs a repeated number
of runs with each of the methods and returns the computation time (among
other output). To make a fair comparison, we fix the relative error of
the methods before hand such that the relative error is below `releps`,
![10^{-4}](https://latex.codecogs.com/svg.latex?10%5E%7B-4%7D "10^{-4}").
Ground truth is computed with brute force MC using `n_brute`,
![10^{7}](https://latex.codecogs.com/svg.latex?10%5E%7B7%7D "10^{7}"),
samples.

Since GHQ is deterministic, we use a number of nodes such that this
number of nodes or `streak_length`, 4, less value of nodes with GHQ
gives a relative error which is below the threshold. We use a minimum of
4 nodes at the time of this writing. The error of the simulation based
methods is approximated using `n_reps`, 25, replications.

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
  is_ok_func <- function(vals){
    test_val <- abs((log(vals) - log(truth)) / log(truth)) 
    if(!all(is.finite(test_val)))
      browser()
    test_val < releps
  }
      
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
  
  is_to_large_for_ghq <- n > 16L || p >= 5L
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
            vals["inivls", ] / ex_params$max_maxpts < .999999)
        
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
  # sim_releps <- if(is_to_large_for_ghq) 
  #   NA_integer_ else get_releps(aprx$get_sim_mth)
  sim_releps <- NA_integer_ # just do not use it. It is __very__ slow in  
                            # some cases
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
  
  # get function to use with QMC
  qmc_releps <- if(is_to_large_for_ghq)
    NA_integer_ else get_releps(aprx$get_qmc)
  qmc_func <- if(!is.na(qmc_releps))
     wd(aprx$get_qmc(y = y, eta = eta, Z = Z, S = S, 
                     maxpts = ex_params$max_maxpts, abseps = -1,
                     releps = qmc_releps))
  else 
    NA

  # get function to use with adaptive QMC
  Aqmc_releps <- get_releps(aprx$get_Aqmc)
  Aqmc_func <- if(!is.null(Aqmc_releps))
    wd(aprx$get_Aqmc(y = y, eta = eta, Z = Z, S = S, 
                     maxpts = ex_params$max_maxpts, abseps = -1,
                     releps = Aqmc_releps))
  else 
    NA
    
  # perform the comparison
  out <- sapply(
    list(GHQ = ghq_func, AGHQ = aghq_func, CDF = cdf_func, 
         GenzMonahan = sim_func, GenzMonahanA = Asim_func, 
         QMC = qmc_func, QMCA = Aqmc_func), 
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
    qmc_releps = qmc_releps, Aqmc_releps = Aqmc_releps,
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
    sprintf("                    QMC releps: %13.8f", x$qmc_releps),
    sprintf("           Adaptive QMC releps: %13.8f", x$Aqmc_releps), 
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
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.00007813
#>                     QMC releps:    0.00007813
#>            Adaptive QMC releps:    0.00003906
#>   Log-likelihood estiamte (SE):   -0.84934116 (0.00000362)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                            0.427693                      -0.849350
#> AGHQ                           0.427693                      -0.849351
#> CDF                            0.427697                      -0.849339
#> GenzMonahan                          NA                             NA
#> GenzMonahanA                   0.427685                      -0.849369
#> QMC                            0.427700                      -0.849334
#> QMCA                           0.427690                      -0.849356
#> 
#> SD & RMSE (/10000.00)
#>            GHQ      AGHQ       CDF GenzMonahan GenzMonahanA       QMC      QMCA
#> SD   0.0866955 0.0192985 0.1465686          NA     0.129611 0.1003660 0.0618162
#> RMSE 0.1035563 0.0454279 0.0564467          NA     0.149582 0.0756879 0.1175650
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>       0.0000       0.0002       0.0004           NA       0.0030       0.0158 
#>         QMCA 
#>       0.0350
sim_experiment(n = 10L, p = 2L, n_threads = 6L)
#> Warning in (function() {: found no node value
#>          # brute force samples:      10000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             6
#>                     CDF releps:    0.01000000
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.01000000
#>                     QMC releps:    0.00062500
#>            Adaptive QMC releps:    0.01000000
#>   Log-likelihood estiamte (SE):   -9.31677950 (0.00000130)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                       0.0000899031                       -9.31678
#> CDF                        0.0000899088                       -9.31671
#> GenzMonahan                          NA                             NA
#> GenzMonahanA               0.0000899034                       -9.31677
#> QMC                        0.0000899034                       -9.31677
#> QMCA                       0.0000899100                       -9.31670
#> 
#> SD & RMSE (/10000.00)
#>      GHQ         AGHQ         CDF GenzMonahan  GenzMonahanA         QMC
#> SD    NA 0.0000111518 0.000166587          NA 0.00000497372 0.000118604
#> RMSE  NA 0.0000113043 0.000161444          NA 0.00000515647 0.000116758
#>             QMCA
#> SD   0.000341325
#> RMSE 0.000299828
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>           NA       0.0000       0.0118           NA       0.0008       0.0102 
#>         QMCA 
#>       0.0014

sim_experiment(n = 3L , p = 5L, n_threads = 6L)
#>          # brute force samples:      10000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             6
#>                     CDF releps:    0.00500000
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.00500000
#>                     QMC releps:            NA
#>            Adaptive QMC releps:    0.00031250
#>   Log-likelihood estiamte (SE):   -4.15066843 (0.00000438)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                          0.0157539                       -4.15067
#> CDF                           0.0157552                       -4.15059
#> GenzMonahan                          NA                             NA
#> GenzMonahanA                  0.0157525                       -4.15076
#> QMC                                  NA                             NA
#> QMCA                          0.0157522                       -4.15077
#> 
#> SD & RMSE (/10000.00)
#>      GHQ      AGHQ       CDF GenzMonahan GenzMonahanA QMC      QMCA
#> SD    NA 0.0101435 0.0233646          NA    0.0280678  NA 0.0171751
#> RMSE  NA 0.0101460 0.0174671          NA    0.0289037  NA 0.0196331
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>           NA       0.0024       0.0004           NA       0.0002           NA 
#>         QMCA 
#>       0.0082
sim_experiment(n = 10L, p = 5L, n_threads = 6L)
#>          # brute force samples:      10000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             7
#>                     CDF releps:    0.01000000
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.00031250
#>                     QMC releps:            NA
#>            Adaptive QMC releps:    0.00031250
#>   Log-likelihood estiamte (SE):   -4.74228327 (0.00000886)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                         0.00871868                       -4.74229
#> CDF                          0.00871860                       -4.74230
#> GenzMonahan                          NA                             NA
#> GenzMonahanA                 0.00871825                       -4.74234
#> QMC                                  NA                             NA
#> QMCA                         0.00871736                       -4.74244
#> 
#> SD & RMSE (/10000.00)
#>      GHQ       AGHQ        CDF GenzMonahan GenzMonahanA QMC      QMCA
#> SD    NA 0.00365720 0.00506908          NA   0.01057491  NA 0.0100375
#> RMSE  NA 0.00336134 0.00222698          NA   0.00786363  NA 0.0175209
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>           NA       0.0138       0.0112           NA       0.0962           NA 
#>         QMCA 
#>       0.0230

sim_experiment(n = 3L , p = 7L, n_threads = 6L)
#>          # brute force samples:      10000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             6
#>                     CDF releps:    0.01000000
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.00031250
#>                     QMC releps:            NA
#>            Adaptive QMC releps:    0.00007813
#>   Log-likelihood estiamte (SE):   -2.05643127 (0.00000240)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                           0.127910                       -2.05643
#> CDF                            0.127911                       -2.05642
#> GenzMonahan                          NA                             NA
#> GenzMonahanA                   0.127892                       -2.05657
#> QMC                                  NA                             NA
#> QMCA                           0.127909                       -2.05644
#> 
#> SD & RMSE (/10000.00)
#>      GHQ      AGHQ       CDF GenzMonahan GenzMonahanA QMC      QMCA
#> SD    NA 0.0415225 0.0664377          NA     0.128076  NA 0.0355031
#> RMSE  NA 0.0411920 0.0687539          NA     0.179899  NA 0.0180068
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>           NA       0.0872       0.0002           NA       0.0006           NA 
#>         QMCA 
#>       0.0310
sim_experiment(n = 10L, p = 7L, n_threads = 6L)
#>          # brute force samples:      10000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             7
#>                     CDF releps:    0.01000000
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.00031250
#>                     QMC releps:            NA
#>            Adaptive QMC releps:    0.00031250
#>   Log-likelihood estiamte (SE):   -6.07908563 (0.00003558)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                         0.00229026                       -6.07909
#> CDF                          0.00229027                       -6.07909
#> GenzMonahan                          NA                             NA
#> GenzMonahanA                 0.00229033                       -6.07906
#> QMC                                  NA                             NA
#> QMCA                         0.00229005                       -6.07918
#> 
#> SD & RMSE (/10000.00)
#>      GHQ       AGHQ         CDF GenzMonahan GenzMonahanA QMC       QMCA
#> SD    NA 0.00209120 0.002329381          NA   0.00277851  NA 0.00274544
#> RMSE  NA 0.00226669 0.000614008          NA   0.00358244  NA 0.00273783
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>           NA       0.6962       0.0106           NA       1.1120           NA 
#>         QMCA 
#>       0.3972

sim_experiment(n = 20L, p = 7L, n_threads = 6L)
#>          # brute force samples:      10000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             7
#>                     CDF releps:    0.01000000
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.00062500
#>                     QMC releps:            NA
#>            Adaptive QMC releps:    0.00062500
#>   Log-likelihood estiamte (SE):  -11.43835350 (0.00002357)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                       0.0000107740                       -11.4384
#> CDF                        0.0000107748                       -11.4383
#> GenzMonahan                          NA                             NA
#> GenzMonahanA               0.0000107762                       -11.4382
#> QMC                                  NA                             NA
#> QMCA                       0.0000107743                       -11.4384
#> 
#> SD & RMSE (/10000.00)
#>      GHQ          AGHQ          CDF GenzMonahan GenzMonahanA QMC         QMCA
#> SD    NA 0.00000534394 0.0000700732          NA 0.0000261371  NA 0.0000235443
#> RMSE  NA 0.00000741494 0.0000282347          NA 0.0000284364  NA 0.0000537472
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>           NA       1.2740       0.0336           NA       0.1924           NA 
#>         QMCA 
#>       0.0796
```

Next, we apply the method a number of times for a of combination of
number of observations, `n`, and number of random effects, `p`.

``` r
# number of observations i  n the cluster
n_vals <- 2^1:3
# number of random effects
p_vals <- 2:3
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

We create a table where we summarize the results below. First we start
with the average computation time, then we show the mean scaled RMSE,
and we end by looking at the number of nodes that we need to use with
GHQ. The latter shows why GHQ becomes slower as the cluster size, `n`,
increases. The computation time is in 1000s of a second,
`comp_time_mult`. The mean scaled RMSE is multiplied by
![10^{5}](https://latex.codecogs.com/svg.latex?10%5E%7B5%7D "10^{5}"),
`err_mult`.

``` r
comp_time_mult <- 1000 # millisecond
err_mult <- 1e5
```

``` r
#####
# show number of complete cases
.get_nice_names <- function(x){
  x <- gsub(
    "^GenzMonahan$", "Genz & Monahan (1999)", x)
  x <- gsub(
    "^GenzMonahanA$", "Genz & Monahan (1999) Adaptive", x)
  # fix stupid typo at one point
  x <- gsub("^ADHQ$", "AGHQ", x)
  x <- gsub("^QMCA$", "Adaptive QMC", x)
  x
}

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
  rnames[[2L]] <- .get_nice_names(rnames[[2L]])
  
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

| n   | method/p                       |    2|    3|
|:----|:-------------------------------|----:|----:|
| 2   | GHQ                            |   20|   20|
|     | AGHQ                           |   20|   20|
|     | CDF                            |   20|   20|
|     | Genz & Monahan (1999) Adaptive |   19|   20|
|     | QMC                            |   20|   20|
|     | Adaptive QMC                   |   20|   20|
| 3   | GHQ                            |   20|   20|
|     | AGHQ                           |   20|   20|
|     | CDF                            |   20|   20|
|     | Genz & Monahan (1999) Adaptive |   20|   19|
|     | QMC                            |   20|   20|
|     | Adaptive QMC                   |   20|   20|

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
  rnames[[2L]] <- .get_nice_names(rnames[[2L]])
  
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

| n   | method/p                       |       2|       3|
|:----|:-------------------------------|-------:|-------:|
| 2   | GHQ                            |    0.07|    0.25|
|     | AGHQ                           |    0.09|    0.10|
|     | CDF                            |    0.07|    0.05|
|     | Genz & Monahan (1999)          |        |        |
|     | Genz & Monahan (1999) Adaptive |        |  288.57|
|     | QMC                            |   17.06|   33.07|
|     | Adaptive QMC                   |  388.31|  341.93|
| 3   | GHQ                            |    0.07|    0.35|
|     | AGHQ                           |    0.07|    0.14|
|     | CDF                            |    1.17|    0.88|
|     | Genz & Monahan (1999)          |        |        |
|     | Genz & Monahan (1999) Adaptive |  116.49|        |
|     | QMC                            |   18.93|   36.68|
|     | Adaptive QMC                   |  159.50|  284.79|

``` r
show_run_times(na.rm = TRUE)
```

**NAs have been removed. Cells may not be comparable (means)**

| n   | method/p                       |       2|       3|
|:----|:-------------------------------|-------:|-------:|
| 2   | GHQ                            |    0.07|    0.25|
|     | AGHQ                           |    0.09|    0.10|
|     | CDF                            |    0.07|    0.05|
|     | Genz & Monahan (1999) Adaptive |  321.96|  288.57|
|     | QMC                            |   17.06|   33.07|
|     | Adaptive QMC                   |  388.31|  341.93|
| 3   | GHQ                            |    0.07|    0.35|
|     | AGHQ                           |    0.07|    0.14|
|     | CDF                            |    1.17|    0.88|
|     | Genz & Monahan (1999) Adaptive |  116.49|  109.01|
|     | QMC                            |   18.93|   36.68|
|     | Adaptive QMC                   |  159.50|  284.79|

``` r
show_run_times(TRUE)
```

**Only showing complete cases (means)**

| n   | method/p                       |    2|    3|
|:----|:-------------------------------|----:|----:|
| 2   | GHQ                            |     |     |
|     | AGHQ                           |     |     |
|     | CDF                            |     |     |
|     | Genz & Monahan (1999)          |     |     |
|     | Genz & Monahan (1999) Adaptive |     |     |
|     | QMC                            |     |     |
|     | Adaptive QMC                   |     |     |
| 3   | GHQ                            |     |     |
|     | AGHQ                           |     |     |
|     | CDF                            |     |     |
|     | Genz & Monahan (1999)          |     |     |
|     | Genz & Monahan (1999) Adaptive |     |     |
|     | QMC                            |     |     |
|     | Adaptive QMC                   |     |     |

**Number of complete cases**

|     |    2|    3|
|-----|----:|----:|
| 2   |    0|    0|
| 3   |    0|    0|

``` r

# show medians instead
med_func <- function(x, na.rm)
  apply(x, 1, median, na.rm = na.rm)
show_run_times(meth = med_func, suffix = " (median)", FALSE)
```

**Blank cells have at least one failure (median)**

| n   | method/p                       |      2|      3|
|:----|:-------------------------------|------:|------:|
| 2   | GHQ                            |   0.00|   0.20|
|     | AGHQ                           |   0.00|   0.00|
|     | CDF                            |   0.00|   0.00|
|     | Genz & Monahan (1999)          |       |       |
|     | Genz & Monahan (1999) Adaptive |       |  12.40|
|     | QMC                            |  15.90|  30.80|
|     | Adaptive QMC                   |  52.20|  49.00|
| 3   | GHQ                            |   0.00|   0.20|
|     | AGHQ                           |   0.00|   0.20|
|     | CDF                            |   0.80|   0.40|
|     | Genz & Monahan (1999)          |       |       |
|     | Genz & Monahan (1999) Adaptive |   1.00|       |
|     | QMC                            |  18.30|  25.30|
|     | Adaptive QMC                   |  15.70|  25.70|

``` r
show_run_times(meth = med_func, suffix = " (median)", na.rm = TRUE)
```

**NAs have been removed. Cells may not be comparable (median)**

| n   | method/p                       |      2|      3|
|:----|:-------------------------------|------:|------:|
| 2   | GHQ                            |   0.00|   0.20|
|     | AGHQ                           |   0.00|   0.00|
|     | CDF                            |   0.00|   0.00|
|     | Genz & Monahan (1999) Adaptive |   8.60|  12.40|
|     | QMC                            |  15.90|  30.80|
|     | Adaptive QMC                   |  52.20|  49.00|
| 3   | GHQ                            |   0.00|   0.20|
|     | AGHQ                           |   0.00|   0.20|
|     | CDF                            |   0.80|   0.40|
|     | Genz & Monahan (1999) Adaptive |   1.00|   5.20|
|     | QMC                            |  18.30|  25.30|
|     | Adaptive QMC                   |  15.70|  25.70|

``` r
show_run_times(meth = med_func, suffix = " (median)", TRUE)
```

**Only showing complete cases (median)**

| n   | method/p                       |    2|    3|
|:----|:-------------------------------|----:|----:|
| 2   | GHQ                            |     |     |
|     | AGHQ                           |     |     |
|     | CDF                            |     |     |
|     | Genz & Monahan (1999)          |     |     |
|     | Genz & Monahan (1999) Adaptive |     |     |
|     | QMC                            |     |     |
|     | Adaptive QMC                   |     |     |
| 3   | GHQ                            |     |     |
|     | AGHQ                           |     |     |
|     | CDF                            |     |     |
|     | Genz & Monahan (1999)          |     |     |
|     | Genz & Monahan (1999) Adaptive |     |     |
|     | QMC                            |     |     |
|     | Adaptive QMC                   |     |     |

**Number of complete cases**

|     |    2|    3|
|-----|----:|----:|
| 2   |    0|    0|
| 3   |    0|    0|

``` r

# show quantiles instead
med_func <- function(x, prob = .75, ...)
  apply(x, 1, function(z) quantile(na.omit(z), probs = prob))
show_run_times(meth = med_func, suffix = " (75% quantile)", na.rm = TRUE)
```

**NAs have been removed. Cells may not be comparable (75% quantile)**

| n   | method/p                       |       2|       3|
|:----|:-------------------------------|-------:|-------:|
| 2   | GHQ                            |    0.20|    0.20|
|     | AGHQ                           |    0.20|    0.20|
|     | CDF                            |    0.20|    0.05|
|     | Genz & Monahan (1999) Adaptive |  268.00|  329.00|
|     | QMC                            |   24.65|   37.45|
|     | Adaptive QMC                   |  838.15|  168.00|
| 3   | GHQ                            |    0.20|    0.40|
|     | AGHQ                           |    0.20|    0.20|
|     | CDF                            |    1.45|    0.90|
|     | Genz & Monahan (1999) Adaptive |   46.40|   40.80|
|     | QMC                            |   27.65|   40.70|
|     | Adaptive QMC                   |   69.80|  297.45|

``` r
show_run_times(meth = med_func, suffix = " (75% quantile)", TRUE)
```

**Only showing complete cases (75% quantile)**

| n   | method/p                       |    2|    3|
|:----|:-------------------------------|----:|----:|
| 2   | GHQ                            |     |     |
|     | AGHQ                           |     |     |
|     | CDF                            |     |     |
|     | Genz & Monahan (1999)          |     |     |
|     | Genz & Monahan (1999) Adaptive |     |     |
|     | QMC                            |     |     |
|     | Adaptive QMC                   |     |     |
| 3   | GHQ                            |     |     |
|     | AGHQ                           |     |     |
|     | CDF                            |     |     |
|     | Genz & Monahan (1999)          |     |     |
|     | Genz & Monahan (1999) Adaptive |     |     |
|     | QMC                            |     |     |
|     | Adaptive QMC                   |     |     |

**Number of complete cases**

|     |    2|    3|
|-----|----:|----:|
| 2   |    0|    0|
| 3   |    0|    0|

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
  rnames[[2L]] <- .get_nice_names(rnames[[2L]])
  
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

| n   | method/p                       |     2|     3|
|:----|:-------------------------------|-----:|-----:|
| 2   | GHQ                            |  4.84|  4.40|
|     | AGHQ                           |  4.43|  4.15|
|     | CDF                            |  0.96|  0.81|
|     | Genz & Monahan (1999)          |      |      |
|     | Genz & Monahan (1999) Adaptive |      |  3.63|
|     | QMC                            |  4.55|  4.79|
|     | Adaptive QMC                   |  5.90|  5.84|
| 3   | GHQ                            |  8.59|  7.02|
|     | AGHQ                           |  4.43|  4.39|
|     | CDF                            |  6.82|  9.71|
|     | Genz & Monahan (1999)          |      |      |
|     | Genz & Monahan (1999) Adaptive |  5.98|      |
|     | QMC                            |  8.91|  9.67|
|     | Adaptive QMC                   |  6.47|  6.98|

``` r
show_scaled_mean_rmse(na.rm = TRUE)
```

**NAs have been removed. Cells may not be comparable**

| n   | method/p                       |     2|     3|
|:----|:-------------------------------|-----:|-----:|
| 2   | GHQ                            |  4.84|  4.40|
|     | AGHQ                           |  4.43|  4.15|
|     | CDF                            |  0.96|  0.81|
|     | Genz & Monahan (1999) Adaptive |  4.52|  3.63|
|     | QMC                            |  4.55|  4.79|
|     | Adaptive QMC                   |  5.90|  5.84|
| 3   | GHQ                            |  8.59|  7.02|
|     | AGHQ                           |  4.43|  4.39|
|     | CDF                            |  6.82|  9.71|
|     | Genz & Monahan (1999) Adaptive |  5.98|  6.21|
|     | QMC                            |  8.91|  9.67|
|     | Adaptive QMC                   |  6.47|  6.98|

``` r
show_scaled_mean_rmse(TRUE)
```

**Only showing complete cases**

| n   | method/p                       |    2|    3|
|:----|:-------------------------------|----:|----:|
| 2   | GHQ                            |     |     |
|     | AGHQ                           |     |     |
|     | CDF                            |     |     |
|     | Genz & Monahan (1999)          |     |     |
|     | Genz & Monahan (1999) Adaptive |     |     |
|     | QMC                            |     |     |
|     | Adaptive QMC                   |     |     |
| 3   | GHQ                            |     |     |
|     | AGHQ                           |     |     |
|     | CDF                            |     |     |
|     | Genz & Monahan (1999)          |     |     |
|     | Genz & Monahan (1999) Adaptive |     |     |
|     | QMC                            |     |     |
|     | Adaptive QMC                   |     |     |

**Number of complete cases**

|     |    2|    3|
|-----|----:|----:|
| 2   |    0|    0|
| 3   |    0|    0|

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

| n   | quantile/p |      2|      3|
|:----|:-----------|------:|------:|
| 2   | 0%         |   5.00|   6.00|
|     | 25%        |   8.00|   7.75|
|     | 50%        |  11.00|   9.00|
|     | 75%        |  12.25|  10.00|
|     | 100%       |  21.00|  18.00|
| 3   | 0%         |   4.00|   6.00|
|     | 25%        |   9.00|   8.00|
|     | 50%        |  10.00|   9.00|
|     | 75%        |  12.25|  11.00|
|     | 100%       |  23.00|  18.00|

**Number of complete cases**

|     |    2|    3|
|-----|----:|----:|
| 2   |   20|   20|
| 3   |   20|   20|

``` r
show_n_nodes(TRUE)
```

**Only showing complete cases (Adaptive GHQ)**

| n   | quantile/p |      2|      3|
|:----|:-----------|------:|------:|
| 2   | 0%         |   4.00|   4.00|
|     | 25%        |   6.75|   6.00|
|     | 50%        |   7.00|   7.00|
|     | 75%        |   9.00|   7.25|
|     | 100%       |  10.00|  12.00|
| 3   | 0%         |   4.00|   5.00|
|     | 25%        |   6.00|   6.00|
|     | 50%        |   7.00|   7.00|
|     | 75%        |   7.25|   7.25|
|     | 100%       |  15.00|  11.00|

**Number of complete cases**

|     |    2|    3|
|-----|----:|----:|
| 2   |   20|   20|
| 3   |   20|   20|

Quasi-Monte Carlo Method
------------------------

We use the Fortran code to from the `randtoolbox` package to generate
the Sobol sequences which we use for Quasi-Monte Carlo method. However,
there is a big overhead which can be avoided in the package so we have
created our own interface to the Fortran functions. As we show below,
the difference in computation time is quite substantial.

``` r
# assign function to get Sobol sequences from this package
library(randtoolbox)
#> Loading required package: rngWELL
#> This is randtoolbox. For an overview, type 'help("randtoolbox")'.
get_sobol_seq <- function(dim, scrambling = 0L, seed = formals(sobol)$seed){
  ptr <- mixprobit:::get_sobol_obj(dimen = dim, scrambling = scrambling, 
                                   seed = seed)
  
  function(n)
    mixprobit:::eval_sobol(n, ptr = ptr)
}

#####
# differences when initializing
dim <- 3L
n   <- 10L
all.equal(get_sobol_seq(dim)(n), t(sobol(n = n, dim = dim)))
#> [1] TRUE
microbenchmark::microbenchmark(
  mixprobit = get_sobol_seq(dim)(n), 
  randtoolbox = sobol(n = n, dim = dim), times = 1000)
#> Unit: microseconds
#>         expr    min    lq  mean median    uq  max neval
#>    mixprobit  7.996  9.20 13.36  11.23 12.21 1918  1000
#>  randtoolbox 54.003 56.84 63.03  58.61 62.84 1702  1000

# w/ larger dim
dim <- 50L
all.equal(get_sobol_seq(dim)(n), t(sobol(n = n, dim = dim)))
#> [1] TRUE
microbenchmark::microbenchmark(
  mixprobit = get_sobol_seq(dim)(n), 
  randtoolbox = sobol(n = n, dim = dim), times = 1000)
#> Unit: microseconds
#>         expr   min    lq  mean median   uq  max neval
#>    mixprobit 16.86 19.21 22.21  21.99 23.9  109  1000
#>  randtoolbox 71.05 76.61 84.20  79.61 84.0 2382  1000

#####
# after having initialized
dim <- 3L
sobol_obj <- get_sobol_seq(dim)
invisible(sobol_obj(1L))
invisible(sobol(n = 1L, dim = dim))

n <- 10L
all.equal(sobol_obj(n), t(sobol(n = n, dim = dim, init = FALSE)))
#> [1] TRUE
microbenchmark::microbenchmark(
  `mixprobit   (1 point)`        = sobol_obj(1L), 
  `randtoolbox (1 point)`        = sobol(n = 1L, dim = dim, init = FALSE), 
  `mixprobit   (100 points)`     = sobol_obj(100L), 
  `randtoolbox (100 points)`     = sobol(n = 100L, dim = dim, init = FALSE), 
  `mixprobit   (10000 points)`   = sobol_obj(10000L), 
  `randtoolbox (10000 points)`   = sobol(n = 10000L, dim = dim, init = FALSE), 
  
  times = 1000)
#> Unit: microseconds
#>                        expr     min      lq    mean  median      uq     max
#>       mixprobit   (1 point)   3.357   4.117   5.395   5.411   6.017   37.91
#>       randtoolbox (1 point)  36.772  39.755  42.541  41.048  42.528  127.01
#>    mixprobit   (100 points)   5.432   6.557   7.938   7.790   8.529   44.07
#>    randtoolbox (100 points)  41.917  44.310  47.251  45.623  47.284  140.90
#>  mixprobit   (10000 points) 195.290 207.130 224.811 209.834 213.380 2014.05
#>  randtoolbox (10000 points) 368.478 383.034 439.691 388.126 395.653 3112.28
#>  neval
#>   1000
#>   1000
#>   1000
#>   1000
#>   1000
#>   1000

#####
# similar conclusions apply w/ scrambling
dim <- 10L
n <- 10L
all.equal(get_sobol_seq(dim, scrambling = 1L)(n), 
          t(sobol(n = n, dim = dim, scrambling = 1L)))
#> [1] TRUE

microbenchmark::microbenchmark(
  mixprobit = get_sobol_seq(dim, scrambling = 1L)(n), 
  randtoolbox = sobol(n = n, dim = dim, scrambling = 1L), times = 1000)
#> Unit: microseconds
#>         expr   min    lq  mean median    uq    max neval
#>    mixprobit 273.8 280.9 288.5  282.7 286.5 2195.6  1000
#>  randtoolbox 324.8 334.4 342.1  338.2 342.0  617.6  1000

sobol_obj <- get_sobol_seq(dim, scrambling = 1L)
invisible(sobol_obj(1L))
invisible(sobol(n = 1L, dim = dim, scrambling = 1L))

all.equal(sobol_obj(n), t(sobol(n = n, dim = dim, init = FALSE)))
#> [1] TRUE
microbenchmark::microbenchmark(
  `mixprobit   (1 point)`        = sobol_obj(1L), 
  `randtoolbox (1 point)`        = sobol(n = 1L, dim = dim, init = FALSE), 
  `mixprobit   (100 points)`     = sobol_obj(100L), 
  `randtoolbox (100 points)`     = sobol(n = 100L, dim = dim, init = FALSE), 
  `mixprobit   (10000 points)`   = sobol_obj(10000L), 
  `randtoolbox (10000 points)`   = sobol(n = 10000L, dim = dim, init = FALSE), 
  
  times = 1000)
#> Unit: microseconds
#>                        expr     min      lq     mean   median       uq      max
#>       mixprobit   (1 point)   3.407   4.453    6.829    5.917    7.402    54.60
#>       randtoolbox (1 point)  38.373  41.239   47.891   44.537   49.005   255.99
#>    mixprobit   (100 points)   7.268   8.948   11.435   10.218   12.002    56.76
#>    randtoolbox (100 points)  48.285  51.647   59.618   54.676   58.998  1372.98
#>  mixprobit   (10000 points) 345.635 395.233  472.820  409.373  427.584  3534.30
#>  randtoolbox (10000 points) 948.764 993.316 1268.212 1011.101 1112.146 34449.88
#>  neval
#>   1000
#>   1000
#>   1000
#>   1000
#>   1000
#>   1000
```

Lastly, the C++ interface we have created allow us to call the Fortran
from C++ directly. This was the primary motivation for creating our own
interface.

References
----------

Barrett, Jessica, Peter Diggle, Robin Henderson, and David
Taylor-Robinson. 2015. “Joint Modelling of Repeated Measurements and
Time-to-Event Outcomes: Flexible Model Specification and Exact
Likelihood Inference.” *Journal of the Royal Statistical Society: Series
B (Statistical Methodology)* 77 (1): 131–48.
<https://doi.org/10.1111/rssb.12060>.

Genz, Alan, and Frank Bretz. 2002. “Comparison of Methods for the
Computation of Multivariate T Probabilities.” *Journal of Computational
and Graphical Statistics* 11 (4). Taylor & Francis: 950–71.
<https://doi.org/10.1198/106186002394>.

Genz, Alan., and John. Monahan. 1998. “Stochastic Integration Rules for
Infinite Regions.” *SIAM Journal on Scientific Computing* 19 (2):
426–39. <https://doi.org/10.1137/S1064827595286803>.

Genz, Alan, and John Monahan. 1999. “A Stochastic Algorithm for
High-Dimensional Integrals over Unbounded Regions with Gaussian Weight.”
*Journal of Computational and Applied Mathematics* 112 (1): 71–81.
<https://doi.org/https://doi.org/10.1016/S0377-0427(99)00214-9>.

Hajivassiliou, Vassilis, Daniel McFadden, and Paul Ruud. 1996.
“Simulation of Multivariate Normal Rectangle Probabilities and Their
Derivatives Theoretical and Computational Results.” *Journal of
Econometrics* 72 (1): 85–134.
<https://doi.org/https://doi.org/10.1016/0304-4076(94)01716-6>.

Liu, Qing, and Donald A. Pierce. 1994. “A Note on Gauss-Hermite
Quadrature.” *Biometrika* 81 (3). \[Oxford University Press, Biometrika
Trust\]: 624–29. <http://www.jstor.org/stable/2337136>.

Ochi, Y., and Ross L. Prentice. 1984. “Likelihood Inference in a
Correlated Probit Regression Model.” *Biometrika* 71 (3). \[Oxford
University Press, Biometrika Trust\]: 531–43.
<http://www.jstor.org/stable/2336562>.

Pawitan, Y., M. Reilly, E. Nilsson, S. Cnattingius, and P. Lichtenstein.
2004. “Estimation of Genetic and Environmental Factors for Binary Traits
Using Family Data.” *Statistics in Medicine* 23 (3): 449–65.
<https://doi.org/10.1002/sim.1603>.
