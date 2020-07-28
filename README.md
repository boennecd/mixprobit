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
  get_sim_mth <- function(y, eta, Z, S, maxpts, abseps = -1, releps = 1e-3, 
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
                      releps = 1e-4, n_seqs = 10L, abseps)
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
      rWishart(1, 5 * p, diag(1 / p / 5, p)))
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
#> [1] 2.019
var(replicate(1000, with(get_sim_dat(10, 3), u %*% Z + eta)))
#> [1] 2.026
var(replicate(1000, with(get_sim_dat(10, 4), u %*% Z + eta)))
#> [1] 2.014
var(replicate(1000, with(get_sim_dat(10, 5), u %*% Z + eta)))
#> [1] 2.042
var(replicate(1000, with(get_sim_dat(10, 6), u %*% Z + eta)))
#> [1] 1.978
var(replicate(1000, with(get_sim_dat(10, 7), u %*% Z + eta)))
#> [1] 1.984
var(replicate(1000, with(get_sim_dat(10, 8), u %*% Z + eta)))
#> [1] 2.013
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
truth_maybe_cdf
#> [1] 6.182e-05
#> attr(,"inform")
#> [1] 0
#> attr(,"error")
#> [1] 6.144e-08
#> attr(,"intvls")
#> [1] 9952

truth_maybe_qmc <- wd(
  aprx$get_qmc(y = y, eta = eta, Z = Z, S = S, maxpts = 1e7, 
               releps = 1e-11)())
truth_maybe_qmc
#> [1] 6.184e-05
#> attr(,"intvls")
#> [1] 10000000
#> attr(,"error")
#> [1] 6.227e-10

truth_maybe_Aqmc <- wd(
  aprx$get_Aqmc(y = y, eta = eta, Z = Z, S = S, maxpts = 1e7, 
                releps = 1e-11)())
truth_maybe_Aqmc
#> [1] 6.184e-05
#> attr(,"intvls")
#> [1] 10000000
#> attr(,"error")
#> [1] 3.495e-10

truth_maybe_Amc <- wd(
  aprx$get_Asim_mth(y = y, eta = eta, Z = Z, S = S, maxpts = 1e7, 
                    abseps = 1e-11)(2L))
truth_maybe_Amc
#> [1] 6.188e-05
#> attr(,"error")
#> [1] 2.401e-08
#> attr(,"inform")
#> [1] 0
#> attr(,"inivls")
#> [1] 24181

truth <- wd(
  mixprobit:::aprx_binary_mix_brute(y = y, eta = eta, Z = Z, Sigma = S, 
                                    n_sim = 1e8, n_threads = 6L))

c(Estiamte = truth, SE = attr(truth, "SE"),  
  `Estimate (log)` = log(c(truth)),  
  `SE (log)` = abs(attr(truth, "SE") / truth))
#>       Estiamte             SE Estimate (log)       SE (log) 
#>      6.184e-05      2.566e-10     -9.691e+00      4.149e-06

tr <- c(truth)
all.equal(tr, c(truth_maybe_cdf))
#> [1] "Mean relative difference: 0.000305"
all.equal(tr, c(truth_maybe_qmc))
#> [1] "Mean relative difference: 2.435e-05"
all.equal(tr, c(truth_maybe_Aqmc))
#> [1] "Mean relative difference: 9.696e-06"
all.equal(tr, c(truth_maybe_Amc))
#> [1] "Mean relative difference: 0.0005788"

# compare with using fewer samples and GHQ
all.equal(tr,   GHQ_R())
#> [1] "Mean relative difference: 2.226e-05"
all.equal(tr,   GHQ_cpp())
#> [1] "Mean relative difference: 2.226e-05"
all.equal(tr,   AGHQ_cpp())
#> [1] "Mean relative difference: 2.068e-06"
comp <- function(f, ...)
  mean(replicate(10, abs((tr - c(f())) / tr)))
comp(cdf_aprx_R)
#> [1] 9.598e-05
comp(qmc_aprx)
#> [1] 0.001256
comp(qmc_Aaprx)
#> [1] 0.0002439
comp(cdf_aprx_cpp)
#> [1] 0.0003223
comp(function() sim_aprx(1L))
#> [1] 0.006832
comp(function() sim_aprx(2L))
#> [1] 0.004851
comp(function() sim_aprx(3L))
#> [1] 0.01262
comp(function() sim_aprx(4L))
#> [1] 0.004099
comp(function() sim_Aaprx(1L))
#> [1] 0.0003946
comp(function() sim_Aaprx(2L))
#> [1] 0.0002729
comp(function() sim_Aaprx(3L))
#> [1] 0.0007611
comp(function() sim_Aaprx(4L))
#> [1] 0.0006802

# compare computations times
system.time(GHQ_R()) # way too slow (seconds!). Use C++ method instead
#>    user  system elapsed 
#>   1.347   0.000   1.347
microbenchmark::microbenchmark(
  `GHQ (C++)` = GHQ_cpp(), `AGHQ (C++)` = AGHQ_cpp(),
  `CDF` = cdf_aprx_R(), `CDF (C++)` = cdf_aprx_cpp(),
  QMC = qmc_aprx(), `QMC Adaptive` = qmc_Aaprx(),
  `Genz & Monahan (1)` = sim_aprx(1L), `Genz & Monahan (2)` = sim_aprx(2L),
  `Genz & Monahan (3)` = sim_aprx(3L), `Genz & Monahan (4)` = sim_aprx(4L),
  `Genz & Monahan Adaptive (2)` = sim_Aaprx(2L),
  times = 10)
#> Unit: milliseconds
#>                         expr    min     lq  mean median    uq   max neval
#>                    GHQ (C++) 34.898 35.297 35.59  35.46 35.86 36.58    10
#>                   AGHQ (C++) 35.597 35.971 36.08  36.05 36.18 36.58    10
#>                          CDF 21.084 21.517 21.55  21.56 21.68 21.75    10
#>                    CDF (C++)  7.681  7.777 11.26  12.65 12.69 13.18    10
#>                          QMC 27.691 27.799 27.97  27.85 28.28 28.39    10
#>                 QMC Adaptive 28.759 29.315 29.38  29.34 29.38 30.13    10
#>           Genz & Monahan (1) 25.170 25.529 25.80  25.83 26.24 26.28    10
#>           Genz & Monahan (2) 26.239 26.693 26.96  26.89 27.24 27.84    10
#>           Genz & Monahan (3) 25.103 25.729 25.83  25.79 26.12 26.42    10
#>           Genz & Monahan (4) 25.318 25.407 25.74  25.62 25.90 26.77    10
#>  Genz & Monahan Adaptive (2) 11.097 13.171 13.73  13.58 13.95 16.22    10
```

More Rigorous Comparison
------------------------

We are interested in a more rigorous comparison. Therefor, we define a
function below which for given number of observation in the cluster,
`n`, and given number of random effects, `p`, performs a repeated number
of runs with each of the methods and returns the computation time (among
other output). To make a fair comparison, we fix the relative error of
the methods before hand such that the relative error is below `releps`,
![2\\times 10^{-4}](https://latex.codecogs.com/svg.latex?2%5Ctimes%2010%5E%7B-4%7D "2\times 10^{-4}").
Ground truth is computed with brute force MC using `n_brute`,
![10^{7}](https://latex.codecogs.com/svg.latex?10%5E%7B7%7D "10^{7}"),
samples.

Since GHQ is deterministic, we use a number of nodes such that this
number of nodes or `streak_length`, 4, less value of nodes with GHQ
gives a relative error which is below the threshold. We use a minimum of
4 nodes at the time of this writing. The error of the simulation based
methods is approximated using `n_reps`, 20, replications.

``` r
# default parameters
ex_params <- list(
  streak_length = 4L, 
  max_b = 25L, 
  max_maxpts = 2500000L, 
  releps = 2e-4,
  min_releps = 1e-6,
  key_use = 3L, 
  n_reps = 20L, 
  n_runs = 5L, 
  n_brute = 1e7, 
  n_brute_max = 1e8, 
  n_brute_sds = 4, 
  qmc_n_seqs = 10L)
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
    test_val <- (log(vals) - log(truth)) / log(truth) 
    if(!all(is.finite(test_val)))
      stop("non-finite 'vals'")
    sqrt(mean(test_val^2)) < releps / 2
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
      releps_use <- releps * 1000
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
          warning("found no releps")
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
  formals(aprx$get_qmc)$n_seqs <- ex_params$qmc_n_seqs
  qmc_releps <- if(is_to_large_for_ghq)
    NA_integer_ else get_releps(aprx$get_qmc)
  qmc_func <- if(!is.na(qmc_releps))
     wd(aprx$get_qmc(y = y, eta = eta, Z = Z, S = S, 
                     maxpts = ex_params$max_maxpts, abseps = -1,
                     releps = qmc_releps, 
                     n_seqs = ex_params$qmc_n_seqs))
  else 
    NA

  # get function to use with adaptive QMC
  Aqmc_releps <- get_releps(aprx$get_Aqmc)
  formals(aprx$get_Aqmc)$n_seqs <- ex_params$qmc_n_seqs
  Aqmc_func <- if(!is.null(Aqmc_releps))
    wd(aprx$get_Aqmc(y = y, eta = eta, Z = Z, S = S, 
                     maxpts = ex_params$max_maxpts, abseps = -1,
                     releps = Aqmc_releps, 
                     n_seqs = ex_params$qmc_n_seqs))
  else 
    NA
    
  # perform the comparison
  out <- sapply(
    list(GHQ = ghq_func, AGHQ = aghq_func, CDF = cdf_func, 
         GenzMonahan = sim_func, GenzMonahanA = Asim_func, 
         QMC = qmc_func, QMCA = Aqmc_func), 
    function(func){
      if(!is.function(func) && is.na(func)){
        out <- rep(NA_real_, 7L)
        names(out) <- c("mean", "sd", "mse", "user.self", 
                        "sys.self", "elapsed", "rel_rmse")
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
        mse <- mean((vs - truth)^2)
        rel_rmse <- sqrt(mean(((log(vs) - log(truth)) / log(truth))^2))
        
      } else {
        # we combine the variance estimators
        sd_use <- sqrt(mean(vals["sd", ]^2))
        vals <- vals["value", ]
        mse <- mean((vals - truth)^2)
        rel_rmse <- sqrt(mean(((log(vals) - log(truth)) / log(truth))^2))
        
      }
      
      c(mean = mean(vals), sd = sd_use, mse = mse, ti[1:3] / n_runs, 
        rel_rmse = rel_rmse)            
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
    sprintf("  Log-likelihood estimate (SE): %13.8f (%.8f)", x$ll_truth, 
            x$SE_truth), 
    "", sep = "\n")
  
  xx <- x$vals_n_comp_time["mean", ]
  print(cbind(`Mean estimate (likelihood)`     = xx, 
              `Mean estimate (log-likelihood)` = log(xx)))
  
  mult <- exp(ceiling(log10(1 / ex_params$releps)) * log(10))
  cat(sprintf("\nSD & RMSE (/%.2f)\n", mult))
  print(rbind(SD   = x$vals_n_comp_time ["sd", ],  
              RMSE = sqrt(x$vals_n_comp_time ["mse", ]), 
              `Rel RMSE` = x$vals_n_comp_time["rel_rmse", ]) * mult)
  
  cat("\nComputation times\n")
  print(x$vals_n_comp_time["elapsed", ])
}

set.seed(1)
sim_experiment(n = 3L , p = 2L, n_threads = 6L)
#>          # brute force samples:      10000000
#>                   # nodes  GHQ:            10
#>                   # nodes AGHQ:             6
#>                     CDF releps:    0.10000000
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.20000000
#>                     QMC releps:    0.00039063
#>            Adaptive QMC releps:    0.00039063
#>   Log-likelihood estimate (SE):   -3.20961388 (0.00000229)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                           0.0403722                       -3.20961
#> AGHQ                          0.0403721                       -3.20962
#> CDF                           0.0403682                       -3.20971
#> GenzMonahan                          NA                             NA
#> GenzMonahanA                  0.0403821                       -3.20937
#> QMC                           0.0403744                       -3.20956
#> QMCA                          0.0403676                       -3.20973
#> 
#> SD & RMSE (/10000.00)
#>                GHQ      AGHQ       CDF GenzMonahan GenzMonahanA       QMC
#> SD       0.0425442 0.0382665 0.1840377          NA     0.147125 0.0561127
#> RMSE     0.0406504 0.0386503 0.0839231          NA     0.117434 0.0478307
#> Rel RMSE 0.3137423 0.2983075 0.6477030          NA     0.906092 0.3690803
#>               QMCA
#> SD       0.0518171
#> RMSE     0.0848911
#> Rel RMSE 0.6551877
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>       0.0000       0.0002       0.0002           NA       0.0000       0.0082 
#>         QMCA 
#>       0.0022
sim_experiment(n = 10L, p = 2L, n_threads = 6L)
#>          # brute force samples:      10000000
#>                   # nodes  GHQ:            11
#>                   # nodes AGHQ:             9
#>                     CDF releps:    0.00625000
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.00156250
#>                     QMC releps:    0.00078125
#>            Adaptive QMC releps:    0.00078125
#>   Log-likelihood estimate (SE):   -5.75701490 (0.00001203)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                          0.00315992                       -5.75721
#> AGHQ                         0.00316103                       -5.75686
#> CDF                          0.00316142                       -5.75673
#> GenzMonahan                          NA                             NA
#> GenzMonahanA                 0.00316016                       -5.75713
#> QMC                          0.00316023                       -5.75711
#> QMCA                         0.00315908                       -5.75748
#> 
#> SD & RMSE (/10000.00)
#>                 GHQ      AGHQ       CDF GenzMonahan GenzMonahanA        QMC
#> SD       0.00628152 0.0175367 0.0324666          NA    0.0191575 0.00912528
#> RMSE     0.00619129 0.0156882 0.0221506          NA    0.0110268 0.01835440
#> Rel RMSE 0.34027070 0.8622205 1.2170465          NA    0.6060859 1.00880640
#>                QMCA
#> SD       0.00775619
#> RMSE     0.01840486
#> Rel RMSE 1.01191099
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>       0.0002       0.0000       0.0008           NA       0.0056       0.0324 
#>         QMCA 
#>       0.0046

sim_experiment(n = 3L , p = 5L, n_threads = 6L)
#>          # brute force samples:      10000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             6
#>                     CDF releps:    0.10000000
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.20000000
#>                     QMC releps:            NA
#>            Adaptive QMC releps:    0.00078125
#>   Log-likelihood estimate (SE):   -4.28558719 (0.00000480)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                          0.0137655                       -4.28559
#> CDF                           0.0137658                       -4.28557
#> GenzMonahan                          NA                             NA
#> GenzMonahanA                  0.0137663                       -4.28553
#> QMC                                  NA                             NA
#> QMCA                          0.0137639                       -4.28571
#> 
#> SD & RMSE (/10000.00)
#>          GHQ       AGHQ       CDF GenzMonahan GenzMonahanA QMC      QMCA
#> SD        NA 0.00548642 0.0662419          NA    0.0555173  NA 0.0328270
#> RMSE      NA 0.00578451 0.0334886          NA    0.0546603  NA 0.0289218
#> Rel RMSE  NA 0.09805812 0.5675991          NA    0.9262607  NA 0.4903471
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>           NA       0.0022       0.0002           NA       0.0002           NA 
#>         QMCA 
#>       0.0024
sim_experiment(n = 10L, p = 5L, n_threads = 6L)
#>          # brute force samples:      10000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             6
#>                     CDF releps:    0.00625000
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.00039063
#>                     QMC releps:            NA
#>            Adaptive QMC releps:    0.00078125
#>   Log-likelihood estimate (SE):   -8.17648526 (0.00002096)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                        0.000281184                       -8.17650
#> CDF                         0.000280936                       -8.17738
#> GenzMonahan                          NA                             NA
#> GenzMonahanA                0.000281158                       -8.17659
#> QMC                                  NA                             NA
#> QMCA                        0.000281130                       -8.17669
#> 
#> SD & RMSE (/10000.00)
#>          GHQ       AGHQ        CDF GenzMonahan GenzMonahanA QMC        QMCA
#> SD        NA 0.00139986 0.00308127          NA  0.000426338  NA 0.000789658
#> RMSE      NA 0.00145025 0.00474314          NA  0.000530630  NA 0.001633417
#> Rel RMSE  NA 0.63113957 2.06616128          NA  0.230842606  NA 0.710572919
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>           NA       0.0060       0.0016           NA       0.2070           NA 
#>         QMCA 
#>       0.0184

sim_experiment(n = 3L , p = 7L, n_threads = 6L)
#>          # brute force samples:      10000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             7
#>                     CDF releps:    0.20000000
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.00039063
#>                     QMC releps:            NA
#>            Adaptive QMC releps:    0.00039063
#>   Log-likelihood estimate (SE):   -3.00436905 (0.00001397)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                          0.0495704                       -3.00436
#> CDF                           0.0495616                       -3.00454
#> GenzMonahan                          NA                             NA
#> GenzMonahanA                  0.0495644                       -3.00448
#> QMC                                  NA                             NA
#> QMCA                          0.0495698                       -3.00437
#> 
#> SD & RMSE (/10000.00)
#>          GHQ      AGHQ      CDF GenzMonahan GenzMonahanA QMC      QMCA
#> SD        NA 0.0118649 0.270362          NA    0.0750196  NA 0.0694929
#> RMSE      NA 0.0108531 0.102866          NA    0.1105077  NA 0.1118697
#> Rel RMSE  NA 0.0728769 0.690809          NA    0.7420926  NA 0.7512423
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>           NA       0.2704       0.0002           NA       0.0138           NA 
#>         QMCA 
#>       0.0240
sim_experiment(n = 10L, p = 7L, n_threads = 6L)
#>          # brute force samples:      10000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             7
#>                     CDF releps:    0.00312500
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.00078125
#>                     QMC releps:            NA
#>            Adaptive QMC releps:    0.00078125
#>   Log-likelihood estimate (SE):   -5.38268291 (0.00004008)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                         0.00459523                       -5.38274
#> CDF                          0.00459344                       -5.38313
#> GenzMonahan                          NA                             NA
#> GenzMonahanA                 0.00459603                       -5.38256
#> QMC                                  NA                             NA
#> QMCA                         0.00459544                       -5.38269
#> 
#> SD & RMSE (/10000.00)
#>          GHQ       AGHQ       CDF GenzMonahan GenzMonahanA QMC      QMCA
#> SD        NA 0.00394589 0.0338897          NA    0.0139356  NA 0.0128410
#> RMSE      NA 0.00595997 0.0423745          NA    0.0157182  NA 0.0064302
#> Rel RMSE  NA 0.24096867 1.7139237          NA    0.6354066  NA 0.2599421
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>           NA       0.6602       0.0018           NA       0.1938           NA 
#>         QMCA 
#>       0.0752
sim_experiment(n = 20L, p = 7L, n_threads = 6L)
#>          # brute force samples:      10000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             7
#>                     CDF releps:    0.00312500
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.00156250
#>                     QMC releps:            NA
#>            Adaptive QMC releps:    0.00156250
#>   Log-likelihood estimate (SE):  -11.53610205 (0.00005639)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                      0.00000977084                       -11.5361
#> CDF                       0.00000976374                       -11.5368
#> GenzMonahan                          NA                             NA
#> GenzMonahanA              0.00000976411                       -11.5368
#> QMC                                  NA                             NA
#> QMCA                      0.00000977045                       -11.5361
#> 
#> SD & RMSE (/10000.00)
#>          GHQ         AGHQ          CDF GenzMonahan GenzMonahanA QMC
#> SD        NA 0.0000119040 0.0000917157          NA 0.0000592079  NA
#> RMSE      NA 0.0000121726 0.0000846883          NA 0.0000843849  NA
#> Rel RMSE  NA 0.1080068540 0.7517498453          NA 0.7490842696  NA
#>                  QMCA
#> SD       0.0000564266
#> RMSE     0.0000415496
#> Rel RMSE 0.3687212135
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>           NA       1.2198       0.0102           NA       0.2680           NA 
#>         QMCA 
#>       0.1018
```

Next, we apply the method a number of times for a of combination of
number of observations, `n`, and number of random effects, `p`.

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
      message(sprintf("Loading results with n %3d and p %3d", n, p))
      
    
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

| n   | method/p                       |    2|    3|    4|    5|    6|    7|
|:----|:-------------------------------|----:|----:|----:|----:|----:|----:|
| 2   | GHQ                            |   99|  100|  100|    0|    0|    0|
|     | AGHQ                           |  100|  100|  100|  100|  100|  100|
|     | CDF                            |  100|  100|  100|  100|  100|  100|
|     | Genz & Monahan (1999) Adaptive |  100|  100|  100|   99|  100|  100|
|     | QMC                            |  100|  100|  100|    0|    0|    0|
|     | Adaptive QMC                   |  100|  100|  100|  100|  100|  100|
| 4   | GHQ                            |   99|   99|  100|    0|    0|    0|
|     | AGHQ                           |  100|   99|  100|  100|  100|  100|
|     | CDF                            |  100|   99|  100|  100|  100|  100|
|     | Genz & Monahan (1999) Adaptive |   99|   99|  100|  100|  100|  100|
|     | QMC                            |  100|   99|  100|    0|    0|    0|
|     | Adaptive QMC                   |  100|  100|  100|  100|  100|  100|
| 8   | GHQ                            |   99|  100|  100|    0|    0|    0|
|     | AGHQ                           |  100|  100|  100|  100|  100|  100|
|     | CDF                            |  100|  100|  100|  100|  100|  100|
|     | Genz & Monahan (1999) Adaptive |   99|   99|  100|  100|  100|  100|
|     | QMC                            |  100|  100|  100|    0|    0|    0|
|     | Adaptive QMC                   |  100|  100|  100|  100|  100|  100|
| 16  | GHQ                            |   81|  100|  100|    0|    0|    0|
|     | AGHQ                           |  100|  100|  100|  100|  100|  100|
|     | CDF                            |  100|  100|  100|  100|  100|  100|
|     | Genz & Monahan (1999) Adaptive |  100|  100|  100|  100|  100|  100|
|     | QMC                            |  100|  100|   97|    0|    0|    0|
|     | Adaptive QMC                   |  100|  100|  100|  100|  100|  100|
| 32  | AGHQ                           |  100|  100|  100|  100|  100|  100|
|     | CDF                            |  100|  100|  100|  100|  100|  100|
|     | Genz & Monahan (1999) Adaptive |  100|  100|  100|  100|  100|  100|
|     | Adaptive QMC                   |  100|  100|  100|  100|  100|  100|

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
  
  is_complete <- apply(comp_times, 3, function(x){
    if(remove_nas){
      consider <- !apply(is.na(x), 1L, all)
      apply(!is.na(x[consider, , drop = FALSE]), 2, all)
    } else 
      rep(TRUE, NCOL(x))
  })
  
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

| n   | method/p                       |       2|       3|       4|       5|       6|       7|
|:----|:-------------------------------|-------:|-------:|-------:|-------:|-------:|-------:|
| 2   | GHQ                            |        |    0.19|    0.89|        |        |        |
|     | AGHQ                           |    0.04|    0.14|    0.57|    4.45|   24.97|  173.44|
|     | CDF                            |    0.05|    0.06|    0.03|    0.04|    0.05|    0.03|
|     | Genz & Monahan (1999)          |        |        |        |        |        |        |
|     | Genz & Monahan (1999) Adaptive |   40.54|   37.58|  112.46|        |  100.08|  137.94|
|     | QMC                            |    7.95|   13.35|   17.99|        |        |        |
|     | Adaptive QMC                   |   56.88|   49.54|   65.01|   91.12|   59.33|   86.45|
| 4   | GHQ                            |        |        |    2.69|        |        |        |
|     | AGHQ                           |    0.06|        |    0.88|    6.41|   36.83|  227.40|
|     | CDF                            |    0.84|        |    0.48|    0.45|    0.48|    0.41|
|     | Genz & Monahan (1999)          |        |        |        |        |        |        |
|     | Genz & Monahan (1999) Adaptive |        |        |  108.06|  160.24|  176.63|  174.74|
|     | QMC                            |   18.07|        |   45.05|        |        |        |
|     | Adaptive QMC                   |   81.70|   87.35|   58.29|   90.29|   71.82|   53.62|
| 8   | GHQ                            |        |    1.21|    8.64|        |        |        |
|     | AGHQ                           |    0.08|    0.27|    1.46|    9.19|   59.83|  392.17|
|     | CDF                            |    3.42|    2.41|    1.71|    1.61|    1.10|    1.26|
|     | Genz & Monahan (1999)          |        |        |        |        |        |        |
|     | Genz & Monahan (1999) Adaptive |        |        |  157.59|  171.16|  226.70|  300.54|
|     | QMC                            |   40.55|   90.85|  121.96|        |        |        |
|     | Adaptive QMC                   |  116.90|   93.93|   44.17|   48.23|   60.23|   99.19|
| 16  | GHQ                            |        |    4.72|   38.33|        |        |        |
|     | AGHQ                           |    0.12|    0.42|    2.19|   14.39|   90.19|  567.66|
|     | CDF                            |   13.84|   12.15|   10.41|   10.04|    9.37|   11.49|
|     | Genz & Monahan (1999)          |        |        |        |        |        |        |
|     | Genz & Monahan (1999) Adaptive |   10.54|   23.71|  141.58|  250.63|  235.68|  267.39|
|     | QMC                            |   66.08|  270.98|        |        |        |        |
|     | Adaptive QMC                   |   29.06|   43.47|   21.22|   22.91|   25.41|   29.30|
| 32  | GHQ                            |        |        |        |        |        |        |
|     | AGHQ                           |    0.14|    0.61|    3.28|   20.70|  111.31|  632.67|
|     | CDF                            |   69.02|   65.78|   76.68|   63.49|   59.50|   46.21|
|     | Genz & Monahan (1999)          |        |        |        |        |        |        |
|     | Genz & Monahan (1999) Adaptive |    5.77|    5.22|   56.33|  130.43|  111.39|   60.77|
|     | QMC                            |        |        |        |        |        |        |
|     | Adaptive QMC                   |   41.67|    9.18|   16.88|   13.77|   13.40|   11.95|

``` r
show_run_times(na.rm = TRUE)
```

**NAs have been removed. Cells may not be comparable (means)**

| n   | method/p                       |       2|       3|       4|       5|       6|       7|
|:----|:-------------------------------|-------:|-------:|-------:|-------:|-------:|-------:|
| 2   | GHQ                            |    0.04|    0.19|    0.89|        |        |        |
|     | AGHQ                           |    0.04|    0.14|    0.57|    4.45|   24.97|  173.44|
|     | CDF                            |    0.05|    0.06|    0.03|    0.04|    0.05|    0.03|
|     | Genz & Monahan (1999) Adaptive |   40.54|   37.58|  112.46|  100.91|  100.08|  137.94|
|     | QMC                            |    7.95|   13.35|   17.99|        |        |        |
|     | Adaptive QMC                   |   56.88|   49.54|   65.01|   91.12|   59.33|   86.45|
| 4   | GHQ                            |    0.08|    0.38|    2.69|        |        |        |
|     | AGHQ                           |    0.06|    0.18|    0.88|    6.41|   36.83|  227.40|
|     | CDF                            |    0.84|    0.62|    0.48|    0.45|    0.48|    0.41|
|     | Genz & Monahan (1999) Adaptive |   37.22|   46.80|  108.06|  160.24|  176.63|  174.74|
|     | QMC                            |   18.07|   25.02|   45.05|        |        |        |
|     | Adaptive QMC                   |   81.70|   87.35|   58.29|   90.29|   71.82|   53.62|
| 8   | GHQ                            |    0.15|    1.21|    8.64|        |        |        |
|     | AGHQ                           |    0.08|    0.27|    1.46|    9.19|   59.83|  392.17|
|     | CDF                            |    3.42|    2.41|    1.71|    1.61|    1.10|    1.26|
|     | Genz & Monahan (1999) Adaptive |   47.23|   42.26|  157.59|  171.16|  226.70|  300.54|
|     | QMC                            |   40.55|   90.85|  121.96|        |        |        |
|     | Adaptive QMC                   |  116.90|   93.93|   44.17|   48.23|   60.23|   99.19|
| 16  | GHQ                            |    0.45|    4.72|   38.33|        |        |        |
|     | AGHQ                           |    0.12|    0.42|    2.19|   14.39|   90.19|  567.66|
|     | CDF                            |   13.84|   12.15|   10.41|   10.04|    9.37|   11.49|
|     | Genz & Monahan (1999) Adaptive |   10.54|   23.71|  141.58|  250.63|  235.68|  267.39|
|     | QMC                            |   66.08|  270.98|  563.48|        |        |        |
|     | Adaptive QMC                   |   29.06|   43.47|   21.22|   22.91|   25.41|   29.30|
| 32  | AGHQ                           |    0.14|    0.61|    3.28|   20.70|  111.31|  632.67|
|     | CDF                            |   69.02|   65.78|   76.68|   63.49|   59.50|   46.21|
|     | Genz & Monahan (1999) Adaptive |    5.77|    5.22|   56.33|  130.43|  111.39|   60.77|
|     | Adaptive QMC                   |   41.67|    9.18|   16.88|   13.77|   13.40|   11.95|

``` r
show_run_times(TRUE)
```

**Only showing complete cases (means)**

| n   | method/p                       |      2|       3|       4|       5|       6|       7|
|:----|:-------------------------------|------:|-------:|-------:|-------:|-------:|-------:|
| 2   | GHQ                            |   0.04|    0.19|    0.89|        |        |        |
|     | AGHQ                           |   0.04|    0.14|    0.57|    4.40|   24.97|  173.44|
|     | CDF                            |   0.05|    0.06|    0.03|    0.04|    0.05|    0.03|
|     | Genz & Monahan (1999)          |       |        |        |        |        |        |
|     | Genz & Monahan (1999) Adaptive |  40.70|   37.58|  112.46|  100.91|  100.08|  137.94|
|     | QMC                            |   7.97|   13.35|   17.99|        |        |        |
|     | Adaptive QMC                   |  57.15|   49.54|   65.01|   82.61|   59.33|   86.45|
| 4   | GHQ                            |   0.08|    0.38|    2.69|        |        |        |
|     | AGHQ                           |   0.06|    0.18|    0.88|    6.41|   36.83|  227.40|
|     | CDF                            |   0.85|    0.62|    0.48|    0.45|    0.48|    0.41|
|     | Genz & Monahan (1999)          |       |        |        |        |        |        |
|     | Genz & Monahan (1999) Adaptive |  36.81|   46.80|  108.06|  160.24|  176.63|  174.74|
|     | QMC                            |  17.87|   25.02|   45.05|        |        |        |
|     | Adaptive QMC                   |  72.24|   77.11|   58.29|   90.29|   71.82|   53.62|
| 8   | GHQ                            |   0.15|    1.21|    8.64|        |        |        |
|     | AGHQ                           |   0.08|    0.27|    1.46|    9.19|   59.83|  392.17|
|     | CDF                            |   3.34|    2.43|    1.71|    1.61|    1.10|    1.26|
|     | Genz & Monahan (1999)          |       |        |        |        |        |        |
|     | Genz & Monahan (1999) Adaptive |  47.23|   42.26|  157.59|  171.16|  226.70|  300.54|
|     | QMC                            |  40.37|   90.44|  121.96|        |        |        |
|     | Adaptive QMC                   |  98.25|   74.66|   44.17|   48.23|   60.23|   99.19|
| 16  | GHQ                            |   0.45|    4.72|   37.75|        |        |        |
|     | AGHQ                           |   0.12|    0.42|    2.18|   14.39|   90.19|  567.66|
|     | CDF                            |  13.35|   12.15|   10.33|   10.04|    9.37|   11.49|
|     | Genz & Monahan (1999)          |       |        |        |        |        |        |
|     | Genz & Monahan (1999) Adaptive |  12.37|   23.71|  144.81|  250.63|  235.68|  267.39|
|     | QMC                            |  77.14|  270.98|  563.48|        |        |        |
|     | Adaptive QMC                   |  32.64|   43.47|   21.54|   22.91|   25.41|   29.30|
| 32  | GHQ                            |       |        |        |        |        |        |
|     | AGHQ                           |   0.14|    0.61|    3.28|   20.70|  111.31|  632.67|
|     | CDF                            |  69.02|   65.78|   76.68|   63.49|   59.50|   46.21|
|     | Genz & Monahan (1999)          |       |        |        |        |        |        |
|     | Genz & Monahan (1999) Adaptive |   5.77|    5.22|   56.33|  130.43|  111.39|   60.77|
|     | QMC                            |       |        |        |        |        |        |
|     | Adaptive QMC                   |  41.67|    9.18|   16.88|   13.77|   13.40|   11.95|

**Number of complete cases**

|     |    2|    3|    4|    5|    6|    7|
|-----|----:|----:|----:|----:|----:|----:|
| 2   |   99|  100|  100|   99|  100|  100|
| 4   |   98|   99|  100|  100|  100|  100|
| 8   |   99|   99|  100|  100|  100|  100|
| 16  |   81|  100|   97|  100|  100|  100|
| 32  |  100|  100|  100|  100|  100|  100|

``` r

# show medians instead
med_func <- function(x, na.rm)
  apply(x, 1, median, na.rm = na.rm)
show_run_times(meth = med_func, suffix = " (median)", FALSE)
```

**Blank cells have at least one failure (median)**

| n   | method/p                       |      2|      3|      4|      5|       6|       7|
|:----|:-------------------------------|------:|------:|------:|------:|-------:|-------:|
| 2   | GHQ                            |       |   0.20|   0.80|       |        |        |
|     | AGHQ                           |   0.00|   0.20|   0.60|   4.80|   13.70|   86.50|
|     | CDF                            |   0.00|   0.00|   0.00|   0.00|    0.00|    0.00|
|     | Genz & Monahan (1999)          |       |       |       |       |        |        |
|     | Genz & Monahan (1999) Adaptive |   0.90|   4.40|  16.20|       |   22.80|   32.00|
|     | QMC                            |   7.60|  11.90|  15.80|       |        |        |
|     | Adaptive QMC                   |   6.50|  11.50|  13.20|  15.00|   19.00|   21.70|
| 4   | GHQ                            |       |       |   2.20|       |        |        |
|     | AGHQ                           |   0.00|       |   1.00|   7.20|   48.40|  129.10|
|     | CDF                            |   0.70|       |   0.20|   0.20|    0.20|    0.20|
|     | Genz & Monahan (1999)          |       |       |       |       |        |        |
|     | Genz & Monahan (1999) Adaptive |       |       |  24.20|  37.90|   54.40|   46.70|
|     | QMC                            |  12.40|       |  32.40|       |        |        |
|     | Adaptive QMC                   |   8.40|  11.00|  15.00|  18.30|   19.60|   20.60|
| 8   | GHQ                            |       |   1.00|   6.50|       |        |        |
|     | AGHQ                           |   0.00|   0.20|   1.60|   5.80|   35.00|  215.70|
|     | CDF                            |   0.80|   0.70|   0.60|   0.60|    0.60|    0.60|
|     | Genz & Monahan (1999)          |       |       |       |       |        |        |
|     | Genz & Monahan (1999) Adaptive |       |       |  24.90|  51.30|   58.20|   95.80|
|     | QMC                            |  19.00|  46.10|  81.40|       |        |        |
|     | Adaptive QMC                   |   7.50|   8.40|  11.60|  14.00|   20.20|   24.40|
| 16  | GHQ                            |       |   4.20|  25.90|       |        |        |
|     | AGHQ                           |   0.20|   0.40|   1.80|  10.40|   62.20|  382.10|
|     | CDF                            |   8.50|   8.30|   7.90|   8.40|    7.80|    7.60|
|     | Genz & Monahan (1999)          |       |       |       |       |        |        |
|     | Genz & Monahan (1999) Adaptive |   0.20|   1.20|   9.90|  29.80|   22.80|   60.20|
|     | QMC                            |  19.60|  84.60|       |       |        |        |
|     | Adaptive QMC                   |   3.20|   5.90|   8.00|  11.40|   11.40|   19.00|
| 32  | GHQ                            |       |       |       |       |        |        |
|     | AGHQ                           |   0.20|   0.60|   3.20|  19.60|  117.00|  626.30|
|     | CDF                            |  37.90|  37.80|  36.40|  33.10|   35.40|   35.50|
|     | Genz & Monahan (1999)          |       |       |       |       |        |        |
|     | Genz & Monahan (1999) Adaptive |   0.40|   0.50|   0.80|   2.20|    2.10|    8.50|
|     | QMC                            |       |       |       |       |        |        |
|     | Adaptive QMC                   |   2.40|   4.30|   6.60|   6.00|    8.40|    8.10|

``` r
show_run_times(meth = med_func, suffix = " (median)", na.rm = TRUE)
```

**NAs have been removed. Cells may not be comparable (median)**

| n   | method/p                       |      2|      3|       4|      5|       6|       7|
|:----|:-------------------------------|------:|------:|-------:|------:|-------:|-------:|
| 2   | GHQ                            |   0.00|   0.20|    0.80|       |        |        |
|     | AGHQ                           |   0.00|   0.20|    0.60|   4.80|   13.70|   86.50|
|     | CDF                            |   0.00|   0.00|    0.00|   0.00|    0.00|    0.00|
|     | Genz & Monahan (1999) Adaptive |   0.90|   4.40|   16.20|  11.60|   22.80|   32.00|
|     | QMC                            |   7.60|  11.90|   15.80|       |        |        |
|     | Adaptive QMC                   |   6.50|  11.50|   13.20|  15.00|   19.00|   21.70|
| 4   | GHQ                            |   0.00|   0.40|    2.20|       |        |        |
|     | AGHQ                           |   0.00|   0.20|    1.00|   7.20|   48.40|  129.10|
|     | CDF                            |   0.70|   0.40|    0.20|   0.20|    0.20|    0.20|
|     | Genz & Monahan (1999) Adaptive |   2.60|   6.60|   24.20|  37.90|   54.40|   46.70|
|     | QMC                            |  12.40|  19.00|   32.40|       |        |        |
|     | Adaptive QMC                   |   8.40|  11.00|   15.00|  18.30|   19.60|   20.60|
| 8   | GHQ                            |   0.20|   1.00|    6.50|       |        |        |
|     | AGHQ                           |   0.00|   0.20|    1.60|   5.80|   35.00|  215.70|
|     | CDF                            |   0.80|   0.70|    0.60|   0.60|    0.60|    0.60|
|     | Genz & Monahan (1999) Adaptive |   1.20|   5.20|   24.90|  51.30|   58.20|   95.80|
|     | QMC                            |  19.00|  46.10|   81.40|       |        |        |
|     | Adaptive QMC                   |   7.50|   8.40|   11.60|  14.00|   20.20|   24.40|
| 16  | GHQ                            |   0.40|   4.20|   25.90|       |        |        |
|     | AGHQ                           |   0.20|   0.40|    1.80|  10.40|   62.20|  382.10|
|     | CDF                            |   8.50|   8.30|    7.90|   8.40|    7.80|    7.60|
|     | Genz & Monahan (1999) Adaptive |   0.20|   1.20|    9.90|  29.80|   22.80|   60.20|
|     | QMC                            |  19.60|  84.60|  204.40|       |        |        |
|     | Adaptive QMC                   |   3.20|   5.90|    8.00|  11.40|   11.40|   19.00|
| 32  | AGHQ                           |   0.20|   0.60|    3.20|  19.60|  117.00|  626.30|
|     | CDF                            |  37.90|  37.80|   36.40|  33.10|   35.40|   35.50|
|     | Genz & Monahan (1999) Adaptive |   0.40|   0.50|    0.80|   2.20|    2.10|    8.50|
|     | Adaptive QMC                   |   2.40|   4.30|    6.60|   6.00|    8.40|    8.10|

``` r
show_run_times(meth = med_func, suffix = " (median)", TRUE)
```

**Only showing complete cases (median)**

| n   | method/p                       |      2|      3|       4|      5|       6|       7|
|:----|:-------------------------------|------:|------:|-------:|------:|-------:|-------:|
| 2   | GHQ                            |   0.00|   0.20|    0.80|       |        |        |
|     | AGHQ                           |   0.00|   0.20|    0.60|   4.80|   13.70|   86.50|
|     | CDF                            |   0.00|   0.00|    0.00|   0.00|    0.00|    0.00|
|     | Genz & Monahan (1999)          |       |       |        |       |        |        |
|     | Genz & Monahan (1999) Adaptive |   0.80|   4.40|   16.20|  11.60|   22.80|   32.00|
|     | QMC                            |   7.60|  11.90|   15.80|       |        |        |
|     | Adaptive QMC                   |   5.80|  11.50|   13.20|  14.60|   19.00|   21.70|
| 4   | GHQ                            |   0.00|   0.40|    2.20|       |        |        |
|     | AGHQ                           |   0.00|   0.20|    1.00|   7.20|   48.40|  129.10|
|     | CDF                            |   0.80|   0.40|    0.20|   0.20|    0.20|    0.20|
|     | Genz & Monahan (1999)          |       |       |        |       |        |        |
|     | Genz & Monahan (1999) Adaptive |   2.50|   6.60|   24.20|  37.90|   54.40|   46.70|
|     | QMC                            |  12.20|  19.00|   32.40|       |        |        |
|     | Adaptive QMC                   |   7.40|  10.80|   15.00|  18.30|   19.60|   20.60|
| 8   | GHQ                            |   0.20|   1.00|    6.50|       |        |        |
|     | AGHQ                           |   0.00|   0.20|    1.60|   5.80|   35.00|  215.70|
|     | CDF                            |   0.60|   0.80|    0.60|   0.60|    0.60|    0.60|
|     | Genz & Monahan (1999)          |       |       |        |       |        |        |
|     | Genz & Monahan (1999) Adaptive |   1.20|   5.20|   24.90|  51.30|   58.20|   95.80|
|     | QMC                            |  18.40|  45.20|   81.40|       |        |        |
|     | Adaptive QMC                   |   7.20|   8.20|   11.60|  14.00|   20.20|   24.40|
| 16  | GHQ                            |   0.40|   4.20|   25.80|       |        |        |
|     | AGHQ                           |   0.20|   0.40|    1.80|  10.40|   62.20|  382.10|
|     | CDF                            |   8.00|   8.30|    8.00|   8.40|    7.80|    7.60|
|     | Genz & Monahan (1999)          |       |       |        |       |        |        |
|     | Genz & Monahan (1999) Adaptive |   0.40|   1.20|    9.60|  29.80|   22.80|   60.20|
|     | QMC                            |  20.20|  84.60|  204.40|       |        |        |
|     | Adaptive QMC                   |   3.80|   5.90|    7.80|  11.40|   11.40|   19.00|
| 32  | GHQ                            |       |       |        |       |        |        |
|     | AGHQ                           |   0.20|   0.60|    3.20|  19.60|  117.00|  626.30|
|     | CDF                            |  37.90|  37.80|   36.40|  33.10|   35.40|   35.50|
|     | Genz & Monahan (1999)          |       |       |        |       |        |        |
|     | Genz & Monahan (1999) Adaptive |   0.40|   0.50|    0.80|   2.20|    2.10|    8.50|
|     | QMC                            |       |       |        |       |        |        |
|     | Adaptive QMC                   |   2.40|   4.30|    6.60|   6.00|    8.40|    8.10|

**Number of complete cases**

|     |    2|    3|    4|    5|    6|    7|
|-----|----:|----:|----:|----:|----:|----:|
| 2   |   99|  100|  100|   99|  100|  100|
| 4   |   98|   99|  100|  100|  100|  100|
| 8   |   99|   99|  100|  100|  100|  100|
| 16  |   81|  100|   97|  100|  100|  100|
| 32  |  100|  100|  100|  100|  100|  100|

``` r

# show quantiles instead
med_func <- function(x, prob = .75, ...)
  apply(x, 1, function(z) quantile(na.omit(z), probs = prob))
show_run_times(meth = med_func, suffix = " (75% quantile)", na.rm = TRUE)
```

**NAs have been removed. Cells may not be comparable (75% quantile)**

| n   | method/p                       |      2|       3|       4|       5|       6|       7|
|:----|:-------------------------------|------:|-------:|-------:|-------:|-------:|-------:|
| 2   | GHQ                            |   0.00|    0.20|    1.00|        |        |        |
|     | AGHQ                           |   0.00|    0.20|    0.60|    5.00|   33.60|  254.10|
|     | CDF                            |   0.00|    0.20|    0.00|    0.00|    0.00|    0.00|
|     | Genz & Monahan (1999) Adaptive |  13.85|   17.45|  107.35|   80.80|  137.90|  161.10|
|     | QMC                            |  10.50|   16.70|   22.70|        |        |        |
|     | Adaptive QMC                   |  21.80|   36.40|   38.00|   51.15|   45.50|   61.45|
| 4   | GHQ                            |   0.20|    0.50|    2.40|        |        |        |
|     | AGHQ                           |   0.20|    0.20|    1.00|    7.60|   51.00|  378.45|
|     | CDF                            |   1.20|    1.00|    0.80|    0.60|    0.60|    0.45|
|     | Genz & Monahan (1999) Adaptive |  11.00|   25.60|   89.15|  143.50|  147.85|  167.50|
|     | QMC                            |  20.70|   27.20|   46.90|        |        |        |
|     | Adaptive QMC                   |  34.80|   45.55|   35.65|   46.20|   57.95|   41.05|
| 8   | GHQ                            |   0.20|    1.40|   10.95|        |        |        |
|     | AGHQ                           |   0.20|    0.40|    1.80|   12.45|   86.40|  626.75|
|     | CDF                            |   4.65|    2.85|    2.40|    1.45|    0.80|    0.80|
|     | Genz & Monahan (1999) Adaptive |   5.80|   18.00|  143.35|  138.05|  230.40|  270.05|
|     | QMC                            |  39.50|  105.15|  129.65|        |        |        |
|     | Adaptive QMC                   |  25.40|   24.85|   33.60|   31.70|   35.05|   50.75|
| 16  | GHQ                            |   0.60|    6.20|   47.95|        |        |        |
|     | AGHQ                           |   0.20|    0.45|    3.20|   22.45|  156.00|  996.05|
|     | CDF                            |  13.45|   13.10|   10.80|   10.40|   10.15|    9.05|
|     | Genz & Monahan (1999) Adaptive |   2.20|    8.05|   51.70|  161.10|  141.35|  188.00|
|     | QMC                            |  42.40|  207.75|  527.60|        |        |        |
|     | Adaptive QMC                   |   9.95|   11.65|   18.35|   21.85|   26.05|   32.85|
| 32  | AGHQ                           |   0.20|    0.80|    3.40|   20.40|  119.05|  643.60|
|     | CDF                            |  89.05|   71.10|   79.20|   59.45|   64.65|   50.95|
|     | Genz & Monahan (1999) Adaptive |   0.60|    3.60|   25.75|   18.95|   18.05|   64.65|
|     | Adaptive QMC                   |   5.45|    8.20|   13.40|   13.15|   14.20|   15.70|

``` r
show_run_times(meth = med_func, suffix = " (75% quantile)", TRUE)
```

**Only showing complete cases (75% quantile)**

| n   | method/p                       |      2|       3|       4|       5|       6|       7|
|:----|:-------------------------------|------:|-------:|-------:|-------:|-------:|-------:|
| 2   | GHQ                            |   0.00|    0.20|    1.00|        |        |        |
|     | AGHQ                           |   0.00|    0.20|    0.60|    5.00|   33.60|  254.10|
|     | CDF                            |   0.00|    0.20|    0.00|    0.00|    0.00|    0.00|
|     | Genz & Monahan (1999)          |       |        |        |        |        |        |
|     | Genz & Monahan (1999) Adaptive |  13.70|   17.45|  107.35|   80.80|  137.90|  161.10|
|     | QMC                            |  10.60|   16.70|   22.70|        |        |        |
|     | Adaptive QMC                   |  20.20|   36.40|   38.00|   49.00|   45.50|   61.45|
| 4   | GHQ                            |   0.20|    0.50|    2.40|        |        |        |
|     | AGHQ                           |   0.20|    0.20|    1.00|    7.60|   51.00|  378.45|
|     | CDF                            |   1.20|    1.00|    0.80|    0.60|    0.60|    0.45|
|     | Genz & Monahan (1999)          |       |        |        |        |        |        |
|     | Genz & Monahan (1999) Adaptive |  10.60|   25.60|   89.15|  143.50|  147.85|  167.50|
|     | QMC                            |  20.15|   27.20|   46.90|        |        |        |
|     | Adaptive QMC                   |  33.00|   40.10|   35.65|   46.20|   57.95|   41.05|
| 8   | GHQ                            |   0.20|    1.40|   10.95|        |        |        |
|     | AGHQ                           |   0.20|    0.40|    1.80|   12.45|   86.40|  626.75|
|     | CDF                            |   4.60|    2.90|    2.40|    1.45|    0.80|    0.80|
|     | Genz & Monahan (1999)          |       |        |        |        |        |        |
|     | Genz & Monahan (1999) Adaptive |   5.80|   18.00|  143.35|  138.05|  230.40|  270.05|
|     | QMC                            |  38.90|   98.60|  129.65|        |        |        |
|     | Adaptive QMC                   |  24.00|   23.40|   33.60|   31.70|   35.05|   50.75|
| 16  | GHQ                            |   0.60|    6.20|   47.80|        |        |        |
|     | AGHQ                           |   0.20|    0.45|    3.20|   22.45|  156.00|  996.05|
|     | CDF                            |  11.60|   13.10|   10.80|   10.40|   10.15|    9.05|
|     | Genz & Monahan (1999)          |       |        |        |        |        |        |
|     | Genz & Monahan (1999) Adaptive |   3.60|    8.05|   51.40|  161.10|  141.35|  188.00|
|     | QMC                            |  59.40|  207.75|  527.60|        |        |        |
|     | Adaptive QMC                   |  12.80|   11.65|   19.40|   21.85|   26.05|   32.85|
| 32  | GHQ                            |       |        |        |        |        |        |
|     | AGHQ                           |   0.20|    0.80|    3.40|   20.40|  119.05|  643.60|
|     | CDF                            |  89.05|   71.10|   79.20|   59.45|   64.65|   50.95|
|     | Genz & Monahan (1999)          |       |        |        |        |        |        |
|     | Genz & Monahan (1999) Adaptive |   0.60|    3.60|   25.75|   18.95|   18.05|   64.65|
|     | QMC                            |       |        |        |        |        |        |
|     | Adaptive QMC                   |   5.45|    8.20|   13.40|   13.15|   14.20|   15.70|

**Number of complete cases**

|     |    2|    3|    4|    5|    6|    7|
|-----|----:|----:|----:|----:|----:|----:|
| 2   |   99|  100|  100|   99|  100|  100|
| 4   |   98|   99|  100|  100|  100|  100|
| 8   |   99|   99|  100|  100|  100|  100|
| 16  |   81|  100|   97|  100|  100|  100|
| 32  |  100|  100|  100|  100|  100|  100|

``` r

#####
# mean scaled RMSE table
show_scaled_mean_rmse <- function(remove_nas = FALSE, na.rm = FALSE){
  # get mean scaled RMSE for the methods and the configurations pairs
  res <- sapply(ex_output, function(x)
    sapply(x[!names(x) %in% c("n", "p")], `[[`, "vals_n_comp_time", 
           simplify = "array"), 
    simplify = "array")
  err <- res["rel_rmse", , , ]
  
  is_complete <- apply(err, 3, function(x){
    if(remove_nas){
      consider <- !apply(is.na(x), 1L, all)
      apply(!is.na(x[consider, , drop = FALSE]), 2, all)
    } else 
      rep(TRUE, NCOL(x))
  })
  dim(is_complete) <- dim(err)[2:3]
  
  err <- lapply(1:dim(err)[3], function(i){
    x <- err[, , i]
    x[, is_complete[, i]]
  })
  
  err <- sapply(err, rowMeans, na.rm = na.rm) * err_mult
  err[is.nan(err)] <- NA_real_
  err <- err[!apply(err, 1, function(x) all(is.na(x))), ]
  
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

| n   | method/p                       |     2|     3|     4|      5|      6|      7|
|:----|:-------------------------------|-----:|-----:|-----:|------:|------:|------:|
| 2   | GHQ                            |      |  4.63|  4.10|       |       |       |
|     | AGHQ                           |  3.71|  3.26|  3.48|   3.34|   3.26|   3.50|
|     | CDF                            |  0.63|  0.68|  0.60|   0.56|   0.65|   0.60|
|     | Genz & Monahan (1999) Adaptive |  6.90|  6.68|  7.70|       |  16.29|   8.38|
|     | QMC                            |  7.57|  6.68|  7.86|       |       |       |
|     | Adaptive QMC                   |  7.32|  7.75|  7.64|   7.29|   8.09|   7.30|
| 4   | GHQ                            |      |      |  5.07|       |       |       |
|     | AGHQ                           |  3.56|      |  3.52|   3.15|   3.73|   3.15|
|     | CDF                            |  7.65|      |  7.58|   9.02|   7.40|   7.69|
|     | Genz & Monahan (1999) Adaptive |      |      |  6.89|  10.37|   8.28|  11.37|
|     | QMC                            |  7.64|      |  7.43|       |       |       |
|     | Adaptive QMC                   |  8.21|  8.11|  7.02|   7.01|   8.04|   7.22|
| 8   | GHQ                            |      |  5.86|  5.26|       |       |       |
|     | AGHQ                           |  3.29|  3.42|  3.12|   3.40|   3.59|   3.59|
|     | CDF                            |  7.67|  7.78|  7.84|   8.38|   7.69|   7.10|
|     | Genz & Monahan (1999) Adaptive |      |      |  8.30|   8.14|   8.56|  13.89|
|     | QMC                            |  7.38|  7.81|  8.08|       |       |       |
|     | Adaptive QMC                   |  7.04|  8.22|  7.22|   7.39|   7.21|   7.48|
| 16  | GHQ                            |      |  6.53|  5.82|       |       |       |
|     | AGHQ                           |  3.96|  3.29|  3.31|   3.58|   3.05|   3.75|
|     | CDF                            |  6.99|  8.59|  6.99|   6.93|   7.51|   7.54|
|     | Genz & Monahan (1999) Adaptive |  5.51|  7.63|  8.37|   9.00|   7.85|   8.55|
|     | QMC                            |  7.41|  7.98|      |       |       |       |
|     | Adaptive QMC                   |  7.26|  7.54|  7.80|   7.21|   7.62|   7.25|
| 32  | GHQ                            |      |      |      |       |       |       |
|     | AGHQ                           |  4.72|  4.44|  4.40|   3.51|   3.85|   3.42|
|     | CDF                            |  7.41|  7.18|  7.44|   7.78|   8.30|   7.43|
|     | Genz & Monahan (1999) Adaptive |  5.27|  6.39|  7.93|   8.02|   8.10|   8.99|
|     | QMC                            |      |      |      |       |       |       |
|     | Adaptive QMC                   |  7.62|  7.84|  7.00|   7.69|   8.12|   7.49|

``` r
show_scaled_mean_rmse(na.rm = TRUE)
```

**NAs have been removed. Cells may not be comparable**

| n   | method/p                       |     2|     3|     4|      5|      6|      7|
|:----|:-------------------------------|-----:|-----:|-----:|------:|------:|------:|
| 2   | GHQ                            |  4.35|  4.63|  4.10|       |       |       |
|     | AGHQ                           |  3.71|  3.26|  3.48|   3.34|   3.26|   3.50|
|     | CDF                            |  0.63|  0.68|  0.60|   0.56|   0.65|   0.60|
|     | Genz & Monahan (1999) Adaptive |  6.90|  6.68|  7.70|  10.43|  16.29|   8.38|
|     | QMC                            |  7.57|  6.68|  7.86|       |       |       |
|     | Adaptive QMC                   |  7.32|  7.75|  7.64|   7.29|   8.09|   7.30|
| 4   | GHQ                            |  5.88|  5.53|  5.07|       |       |       |
|     | AGHQ                           |  3.56|  3.60|  3.52|   3.15|   3.73|   3.15|
|     | CDF                            |  7.65|  7.43|  7.58|   9.02|   7.40|   7.69|
|     | Genz & Monahan (1999) Adaptive |  6.44|  6.85|  6.89|  10.37|   8.28|  11.37|
|     | QMC                            |  7.64|  7.78|  7.43|       |       |       |
|     | Adaptive QMC                   |  8.21|  8.11|  7.02|   7.01|   8.04|   7.22|
| 8   | GHQ                            |  6.15|  5.86|  5.26|       |       |       |
|     | AGHQ                           |  3.29|  3.42|  3.12|   3.40|   3.59|   3.59|
|     | CDF                            |  7.67|  7.78|  7.84|   8.38|   7.69|   7.10|
|     | Genz & Monahan (1999) Adaptive |  6.24|  6.49|  8.30|   8.14|   8.56|  13.89|
|     | QMC                            |  7.38|  7.81|  8.08|       |       |       |
|     | Adaptive QMC                   |  7.04|  8.22|  7.22|   7.39|   7.21|   7.48|
| 16  | GHQ                            |  7.03|  6.53|  5.82|       |       |       |
|     | AGHQ                           |  3.96|  3.29|  3.31|   3.58|   3.05|   3.75|
|     | CDF                            |  6.99|  8.59|  6.99|   6.93|   7.51|   7.54|
|     | Genz & Monahan (1999) Adaptive |  5.51|  7.63|  8.37|   9.00|   7.85|   8.55|
|     | QMC                            |  7.41|  7.98|  8.26|       |       |       |
|     | Adaptive QMC                   |  7.26|  7.54|  7.80|   7.21|   7.62|   7.25|
| 32  | AGHQ                           |  4.72|  4.44|  4.40|   3.51|   3.85|   3.42|
|     | CDF                            |  7.41|  7.18|  7.44|   7.78|   8.30|   7.43|
|     | Genz & Monahan (1999) Adaptive |  5.27|  6.39|  7.93|   8.02|   8.10|   8.99|
|     | Adaptive QMC                   |  7.62|  7.84|  7.00|   7.69|   8.12|   7.49|

``` r
show_scaled_mean_rmse(TRUE)
```

**Only showing complete cases**

| n   | method/p                       |     2|     3|     4|      5|      6|      7|
|:----|:-------------------------------|-----:|-----:|-----:|------:|------:|------:|
| 2   | GHQ                            |  4.35|  4.63|  4.10|       |       |       |
|     | AGHQ                           |  3.69|  3.26|  3.48|   3.33|   3.26|   3.50|
|     | CDF                            |  0.63|  0.68|  0.60|   0.57|   0.65|   0.60|
|     | Genz & Monahan (1999) Adaptive |  6.94|  6.68|  7.70|  10.43|  16.29|   8.38|
|     | QMC                            |  7.57|  6.68|  7.86|       |       |       |
|     | Adaptive QMC                   |  7.34|  7.75|  7.64|   7.26|   8.09|   7.30|
| 4   | GHQ                            |  5.84|  5.53|  5.07|       |       |       |
|     | AGHQ                           |  3.57|  3.60|  3.52|   3.15|   3.73|   3.15|
|     | CDF                            |  7.61|  7.43|  7.58|   9.02|   7.40|   7.69|
|     | Genz & Monahan (1999) Adaptive |  6.45|  6.85|  6.89|  10.37|   8.28|  11.37|
|     | QMC                            |  7.61|  7.78|  7.43|       |       |       |
|     | Adaptive QMC                   |  8.07|  8.05|  7.02|   7.01|   8.04|   7.22|
| 8   | GHQ                            |  6.15|  5.85|  5.26|       |       |       |
|     | AGHQ                           |  3.24|  3.37|  3.12|   3.40|   3.59|   3.59|
|     | CDF                            |  7.64|  7.79|  7.84|   8.38|   7.69|   7.10|
|     | Genz & Monahan (1999) Adaptive |  6.24|  6.49|  8.30|   8.14|   8.56|  13.89|
|     | QMC                            |  7.33|  7.84|  8.08|       |       |       |
|     | Adaptive QMC                   |  6.90|  8.19|  7.22|   7.39|   7.21|   7.48|
| 16  | GHQ                            |  7.03|  6.53|  5.84|       |       |       |
|     | AGHQ                           |  4.06|  3.29|  3.33|   3.58|   3.05|   3.75|
|     | CDF                            |  6.95|  8.59|  7.02|   6.93|   7.51|   7.54|
|     | Genz & Monahan (1999) Adaptive |  5.75|  7.63|  8.34|   9.00|   7.85|   8.55|
|     | QMC                            |  7.60|  7.98|  8.26|       |       |       |
|     | Adaptive QMC                   |  7.04|  7.54|  7.83|   7.21|   7.62|   7.25|
| 32  | GHQ                            |      |      |      |       |       |       |
|     | AGHQ                           |  4.72|  4.44|  4.40|   3.51|   3.85|   3.42|
|     | CDF                            |  7.41|  7.18|  7.44|   7.78|   8.30|   7.43|
|     | Genz & Monahan (1999) Adaptive |  5.27|  6.39|  7.93|   8.02|   8.10|   8.99|
|     | QMC                            |      |      |      |       |       |       |
|     | Adaptive QMC                   |  7.62|  7.84|  7.00|   7.69|   8.12|   7.49|

**Number of complete cases**

|     |    2|    3|    4|    5|    6|    7|
|-----|----:|----:|----:|----:|----:|----:|
| 2   |   99|  100|  100|   99|  100|  100|
| 4   |   98|   99|  100|  100|  100|  100|
| 8   |   99|   99|  100|  100|  100|  100|
| 16  |   81|  100|   97|  100|  100|  100|
| 32  |  100|  100|  100|  100|  100|  100|

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
| 2   | 0%         |   6.00|   5.00|   6.00|     |     |     |
|     | 25%        |   8.00|   7.00|   7.00|     |     |     |
|     | 50%        |   9.00|   8.00|   8.00|     |     |     |
|     | 75%        |  10.00|   9.00|   8.00|     |     |     |
|     | 100%       |  17.00|  16.00|  13.00|     |     |     |
| 4   | 0%         |   8.00|   7.00|   6.00|     |     |     |
|     | 25%        |   9.00|   8.00|   8.00|     |     |     |
|     | 50%        |  11.00|   9.00|   9.00|     |     |     |
|     | 75%        |  12.00|  11.00|   9.00|     |     |     |
|     | 100%       |  18.00|  14.00|  17.00|     |     |     |
| 8   | 0%         |   8.00|   6.00|   7.00|     |     |     |
|     | 25%        |  11.50|  10.00|   9.00|     |     |     |
|     | 50%        |  13.00|  11.00|  10.00|     |     |     |
|     | 75%        |  16.00|  13.00|  11.25|     |     |     |
|     | 100%       |  25.00|  20.00|  14.00|     |     |     |
| 16  | 0%         |  10.00|   9.00|   9.00|     |     |     |
|     | 25%        |  14.00|  13.00|  11.00|     |     |     |
|     | 50%        |  17.00|  15.00|  12.00|     |     |     |
|     | 75%        |  21.00|  17.00|  14.00|     |     |     |
|     | 100%       |  25.00|  22.00|  19.00|     |     |     |

**Number of complete cases**

|     |    2|    3|    4|    5|    6|    7|
|-----|----:|----:|----:|----:|----:|----:|
| 2   |   99|  100|  100|    0|    0|    0|
| 4   |   99|   99|  100|    0|    0|    0|
| 8   |   99|  100|  100|    0|    0|    0|
| 16  |   81|  100|  100|    0|    0|    0|
| 32  |    0|    0|    0|    0|    0|    0|

``` r
show_n_nodes(TRUE)
```

**Only showing complete cases (Adaptive GHQ)**

| n   | quantile/p |      2|     3|     4|     5|     6|     7|
|:----|:-----------|------:|-----:|-----:|-----:|-----:|-----:|
| 2   | 0%         |   4.00|  5.00|  4.00|  4.00|  5.00|  5.00|
|     | 25%        |   6.00|  6.00|  6.00|  6.00|  6.00|  6.00|
|     | 50%        |   7.00|  7.00|  7.00|  7.00|  6.00|  6.00|
|     | 75%        |   7.00|  7.00|  7.00|  7.00|  7.00|  7.00|
|     | 100%       |  11.00|  9.00|  9.00|  9.00|  8.00|  9.00|
| 4   | 0%         |   6.00|  5.00|  5.00|  5.00|  6.00|  5.00|
|     | 25%        |   6.00|  6.00|  6.00|  6.00|  6.00|  6.00|
|     | 50%        |   7.00|  7.00|  7.00|  7.00|  7.00|  6.00|
|     | 75%        |   7.00|  7.00|  7.00|  7.00|  7.00|  7.00|
|     | 100%       |  11.00|  9.00|  9.00|  9.00|  8.00|  7.00|
| 8   | 0%         |   4.00|  4.00|  5.00|  6.00|  6.00|  6.00|
|     | 25%        |   6.00|  6.00|  6.00|  6.00|  6.00|  6.00|
|     | 50%        |   7.00|  7.00|  7.00|  6.00|  6.00|  6.00|
|     | 75%        |   7.00|  7.00|  7.00|  7.00|  7.00|  7.00|
|     | 100%       |  12.00|  9.00|  9.00|  8.00|  8.00|  8.00|
| 16  | 0%         |   4.00|  4.00|  4.00|  5.00|  5.00|  6.00|
|     | 25%        |   6.00|  6.00|  6.00|  6.00|  6.00|  6.00|
|     | 50%        |   6.00|  6.00|  6.00|  6.00|  6.00|  6.00|
|     | 75%        |   7.00|  7.00|  7.00|  7.00|  7.00|  7.00|
|     | 100%       |   9.00|  9.00|  7.00|  7.00|  7.00|  7.00|
| 32  | 0%         |   4.00|  4.00|  4.00|  4.00|  4.00|  5.00|
|     | 25%        |   4.00|  5.00|  6.00|  6.00|  6.00|  6.00|
|     | 50%        |   6.00|  6.00|  6.00|  6.00|  6.00|  6.00|
|     | 75%        |   6.00|  6.00|  6.00|  6.00|  6.00|  6.00|
|     | 100%       |   8.00|  7.00|  7.00|  7.00|  7.00|  7.00|

**Number of complete cases**

|     |    2|    3|    4|    5|    6|    7|
|-----|----:|----:|----:|----:|----:|----:|
| 2   |  100|  100|  100|  100|  100|  100|
| 4   |  100|   99|  100|  100|  100|  100|
| 8   |  100|  100|  100|  100|  100|  100|
| 16  |  100|  100|  100|  100|  100|  100|
| 32  |  100|  100|  100|  100|  100|  100|

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
#>         expr    min     lq  mean median    uq  max neval
#>    mixprobit  8.117  8.922 12.69  11.20 12.04 1918  1000
#>  randtoolbox 53.285 56.096 60.07  57.24 58.43 1727  1000

# w/ larger dim
dim <- 50L
all.equal(get_sobol_seq(dim)(n), t(sobol(n = n, dim = dim)))
#> [1] TRUE
microbenchmark::microbenchmark(
  mixprobit = get_sobol_seq(dim)(n), 
  randtoolbox = sobol(n = n, dim = dim), times = 1000)
#> Unit: microseconds
#>         expr   min    lq  mean median    uq    max neval
#>    mixprobit 16.54 18.53 24.01  21.09 23.13 2500.1  1000
#>  randtoolbox 70.93 75.83 81.08  78.56 81.45  312.8  1000

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
#>                        expr     min      lq    mean  median      uq    max
#>       mixprobit   (1 point)   3.361   4.127   5.673   5.453   6.059  133.5
#>       randtoolbox (1 point)  36.993  39.022  42.538  40.361  42.034  158.5
#>    mixprobit   (100 points)   5.365   6.446   7.849   7.726   8.324  108.7
#>    randtoolbox (100 points)  41.091  43.413  46.899  44.775  46.567  227.9
#>  mixprobit   (10000 points) 182.428 190.216 205.958 191.906 194.029 2062.8
#>  randtoolbox (10000 points) 362.201 376.045 449.781 380.796 388.218 3872.2
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
#>         expr   min    lq  mean median    uq  max neval
#>    mixprobit 280.3 282.5 293.8  287.0 289.5 2140  1000
#>  randtoolbox 332.6 338.8 351.4  343.1 349.0  800  1000

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
#>                        expr     min       lq     mean   median       uq     max
#>       mixprobit   (1 point)   3.505    4.576    8.039    6.311    8.614   102.9
#>       randtoolbox (1 point)  38.343   42.635   52.677   46.364   53.785   348.2
#>    mixprobit   (100 points)   6.908    8.837   13.638   10.535   12.912  1379.6
#>    randtoolbox (100 points)  48.353   54.346   68.952   58.519   67.296  2696.9
#>  mixprobit   (10000 points) 306.560  366.503  452.842  382.176  426.077  3871.0
#>  randtoolbox (10000 points) 949.554 1008.468 1369.910 1048.161 1229.039 36978.7
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

Mixed Models with Multinomial Outcomes
--------------------------------------

A related model is with multinomial outcomes (TODO: write more about the
model).

We have also made an implementation for this model. We perform a similar
quick example as before below. We start by assigning functions to
approximate the marginal likelihood. Then we assign a function to draw a
covariance matrix, the random effects, the fixed offsets, and the
outcomes.

``` r
#####
# assign approximation functions
aprx <- within(list(), {
  get_GHQ_cpp <- function(eta, Z, p, Sigma, b, is_adaptive = FALSE){
    mixprobit:::set_GH_rule_cached(b)
    function()
      mixprobit:::aprx_mult_mix_ghq(eta = eta, n_alt = p, Z = Z, 
                                    Sigma = Sigma, b = b, 
                                    is_adaptive = is_adaptive)
  }
  get_AGHQ_cpp <- get_GHQ_cpp
  formals(get_AGHQ_cpp)$is_adaptive <- TRUE
  
  get_cdf_cpp <- function(eta, Z, p, Sigma, maxpts, abseps = -1, 
                          releps = 1e-4)
    function()
      mixprobit:::aprx_mult_mix_cdf(
        n_alt = p, eta = eta, Z = Z, Sigma = Sigma, maxpts = maxpts,
        abseps = abseps, releps = releps)
  
  get_sim_mth <- function(eta, Z, p, Sigma, maxpts, abseps = -1, 
                          releps = 1e-4, is_adaptive = FALSE)
    # Args: 
    #   key: integer which determines degree of integration rule.
    function(key)
      mixprobit:::aprx_mult_mix(
        eta = eta, n_alt = p, Z = Z, Sigma = Sigma, maxpts = maxpts, 
        key = key, abseps = abseps, releps = releps, 
        is_adaptive = is_adaptive)
  get_Asim_mth <- get_sim_mth
  formals(get_Asim_mth)$is_adaptive <- TRUE
  
  get_qmc <- function(eta, Z, p, Sigma, maxpts, is_adaptive = FALSE, 
                      releps = 1e-4, n_seqs = 10L, abseps)
    function(){
      seeds <- sample.int(2147483646L, n_seqs)
      mixprobit:::aprx_mult_mix_qmc(
        eta = eta, n_alt = p, Z = Z, Sigma = Sigma, n_max = maxpts, 
        is_adaptive = is_adaptive, seeds = seeds, releps = releps)
    }
  get_Aqmc <- get_qmc
  formals(get_Aqmc)$is_adaptive <- TRUE
})
```

``` r
#####
# returns a simulated data set from one cluster in a mixed multinomial 
# model.
# 
# Args:
#   n: cluster size.
#   p: number of random effects and number of categories.
get_sim_dat <- function(n, p, Sigma){
  has_Sigma <- !missing(Sigma)
  out <- list(n = n, p = p)
  within(out, {
    Z <- diag(p)
    # covariance matrix of random effects
    if(!has_Sigma)
      Sigma <- drop(                     
        rWishart(1, 5 * p, diag(1 / p / 5, p)))
    S_chol <- chol(Sigma)
    # random effects
    u <- drop(rnorm(p) %*% S_chol)       
    
    dat <- replicate(n, {
      eta <- rnorm(p)
      lp <- drop(eta + Z %*% u)
      A <- rnorm(p, mean = lp)
      y <- which.max(A)
      Z <- t(Z[y, ] - Z[-y, , drop = FALSE])
      eta <- eta[y] - eta[-y]
      list(eta = eta, Z = Z, y = y)
    }, simplify = FALSE)
    
    y <- sapply(dat, `[[`, "y")
    eta <- do.call(c, lapply(dat, `[[`, "eta"))
    Z <- do.call(cbind, lapply(dat, `[[`, "Z"))
    rm(dat, S_chol)
  })
}

# example of one data set
set.seed(1L)
get_sim_dat(n = 3L, p = 3L)
#> $n
#> [1] 3
#> 
#> $p
#> [1] 3
#> 
#> $eta
#> [1] -1.9113 -0.3486  0.1835  1.3276  0.6709  0.8613
#> 
#> $y
#> [1] 3 3 2
#> 
#> $u
#> [1] -0.2510 -0.1036  2.4514
#> 
#> $Sigma
#>         [,1]    [,2]    [,3]
#> [1,]  0.7254  0.2798 -0.3387
#> [2,]  0.2798  1.4856 -0.4120
#> [3,] -0.3387 -0.4120  1.1567
#> 
#> $Z
#>      [,1] [,2] [,3] [,4] [,5] [,6]
#> [1,]   -1    0   -1    0   -1    1
#> [2,]    1   -1    1   -1    0    0
#> [3,]    0    1    0    1    1   -1
```

Here is a quick example where we compare the approximation methods on
one data set.

``` r
#####
# parameters to change
n <- 10L              # cluster size
p <- 4L               # number of random effects and categories
b <- 10L              # number of nodes to use with GHQ
maxpts <- p * 10000L  # factor to set the (maximum) number of
                      # evaluations of the integrand with
                      # the other methods

#####
# variables used in simulation
set.seed(1)
dat <- get_sim_dat(n = n, p = p)

# shorter than calling `with(dat, ...)`
wd <- function(expr)
  eval(bquote(with(dat, .(substitute(expr)))), parent.frame())

#####
# get the functions to use
GHQ_cpp  <- wd(aprx$get_GHQ_cpp (eta = eta, Z = Z, p = p - 1L, 
                                 Sigma = Sigma, b = b))
AGHQ_cpp <- wd(aprx$get_AGHQ_cpp(eta = eta, Z = Z, p = p - 1L, 
                                 Sigma = Sigma, b = b))

cdf_aprx_cpp <- wd(aprx$get_cdf_cpp(eta = eta, Z = Z, p = p - 1L, 
                                    Sigma = Sigma, maxpts = maxpts))

qmc_aprx <- wd(
  aprx$get_qmc(eta = eta, Z = Z, p = p - 1L, Sigma = Sigma, 
               maxpts = maxpts))
qmc_Aaprx <- wd(
  aprx$get_Aqmc(eta = eta, Z = Z, p = p - 1L, Sigma = Sigma, 
                maxpts = maxpts))

sim_aprx <-  wd(aprx$get_sim_mth(eta = eta, Z = Z, p = p - 1L, 
                                 Sigma = Sigma, maxpts = maxpts))
sim_Aaprx <- wd(aprx$get_Asim_mth(eta = eta, Z = Z, p = p - 1L, 
                                 Sigma = Sigma, maxpts = maxpts))


#####
# compare results. Start with the simulation based methods with a lot of
# samples. We take this as the ground truth
truth_maybe_cdf <- wd(
  aprx$get_cdf_cpp (eta = eta, Z = Z, p = p - 1L, Sigma = Sigma,
                    maxpts = 1e6, abseps = -1, releps = 1e-11))()
truth_maybe_cdf
#> [1] 1.157e-07
#> attr(,"inform")
#> [1] 1
#> attr(,"error")
#> [1] 3.138e-11
#> attr(,"intvls")
#> [1] 746368

truth_maybe_Aqmc <- wd(
  aprx$get_Aqmc(eta = eta, Z = Z, p = p - 1L, Sigma = Sigma, maxpts = 1e6, 
                releps = 1e-11)())
truth_maybe_Aqmc
#> [1] 1.156e-07
#> attr(,"intvls")
#> [1] 1000000
#> attr(,"error")
#> [1] 2.287e-13

truth_maybe_Amc <- wd(
  aprx$get_Asim_mth(eta = eta, Z = Z, p = p - 1L, Sigma = Sigma, 
                    maxpts = 1e6, releps = 1e-11)(2L))
truth_maybe_Amc
#> [1] 1.156e-07
#> attr(,"error")
#> [1] 2.612e-12
#> attr(,"inform")
#> [1] 1
#> attr(,"inivls")
#> [1] 999991

truth <- wd(
  mixprobit:::aprx_mult_mix_brute(
    eta = eta, Z = Z, n_alt = p - 1L, Sigma = Sigma,  n_sim = 1e7, 
    n_threads = 6L))
c(Estiamte = truth, SE = attr(truth, "SE"),  
  `Estimate (log)` = log(c(truth)),  
  `SE (log)` = abs(attr(truth, "SE") / truth))
#>       Estiamte             SE Estimate (log)       SE (log) 
#>      1.156e-07      6.322e-13     -1.597e+01      5.467e-06
tr <- c(truth)

all.equal(tr, c(truth_maybe_cdf))
#> [1] "Mean relative difference: 8.311e-05"
all.equal(tr, c(truth_maybe_Aqmc))
#> [1] "Mean relative difference: 4.965e-07"
all.equal(tr, c(truth_maybe_Amc))
#> [1] "Mean relative difference: 1.107e-05"

# compare with using fewer samples and GHQ
all.equal(tr,   GHQ_cpp())
#> [1] "Mean relative difference: 0.02722"
all.equal(tr,   AGHQ_cpp())
#> [1] "Mean relative difference: 4.88e-06"
comp <- function(f, ...)
  mean(replicate(10, abs((tr - c(f())) / tr)))
comp(qmc_aprx)
#> [1] 0.003402
comp(qmc_Aaprx)
#> [1] 4.552e-05
comp(cdf_aprx_cpp)
#> [1] 0.0003883
comp(function() sim_aprx(1L))
#> [1] 0.01113
comp(function() sim_aprx(2L))
#> [1] 0.03465
comp(function() sim_aprx(3L))
#> [1] 0.05755
comp(function() sim_aprx(4L))
#> [1] 0.1135
comp(function() sim_Aaprx(1L))
#> [1] 0.0001677
comp(function() sim_Aaprx(2L))
#> [1] 5.654e-05
comp(function() sim_Aaprx(3L))
#> [1] 7.011e-05
comp(function() sim_Aaprx(4L))
#> [1] 4.103e-05

# compare computations times
microbenchmark::microbenchmark(
  `GHQ (C++)` = GHQ_cpp(), `AGHQ (C++)` = AGHQ_cpp(),
  `CDF (C++)` = cdf_aprx_cpp(),
  QMC = qmc_aprx(), `QMC Adaptive` = qmc_Aaprx(),
  `Genz & Monahan (1)` = sim_aprx(1L), `Genz & Monahan (2)` = sim_aprx(2L),
  `Genz & Monahan (3)` = sim_aprx(3L), `Genz & Monahan (4)` = sim_aprx(4L),
  `Genz & Monahan Adaptive (2)` = sim_Aaprx(2L),
  times = 5)
#> Unit: milliseconds
#>                         expr    min     lq    mean  median      uq     max
#>                    GHQ (C++) 265.92 266.84  271.76  270.10  277.56  278.37
#>                   AGHQ (C++) 232.76 235.60  246.90  239.77  258.10  268.27
#>                    CDF (C++)  64.32  64.93   70.02   65.49   74.46   80.89
#>                          QMC 944.36 960.39  963.07  962.41  963.09  985.11
#>                 QMC Adaptive 506.72 730.31  754.81  768.44  836.26  932.32
#>           Genz & Monahan (1) 929.39 945.51  987.15  975.79 1026.18 1058.89
#>           Genz & Monahan (2) 990.72 998.11 1008.46 1005.85 1020.08 1027.52
#>           Genz & Monahan (3) 978.30 992.57 1054.31  995.86 1045.00 1259.79
#>           Genz & Monahan (4) 956.91 963.25  999.07  972.68  986.23 1116.26
#>  Genz & Monahan Adaptive (2) 915.18 947.27  990.24 1004.16 1033.58 1051.01
#>  neval
#>      5
#>      5
#>      5
#>      5
#>      5
#>      5
#>      5
#>      5
#>      5
#>      5
```

The CDF approach is noticeably faster. One explanation is that the AGHQ
we are using, as of this writing, for the integrand with other methods
uses
![8(c - 1)](https://latex.codecogs.com/svg.latex?8%28c%20-%201%29 "8(c - 1)")
evaluations of the standard normal distribution’s CDF with
![c](https://latex.codecogs.com/svg.latex?c "c") being the number of
categories per observation in the cluster plus the
![K](https://latex.codecogs.com/svg.latex?K "K") evaluations of the
inverse normal CDF, and the overhead of finding the mode. In the example
above, this means that we do `8 * n * (p - 1)`, 240, CDF evaluations for
each of the `maxpts`, 40000, evaluations. We show how long this takes to
compute below when the evaluations points are drawn from the standard
normal distribution

``` r
local({
  Rcpp::sourceCpp(code = "
    // [[Rcpp::depends(RcppArmadillo)]]
    #include <RcppArmadillo.h>
    
    //[[Rcpp::export]]
    double test_pnorm(arma::vec const &x, arma::vec const  &unifs){
      double out(0), p, cp;
      for(auto xi : x){
        for(size_t j = 0; j < 8L; ++j){
          p = xi + (j - 3.5) / 3.5;
          R::pnorm_both(xi, &p, &cp, 1L, 1L);
        }
      }
      
      for(auto ui : unifs)
        out += R::qnorm5(ui, 0, 1, 1L, 0L);
      return out;
    }")
  
  u <- rnorm(n * (p - 1) * maxpts)
  uni <- runif(p * maxpts)
  microbenchmark::microbenchmark(test_pnorm(u, uni), times = 10)
})
#> Unit: milliseconds
#>                expr   min    lq  mean median    uq   max neval
#>  test_pnorm(u, uni) 401.7 403.8 407.6  406.5 412.5 414.7    10
```

In contrast, the CDF approximation can be implemented with only `n * p`
evaluation of the CDF, `n (p - 1)` evulations of the inverse CDF and
`n * (p - 1) - 1`, 29, evaluations of the log of the standard normal
distribution’s PDF for each of `maxpts`, 40000, evaluations. This is
much faster to evaluate as shown below

``` r
local({
  u <- rnorm(n * p * maxpts)
  Rcpp::sourceCpp(code = "
    // [[Rcpp::depends(RcppArmadillo)]]
    #include <RcppArmadillo.h>
    
    //[[Rcpp::export]]
    double test_dnorm(arma::vec const &x){
      static double const norm_const = 1. / std::sqrt(2 * M_PI);
      double out(0), p, cp;
      for(auto xi : x){
        p = xi;
        R::pnorm_both(xi, &p, &cp, 1L, 1L);
        out += p;
      }
      for(auto xi : x) // we do an extra one. Will not matter...
        out += R::qnorm5(xi, 0, 1, 1L, 0L);
      for(auto xi : x) // we do an extra one. Will not matter...
        out += -.5 * xi * xi;
      return out * norm_const;
    }")
  microbenchmark::microbenchmark(test_dnorm(u), times = 10)
})
#> Unit: milliseconds
#>           expr   min    lq  mean median   uq max neval
#>  test_dnorm(u) 96.39 96.48 96.96  96.51 96.7  99    10
```

More Rigorous Comparison (Multinomial)
--------------------------------------

Again, we perform a more rigorous comparison. We fix the relative error
of the methods before hand such that the relative error is below
`releps`,
![5\\times 10^{-4}](https://latex.codecogs.com/svg.latex?5%5Ctimes%2010%5E%7B-4%7D "5\times 10^{-4}").
Ground truth is computed with brute force MC using `n_brute`,
![10^{6}](https://latex.codecogs.com/svg.latex?10%5E%7B6%7D "10^{6}"),
samples.

We use a number of nodes such that this number of nodes or
`streak_length`, 4, less value of nodes with GHQ gives a relative error
which is below the threshold. We use a minimum of 4 nodes at the time of
this writing. The error of the simulation based methods is approximated
using `n_reps`, 20, replications.

``` r
# default parameters
ex_params <- list(
  streak_length = 4L, 
  max_b = 25L, 
  max_maxpts = 1000000L, 
  releps = 5e-4,
  min_releps = 1e-8,
  key_use = 3L, 
  n_reps = 20L, 
  n_runs = 5L, 
  n_brute = 1e6, 
  n_brute_max = 1e8, 
  n_brute_sds = 4, 
  qmc_n_seqs = 10L)
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
      
      truth <- wd(mixprobit:::aprx_mult_mix_brute(
        eta = eta, Z = Z, Sigma = Sigma, n_sim = n_brute, 
        n_threads = n_threads, n_alt = p - 1L, is_is = TRUE))
      
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
    test_val <- (log(vals) - log(truth)) / log(truth)
    if(!all(is.finite(test_val)))
      stop("non-finite 'vals'")
    sqrt(mean(test_val^2)) < releps / 2
  }
      
  # get function to use with GHQ
  get_b <- function(meth){
    if(do_not_run)
      NA_integer_
    else local({
      apx_func <- function(b)
        wd(meth(eta = eta, Z = Z, Sigma = Sigma, b = b, p = p - 1L))()
      
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
  b_use <- NA_integer_
  # b_use <- if(is_to_large_for_ghq)
  #   NA_integer_ else get_b(aprx$get_GHQ_cpp)
  ghq_func <- if(!is.na(b_use))
    wd(aprx$get_GHQ_cpp(eta = eta, Z = Z, Sigma = Sigma, b = b_use, 
                        p = p - 1L))
  else
    NA
  
  # get function to use with AGHQ
  b_use_A <- get_b(aprx$get_AGHQ_cpp)
  aghq_func <- if(!is.na(b_use_A))
    wd(aprx$get_AGHQ_cpp(eta = eta, Z = Z, Sigma = Sigma, b = b_use_A, 
                         p = p - 1L))
  else
    NA
  
  # get function to use with CDF method
  get_releps <- function(meth){
    if(do_not_run)
      NA_integer_
    else {
      releps_use <- releps * 1000
      repeat {
        func <- wd(meth(eta = eta, Z = Z, Sigma = Sigma, 
                        maxpts = ex_params$max_maxpts, p = p - 1L,
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
          warning("found no releps")
          releps_use <- NA_integer_
          break
        }
      }
      releps_use
    }
  }
  
  cdf_releps <- get_releps(aprx$get_cdf_cpp)
  cdf_func <- if(!is.na(cdf_releps))
    wd(aprx$get_cdf_cpp(eta = eta, Z = Z, Sigma = Sigma, p = p - 1L,
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
    wd(aprx$get_sim_mth(eta = eta, Z = Z, Sigma = Sigma, p = p - 1L,
                        maxpts = ex_params$max_maxpts, abseps = -1, 
                        releps = sim_releps))
  else 
    NA
  if(is.function(sim_func))
    formals(sim_func)$key <- key_use
  
  # do the same with the adaptive version
  Asim_releps <- get_releps(aprx$get_Asim_mth)
  Asim_func <- if(!is.na(Asim_releps))
    wd(aprx$get_Asim_mth(eta = eta, Z = Z, Sigma = Sigma, p = p - 1L,
                         maxpts = ex_params$max_maxpts, abseps = -1, 
                         releps = Asim_releps))
  else 
    NA
  if(is.function(Asim_func))
    formals(Asim_func)$key <- key_use
  
  # get function to use with QMC
  formals(aprx$get_qmc)$n_seqs <- ex_params$qmc_n_seqs
  # qmc_releps <- if(is_to_large_for_ghq)
  #   NA_integer_ else get_releps(aprx$get_qmc)
  qmc_releps <- NA_integer_
  qmc_func <- if(!is.na(qmc_releps))
     wd(aprx$get_qmc(eta = eta, Z = Z, Sigma = Sigma, p = p - 1L, 
                     maxpts = ex_params$max_maxpts, abseps = -1,
                     releps = qmc_releps, 
                     n_seqs = ex_params$qmc_n_seqs))
  else 
    NA

  # get function to use with adaptive QMC
  Aqmc_releps <- get_releps(aprx$get_Aqmc)
  formals(aprx$get_Aqmc)$n_seqs <- ex_params$qmc_n_seqs
  Aqmc_func <- if(!is.null(Aqmc_releps))
    wd(aprx$get_Aqmc(eta = eta, Z = Z, Sigma = Sigma, p = p - 1L,
                     maxpts = ex_params$max_maxpts, abseps = -1,
                     releps = Aqmc_releps, 
                     n_seqs = ex_params$qmc_n_seqs))
  else 
    NA
    
  # perform the comparison
  out <- sapply(
    list(GHQ = ghq_func, AGHQ = aghq_func, CDF = cdf_func, 
         GenzMonahan = sim_func, GenzMonahanA = Asim_func, 
         QMC = qmc_func, QMCA = Aqmc_func), 
    function(func){
      if(!is.function(func) && is.na(func)){
        out <- rep(NA_real_, 7L)
        names(out) <- c("mean", "sd", "mse", "user.self", 
                        "sys.self", "elapsed", "rel_rmse")
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
        mse <- mean((vs - truth)^2)
        rel_rmse <- sqrt(mean(((log(vs) - log(truth)) / log(truth))^2))
        
      } else {
        # we combine the variance estimators
        sd_use <- sqrt(mean(vals["sd", ]^2))
        vals <- vals["value", ]
        mse <- mean((vals - truth)^2)
        rel_rmse <- sqrt(mean(((log(vals) - log(truth)) / log(truth))^2))
        
      }
      
      c(mean = mean(vals), sd = sd_use, mse = mse, ti[1:3] / n_runs, 
        rel_rmse = rel_rmse)            
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
    sprintf("  Log-likelihood estimate (SE): %13.8f (%.8f)", x$ll_truth, 
            x$SE_truth), 
    "", sep = "\n")
  
  xx <- x$vals_n_comp_time["mean", ]
  print(cbind(`Mean estimate (likelihood)`     = xx, 
              `Mean estimate (log-likelihood)` = log(xx)))
  
  mult <- exp(ceiling(log10(1 / ex_params$releps)) * log(10))
  cat(sprintf("\nSD & RMSE (/%.2f)\n", mult))
  print(rbind(SD   = x$vals_n_comp_time ["sd", ],  
              RMSE = sqrt(x$vals_n_comp_time ["mse", ]), 
              `Rel RMSE` = x$vals_n_comp_time["rel_rmse", ]) * mult)
  
  cat("\nComputation times\n")
  print(x$vals_n_comp_time["elapsed", ])
}

set.seed(1)
sim_experiment(n =  2L, p = 3L, n_threads = 6L)
#>          # brute force samples:       1000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             7
#>                     CDF releps:    0.25000000
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.00097656
#>                     QMC releps:            NA
#>            Adaptive QMC releps:    0.00097656
#>   Log-likelihood estimate (SE):   -3.24068105 (0.00013983)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                          0.0391405                       -3.24060
#> CDF                           0.0391661                       -3.23994
#> GenzMonahan                          NA                             NA
#> GenzMonahanA                  0.0391346                       -3.24075
#> QMC                                  NA                             NA
#> QMCA                          0.0391342                       -3.24076
#> 
#> SD & RMSE (/10000.00)
#>          GHQ      AGHQ      CDF GenzMonahan GenzMonahanA QMC     QMCA
#> SD        NA 0.0644611 0.398768          NA     0.148358  NA 0.142237
#> RMSE      NA 0.0577083 0.378684          NA     0.118825  NA 0.407591
#> Rel RMSE  NA 0.4550527 2.984086          NA     0.937018  NA 3.215995
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>           NA       0.0012       0.0002           NA       0.3072           NA 
#>         QMCA 
#>       0.2072
sim_experiment(n =  4L, p = 3L, n_threads = 6L)
#>          # brute force samples:       1000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             6
#>                     CDF releps:    0.00195312
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.50000000
#>                     QMC releps:            NA
#>            Adaptive QMC releps:    0.00195312
#>   Log-likelihood estimate (SE):   -4.37606503 (0.00004298)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                          0.0125744                       -4.37610
#> CDF                           0.0125749                       -4.37605
#> GenzMonahan                          NA                             NA
#> GenzMonahanA                  0.0125761                       -4.37596
#> QMC                                  NA                             NA
#> QMCA                          0.0125822                       -4.37548
#> 
#> SD & RMSE (/10000.00)
#>          GHQ       AGHQ       CDF GenzMonahan GenzMonahanA QMC     QMCA
#> SD        NA 0.00838196 0.0763867          NA    0.0286483  NA 0.080496
#> RMSE      NA 0.00957922 0.0502401          NA    0.0346883  NA 0.111286
#> Rel RMSE  NA 0.17409286 0.9129156          NA    0.6303443  NA 2.021164
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>           NA       0.0016       0.0050           NA       0.0010           NA 
#>         QMCA 
#>       0.0110
sim_experiment(n =  8L, p = 3L, n_threads = 6L)
#>          # brute force samples:       1000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             4
#>                     CDF releps:    0.50000000
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.50000000
#>                     QMC releps:            NA
#>            Adaptive QMC releps:    0.50000000
#>   Log-likelihood estimate (SE):  -15.48150224 (0.00001951)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                     0.000000189007                       -15.4815
#> CDF                      0.000000188921                       -15.4819
#> GenzMonahan                          NA                             NA
#> GenzMonahanA             0.000000189016                       -15.4814
#> QMC                                  NA                             NA
#> QMCA                     0.000000189122                       -15.4809
#> 
#> SD & RMSE (/10000.00)
#>          GHQ          AGHQ           CDF GenzMonahan  GenzMonahanA QMC
#> SD        NA 0.00000330864 0.00000565877          NA 0.00000110648  NA
#> RMSE      NA 0.00000304791 0.00000217613          NA 0.00000086175  NA
#> Rel RMSE  NA 1.04013966097 0.74422244997          NA 0.29449958700  NA
#>                   QMCA
#> SD       0.00000235749
#> RMSE     0.00000237140
#> Rel RMSE 0.80978091289
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>           NA       0.0012       0.0014           NA       0.0020           NA 
#>         QMCA 
#>       0.0050
sim_experiment(n = 16L, p = 3L, n_threads = 6L)
#>          # brute force samples:       1000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             5
#>                     CDF releps:    0.00781250
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.50000000
#>                     QMC releps:            NA
#>            Adaptive QMC releps:    0.00390625
#>   Log-likelihood estimate (SE):  -10.95650764 (0.00002054)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                       0.0000174436                       -10.9565
#> CDF                        0.0000174534                       -10.9560
#> GenzMonahan                          NA                             NA
#> GenzMonahanA               0.0000174331                       -10.9571
#> QMC                                  NA                             NA
#> QMCA                       0.0000174412                       -10.9567
#> 
#> SD & RMSE (/10000.00)
#>          GHQ        AGHQ         CDF GenzMonahan GenzMonahanA QMC        QMCA
#> SD        NA 0.000293058 0.000487082          NA  0.000321853  NA 0.000236325
#> RMSE      NA 0.000324009 0.000252211          NA  0.000151333  NA 0.000159988
#> Rel RMSE  NA 1.698267977 1.318483904          NA  0.792251628  NA 0.837190316
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>           NA       0.0048       0.0188           NA       0.0044           NA 
#>         QMCA 
#>       0.0316
sim_experiment(n = 32L, p = 3L, n_threads = 6L)
#>          # brute force samples:       1000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             4
#>                     CDF releps:    0.01562500
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.50000000
#>                     QMC releps:            NA
#>            Adaptive QMC releps:    0.50000000
#>   Log-likelihood estimate (SE):  -27.05525634 (0.00000619)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                0.00000000000177847                       -27.0553
#> CDF                 0.00000000000176998                       -27.0601
#> GenzMonahan                          NA                             NA
#> GenzMonahanA        0.00000000000177854                       -27.0552
#> QMC                                  NA                             NA
#> QMCA                0.00000000000177832                       -27.0554
#> 
#> SD & RMSE (/10000.00)
#>          GHQ               AGHQ                CDF GenzMonahan
#> SD        NA 0.0000000000171133 0.0000000001017300          NA
#> RMSE      NA 0.0000000000202855 0.0000000000925754          NA
#> Rel RMSE  NA 0.4220130688764949 1.9300004824596837          NA
#>                 GenzMonahanA QMC               QMCA
#> SD       0.00000000000534081  NA 0.0000000000265594
#> RMSE     0.00000000000696942  NA 0.0000000000288489
#> Rel RMSE 0.14482335569997018  NA 0.5995844197574175
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>           NA       0.0056       0.0764           NA       0.0082           NA 
#>         QMCA 
#>       0.0198

sim_experiment(n =  2L, p = 4L, n_threads = 6L)
#>          # brute force samples:       1000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             6
#>                     CDF releps:    0.50000000
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.00097656
#>                     QMC releps:            NA
#>            Adaptive QMC releps:    0.00097656
#>   Log-likelihood estimate (SE):   -2.57926328 (0.00005098)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                          0.0758256                       -2.57932
#> CDF                           0.0758560                       -2.57892
#> GenzMonahan                          NA                             NA
#> GenzMonahanA                  0.0758136                       -2.57948
#> QMC                                  NA                             NA
#> QMCA                          0.0758585                       -2.57889
#> 
#> SD & RMSE (/10000.00)
#>          GHQ     AGHQ      CDF GenzMonahan GenzMonahanA QMC     QMCA
#> SD        NA 0.121852 0.817109          NA     0.258170  NA 0.249906
#> RMSE      NA 0.109975 0.426435          NA     0.212061  NA 0.361609
#> Rel RMSE  NA 0.562267 2.179396          NA     1.084488  NA 1.848225
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>           NA       0.0066       0.0004           NA       0.0124           NA 
#>         QMCA 
#>       0.0456
sim_experiment(n =  4L, p = 4L, n_threads = 6L)
#>          # brute force samples:       1000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             7
#>                     CDF releps:    0.12500000
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.00195312
#>                     QMC releps:            NA
#>            Adaptive QMC releps:    0.00195312
#>   Log-likelihood estimate (SE):   -5.48264627 (0.00007380)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                         0.00415816                       -5.48268
#> CDF                          0.00415972                       -5.48231
#> GenzMonahan                          NA                             NA
#> GenzMonahanA                 0.00415941                       -5.48238
#> QMC                                  NA                             NA
#> QMCA                         0.00415439                       -5.48359
#> 
#> SD & RMSE (/10000.00)
#>          GHQ       AGHQ       CDF GenzMonahan GenzMonahanA QMC      QMCA
#> SD        NA 0.00357056 0.0805191          NA    0.0288289  NA 0.0265070
#> RMSE      NA 0.00498885 0.0462447          NA    0.0266737  NA 0.0508586
#> Rel RMSE  NA 0.21884480 2.0276537          NA    1.1693715  NA 2.2325897
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>           NA       0.0258       0.0012           NA       0.0366           NA 
#>         QMCA 
#>       0.0700
sim_experiment(n =  8L, p = 4L, n_threads = 6L)
#>          # brute force samples:       1000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             6
#>                     CDF releps:    0.00781250
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.00781250
#>                     QMC releps:            NA
#>            Adaptive QMC releps:    0.00390625
#>   Log-likelihood estimate (SE):  -12.24320048 (0.00005840)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                      0.00000481819                       -12.2431
#> CDF                       0.00000482602                       -12.2415
#> GenzMonahan                          NA                             NA
#> GenzMonahanA              0.00000479782                       -12.2473
#> QMC                                  NA                             NA
#> QMCA                      0.00000480794                       -12.2452
#> 
#> SD & RMSE (/10000.00)
#>          GHQ         AGHQ         CDF GenzMonahan GenzMonahanA QMC         QMCA
#> SD        NA 0.0000290393 0.000110461          NA  0.000110688  NA 0.0000641956
#> RMSE      NA 0.0000275100 0.000112113          NA  0.000269057  NA 0.0001398326
#> Rel RMSE  NA 0.4666815516 1.898325189          NA  4.580643993  NA 2.3746799461
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>           NA       0.0268       0.0088           NA       0.0066           NA 
#>         QMCA 
#>       0.0566
sim_experiment(n = 16L, p = 4L, n_threads = 6L)
#>          # brute force samples:       1000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             5
#>                     CDF releps:    0.00781250
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.50000000
#>                     QMC releps:            NA
#>            Adaptive QMC releps:    0.01562500
#>   Log-likelihood estimate (SE):  -17.29048839 (0.00003521)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                    0.0000000309632                       -17.2905
#> CDF                     0.0000000309560                       -17.2907
#> GenzMonahan                          NA                             NA
#> GenzMonahanA            0.0000000309266                       -17.2917
#> QMC                                  NA                             NA
#> QMCA                    0.0000000309106                       -17.2922
#> 
#> SD & RMSE (/10000.00)
#>          GHQ           AGHQ            CDF GenzMonahan   GenzMonahanA QMC
#> SD        NA 0.000000949897 0.000000888727          NA 0.000000971357  NA
#> RMSE      NA 0.000001006792 0.000000595495          NA 0.000001109115  NA
#> Rel RMSE  NA 1.886967754811 1.112274674429          NA 2.074524239485  NA
#>                   QMCA
#> SD       0.00000126386
#> RMSE     0.00000193250
#> Rel RMSE 3.61504969292
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>           NA       0.0452       0.1010           NA       0.0058           NA 
#>         QMCA 
#>       0.0178
sim_experiment(n = 32L, p = 4L, n_threads = 6L)
#>          # brute force samples:       1000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             4
#>                     CDF releps:    0.03125000
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.50000000
#>                     QMC releps:            NA
#>            Adaptive QMC releps:    0.50000000
#>   Log-likelihood estimate (SE):  -33.82240832 (0.00002531)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ             0.00000000000000204682                       -33.8225
#> CDF              0.00000000000000205466                       -33.8187
#> GenzMonahan                          NA                             NA
#> GenzMonahanA     0.00000000000000204402                       -33.8239
#> QMC                                  NA                             NA
#> QMCA             0.00000000000000204980                       -33.8210
#> 
#> SD & RMSE (/10000.00)
#>          GHQ                  AGHQ                  CDF GenzMonahan
#> SD        NA 0.0000000000000938917 0.000000000000211170          NA
#> RMSE      NA 0.0000000000001125262 0.000000000000138023          NA
#> Rel RMSE  NA 1.6328776351716158821 1.986213389155755848          NA
#>                   GenzMonahanA QMC                  QMCA
#> SD       0.0000000000000568164  NA 0.0000000000001164525
#> RMSE     0.0000000000000790993  NA 0.0000000000000838864
#> Rel RMSE 1.1473914440671606485  NA 1.2072502765825812165
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>           NA       0.0234       0.0652           NA       0.0122           NA 
#>         QMCA 
#>       0.0356

sim_experiment(n =  2L, p = 5L, n_threads = 6L)
#>          # brute force samples:       1000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             6
#>                     CDF releps:    0.00195312
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.00048828
#>                     QMC releps:            NA
#>            Adaptive QMC releps:    0.00097656
#>   Log-likelihood estimate (SE):   -2.84281256 (0.00005065)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                          0.0582607                       -2.84283
#> CDF                           0.0582618                       -2.84281
#> GenzMonahan                          NA                             NA
#> GenzMonahanA                  0.0582663                       -2.84273
#> QMC                                  NA                             NA
#> QMCA                          0.0582898                       -2.84233
#> 
#> SD & RMSE (/10000.00)
#>          GHQ      AGHQ      CDF GenzMonahan GenzMonahanA QMC     QMCA
#> SD        NA 0.0696056 0.332336          NA     0.107589  NA 0.192243
#> RMSE      NA 0.0640661 0.203880          NA     0.117362  NA 0.343277
#> Rel RMSE  NA 0.3867685 1.231023          NA     0.708480  NA 2.071813
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>           NA       0.0508       0.0036           NA       0.0190           NA 
#>         QMCA 
#>       0.0560
sim_experiment(n =  4L, p = 5L, n_threads = 6L)
#>          # brute force samples:       1000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             6
#>                     CDF releps:    0.00390625
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.00195312
#>                     QMC releps:            NA
#>            Adaptive QMC releps:    0.00195312
#>   Log-likelihood estimate (SE):   -5.74331361 (0.00005898)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                         0.00320372                       -5.74344
#> CDF                          0.00320385                       -5.74340
#> GenzMonahan                          NA                             NA
#> GenzMonahanA                 0.00320200                       -5.74398
#> QMC                                  NA                             NA
#> QMCA                         0.00320396                       -5.74337
#> 
#> SD & RMSE (/10000.00)
#>          GHQ       AGHQ       CDF GenzMonahan GenzMonahanA QMC      QMCA
#> SD        NA 0.00442094 0.0334425          NA    0.0209137  NA 0.0202197
#> RMSE      NA 0.00515591 0.0339049          NA    0.0423540  NA 0.0185545
#> Rel RMSE  NA 0.28020141 1.8421262          NA    2.3036803  NA 1.0086084
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>           NA       0.1038       0.0092           NA       0.0078           NA 
#>         QMCA 
#>       0.0702
sim_experiment(n =  8L, p = 5L, n_threads = 6L)
#>          # brute force samples:       1000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             6
#>                     CDF releps:    0.00781250
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.25000000
#>                     QMC releps:            NA
#>            Adaptive QMC releps:    0.00781250
#>   Log-likelihood estimate (SE):  -11.66493762 (0.00005315)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                      0.00000858958                       -11.6650
#> CDF                       0.00000860078                       -11.6637
#> GenzMonahan                          NA                             NA
#> GenzMonahanA              0.00000858415                       -11.6656
#> QMC                                  NA                             NA
#> QMCA                      0.00000858955                       -11.6650
#> 
#> SD & RMSE (/10000.00)
#>          GHQ         AGHQ         CDF GenzMonahan GenzMonahanA QMC        QMCA
#> SD        NA 0.0000378947 0.000230459          NA  0.000162375  NA 0.000239597
#> RMSE      NA 0.0000399838 0.000223153          NA  0.000112960  NA 0.000184772
#> Rel RMSE  NA 0.3992461999 2.223356216          NA  1.127900474  NA 1.845399991
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>           NA       0.2018       0.0314           NA       0.0048           NA 
#>         QMCA 
#>       0.0248
sim_experiment(n = 16L, p = 5L, n_threads = 6L)
#>          # brute force samples:       1000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             6
#>                     CDF releps:    0.01562500
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.50000000
#>                     QMC releps:            NA
#>            Adaptive QMC releps:    0.00781250
#>   Log-likelihood estimate (SE):  -18.57387788 (0.00004733)
#> 
#>              Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                  NA                             NA
#> AGHQ                   0.00000000857876                       -18.5740
#> CDF                    0.00000000860408                       -18.5710
#> GenzMonahan                          NA                             NA
#> GenzMonahanA           0.00000000856698                       -18.5754
#> QMC                                  NA                             NA
#> QMCA                   0.00000000857596                       -18.5743
#> 
#> SD & RMSE (/10000.00)
#>          GHQ           AGHQ            CDF GenzMonahan   GenzMonahanA QMC
#> SD        NA 0.000000048501 0.000000402865          NA 0.000000422636  NA
#> RMSE      NA 0.000000053834 0.000000261999          NA 0.000000375264  NA
#> Rel RMSE  NA 0.338048342807 1.641225654895          NA 2.359948605129  NA
#>                    QMCA
#> SD       0.000000189870
#> RMSE     0.000000277606
#> Rel RMSE 1.744788829693
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>           NA       0.3950       0.0594           NA       0.0102           NA 
#>         QMCA 
#>       0.0780
sim_experiment(n = 32L, p = 5L, n_threads = 6L)
#>          # brute force samples:       1000000
#>                   # nodes  GHQ:            NA
#>                   # nodes AGHQ:             4
#>                     CDF releps:    0.03125000
#>          Genz & Monahan releps:            NA
#> Adaptive Genz & Monahan releps:    0.50000000
#>                     QMC releps:            NA
#>            Adaptive QMC releps:    0.50000000
#>   Log-likelihood estimate (SE):  -46.15490848 (0.00002711)
#> 
#>                Mean estimate (likelihood) Mean estimate (log-likelihood)
#> GHQ                                    NA                             NA
#> AGHQ         0.00000000000000000000901889                       -46.1550
#> CDF          0.00000000000000000000904841                       -46.1517
#> GenzMonahan                            NA                             NA
#> GenzMonahanA 0.00000000000000000000902318                       -46.1545
#> QMC                                    NA                             NA
#> QMCA         0.00000000000000000000903855                       -46.1528
#> 
#> SD & RMSE (/10000.00)
#>          GHQ                       AGHQ                        CDF GenzMonahan
#> SD        NA 0.000000000000000000533861 0.000000000000000000935806          NA
#> RMSE      NA 0.000000000000000000635923 0.000000000000000000669593          NA
#> Rel RMSE  NA 1.537181219384197383703849 1.603844008841765811368418          NA
#>                        GenzMonahanA QMC                       QMCA
#> SD       0.000000000000000000162306  NA 0.000000000000000000464894
#> RMSE     0.000000000000000000125255  NA 0.000000000000000000475751
#> Rel RMSE 0.300702206219131906728137  NA 1.138135082005084397493988
#> 
#> Computation times
#>          GHQ         AGHQ          CDF  GenzMonahan GenzMonahanA          QMC 
#>           NA       0.1056       0.2130           NA       0.0202           NA 
#>         QMCA 
#>       0.0546
```

``` r
# number of observations in the cluster
n_vals <- 2^(1:5)
# number of random effects
p_vals <- 3:6
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
    cache_file <- file.path(cache_dir, sprintf("mult-n-%03d-p-%03d.Rds", n, p))
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
      message(sprintf("Loading results with n %3d and p %3d", n, p))
      
    
    readRDS(cache_file)
  }, n = gr_vals$n, p = gr_vals$p, SIMPLIFY = FALSE, 
  mc.cores = 4L, mc.preschedule = FALSE)
})()
```

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

| n   | method/p                       |    3|    4|    5|    6|
|:----|:-------------------------------|----:|----:|----:|----:|
| 2   | AGHQ                           |  100|  100|  100|  100|
|     | CDF                            |  100|  100|  100|  100|
|     | Genz & Monahan (1999) Adaptive |  100|  100|  100|  100|
|     | Adaptive QMC                   |  100|  100|  100|  100|
| 4   | AGHQ                           |  100|  100|  100|  100|
|     | CDF                            |  100|  100|  100|  100|
|     | Genz & Monahan (1999) Adaptive |  100|  100|  100|  100|
|     | Adaptive QMC                   |  100|  100|  100|  100|
| 8   | AGHQ                           |  100|  100|  100|  100|
|     | CDF                            |  100|  100|  100|  100|
|     | Genz & Monahan (1999) Adaptive |  100|  100|  100|  100|
|     | Adaptive QMC                   |  100|  100|  100|  100|
| 16  | AGHQ                           |  100|  100|  100|  100|
|     | CDF                            |  100|  100|  100|  100|
|     | Genz & Monahan (1999) Adaptive |  100|  100|  100|  100|
|     | Adaptive QMC                   |  100|  100|  100|  100|
| 32  | AGHQ                           |  100|  100|  100|  100|
|     | CDF                            |  100|  100|  100|  100|
|     | Genz & Monahan (1999) Adaptive |  100|  100|  100|  100|
|     | Adaptive QMC                   |  100|  100|  100|  100|

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
  
  is_complete <- apply(comp_times, 3, function(x){
    if(remove_nas){
      consider <- !apply(is.na(x), 1L, all)
      apply(!is.na(x[consider, , drop = FALSE]), 2, all)
    } else 
      rep(TRUE, NCOL(x))
  })
  
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

| n   | method/p                       |      3|       4|       5|        6|
|:----|:-------------------------------|------:|-------:|-------:|--------:|
| 2   | GHQ                            |       |        |        |         |
|     | AGHQ                           |   1.15|    8.70|   60.96|   521.23|
|     | CDF                            |   0.61|    1.06|    2.06|     3.27|
|     | Genz & Monahan (1999)          |       |        |        |         |
|     | Genz & Monahan (1999) Adaptive |  19.00|  344.99|  498.84|   946.86|
|     | QMC                            |       |        |        |         |
|     | Adaptive QMC                   |  44.24|  134.22|   97.93|   180.91|
| 4   | GHQ                            |       |        |        |         |
|     | AGHQ                           |   2.12|   15.57|  106.49|   841.43|
|     | CDF                            |   1.77|    4.93|    9.83|    16.54|
|     | Genz & Monahan (1999)          |       |        |        |         |
|     | Genz & Monahan (1999) Adaptive |  30.86|  368.31|  142.77|   864.25|
|     | QMC                            |       |        |        |         |
|     | Adaptive QMC                   |  78.83|  114.67|   66.06|   122.49|
| 8   | GHQ                            |       |        |        |         |
|     | AGHQ                           |   3.25|   25.01|  168.76|  1165.05|
|     | CDF                            |   6.78|   17.31|   31.68|    82.60|
|     | Genz & Monahan (1999)          |       |        |        |         |
|     | Genz & Monahan (1999) Adaptive |  11.46|   83.95|  127.65|    70.39|
|     | QMC                            |       |        |        |         |
|     | Adaptive QMC                   |  28.87|   36.04|   34.49|    76.09|
| 16  | GHQ                            |       |        |        |         |
|     | AGHQ                           |   4.42|   25.08|  167.01|  1157.20|
|     | CDF                            |  17.81|   48.96|  111.83|   262.76|
|     | Genz & Monahan (1999)          |       |        |        |         |
|     | Genz & Monahan (1999) Adaptive |   4.55|    6.75|   11.36|    17.56|
|     | QMC                            |       |        |        |         |
|     | Adaptive QMC                   |  12.56|   21.64|   32.53|    44.63|
| 32  | GHQ                            |       |        |        |         |
|     | AGHQ                           |   5.92|   24.82|  115.83|   477.80|
|     | CDF                            |  47.40|  229.36|  402.94|  1006.20|
|     | Genz & Monahan (1999)          |       |        |        |         |
|     | Genz & Monahan (1999) Adaptive |   8.92|   13.03|   22.89|    29.54|
|     | QMC                            |       |        |        |         |
|     | Adaptive QMC                   |  23.17|   37.80|   60.51|    72.00|

``` r
show_run_times(na.rm = TRUE)
```

**NAs have been removed. Cells may not be comparable (means)**

| n   | method/p                       |      3|       4|       5|        6|
|:----|:-------------------------------|------:|-------:|-------:|--------:|
| 2   | AGHQ                           |   1.15|    8.70|   60.96|   521.23|
|     | CDF                            |   0.61|    1.06|    2.06|     3.27|
|     | Genz & Monahan (1999) Adaptive |  19.00|  344.99|  498.84|   946.86|
|     | Adaptive QMC                   |  44.24|  134.22|   97.93|   180.91|
| 4   | AGHQ                           |   2.12|   15.57|  106.49|   841.43|
|     | CDF                            |   1.77|    4.93|    9.83|    16.54|
|     | Genz & Monahan (1999) Adaptive |  30.86|  368.31|  142.77|   864.25|
|     | Adaptive QMC                   |  78.83|  114.67|   66.06|   122.49|
| 8   | AGHQ                           |   3.25|   25.01|  168.76|  1165.05|
|     | CDF                            |   6.78|   17.31|   31.68|    82.60|
|     | Genz & Monahan (1999) Adaptive |  11.46|   83.95|  127.65|    70.39|
|     | Adaptive QMC                   |  28.87|   36.04|   34.49|    76.09|
| 16  | AGHQ                           |   4.42|   25.08|  167.01|  1157.20|
|     | CDF                            |  17.81|   48.96|  111.83|   262.76|
|     | Genz & Monahan (1999) Adaptive |   4.55|    6.75|   11.36|    17.56|
|     | Adaptive QMC                   |  12.56|   21.64|   32.53|    44.63|
| 32  | AGHQ                           |   5.92|   24.82|  115.83|   477.80|
|     | CDF                            |  47.40|  229.36|  402.94|  1006.20|
|     | Genz & Monahan (1999) Adaptive |   8.92|   13.03|   22.89|    29.54|
|     | Adaptive QMC                   |  23.17|   37.80|   60.51|    72.00|

``` r
show_run_times(TRUE)
```

**Only showing complete cases (means)**

| n   | method/p                       |      3|       4|       5|        6|
|:----|:-------------------------------|------:|-------:|-------:|--------:|
| 2   | GHQ                            |       |        |        |         |
|     | AGHQ                           |   1.15|    8.70|   60.96|   521.23|
|     | CDF                            |   0.61|    1.06|    2.06|     3.27|
|     | Genz & Monahan (1999)          |       |        |        |         |
|     | Genz & Monahan (1999) Adaptive |  19.00|  344.99|  498.84|   946.86|
|     | QMC                            |       |        |        |         |
|     | Adaptive QMC                   |  44.24|  134.22|   97.93|   180.91|
| 4   | GHQ                            |       |        |        |         |
|     | AGHQ                           |   2.12|   15.57|  106.49|   841.43|
|     | CDF                            |   1.77|    4.93|    9.83|    16.54|
|     | Genz & Monahan (1999)          |       |        |        |         |
|     | Genz & Monahan (1999) Adaptive |  30.86|  368.31|  142.77|   864.25|
|     | QMC                            |       |        |        |         |
|     | Adaptive QMC                   |  78.83|  114.67|   66.06|   122.49|
| 8   | GHQ                            |       |        |        |         |
|     | AGHQ                           |   3.25|   25.01|  168.76|  1165.05|
|     | CDF                            |   6.78|   17.31|   31.68|    82.60|
|     | Genz & Monahan (1999)          |       |        |        |         |
|     | Genz & Monahan (1999) Adaptive |  11.46|   83.95|  127.65|    70.39|
|     | QMC                            |       |        |        |         |
|     | Adaptive QMC                   |  28.87|   36.04|   34.49|    76.09|
| 16  | GHQ                            |       |        |        |         |
|     | AGHQ                           |   4.42|   25.08|  167.01|  1157.20|
|     | CDF                            |  17.81|   48.96|  111.83|   262.76|
|     | Genz & Monahan (1999)          |       |        |        |         |
|     | Genz & Monahan (1999) Adaptive |   4.55|    6.75|   11.36|    17.56|
|     | QMC                            |       |        |        |         |
|     | Adaptive QMC                   |  12.56|   21.64|   32.53|    44.63|
| 32  | GHQ                            |       |        |        |         |
|     | AGHQ                           |   5.92|   24.82|  115.83|   477.80|
|     | CDF                            |  47.40|  229.36|  402.94|  1006.20|
|     | Genz & Monahan (1999)          |       |        |        |         |
|     | Genz & Monahan (1999) Adaptive |   8.92|   13.03|   22.89|    29.54|
|     | QMC                            |       |        |        |         |
|     | Adaptive QMC                   |  23.17|   37.80|   60.51|    72.00|

**Number of complete cases**

|     |    3|    4|    5|    6|
|-----|----:|----:|----:|----:|
| 2   |  100|  100|  100|  100|
| 4   |  100|  100|  100|  100|
| 8   |  100|  100|  100|  100|
| 16  |  100|  100|  100|  100|
| 32  |  100|  100|  100|  100|

``` r

# show medians instead
med_func <- function(x, na.rm)
  apply(x, 1, median, na.rm = na.rm)
show_run_times(meth = med_func, suffix = " (median)", FALSE)
```

**Blank cells have at least one failure (median)**

| n   | method/p                       |      3|       4|       5|        6|
|:----|:-------------------------------|------:|-------:|-------:|--------:|
| 2   | GHQ                            |       |        |        |         |
|     | AGHQ                           |   1.00|    7.40|   55.10|   403.00|
|     | CDF                            |   0.40|    0.40|    0.60|     2.60|
|     | Genz & Monahan (1999)          |       |        |        |         |
|     | Genz & Monahan (1999) Adaptive |   0.60|  113.50|  119.80|   380.80|
|     | QMC                            |       |        |        |         |
|     | Adaptive QMC                   |  21.10|   72.40|   70.20|   123.60|
| 4   | GHQ                            |       |        |        |         |
|     | AGHQ                           |   2.00|   14.30|  106.80|   812.10|
|     | CDF                            |   0.60|    4.80|    7.60|    15.90|
|     | Genz & Monahan (1999)          |       |        |        |         |
|     | Genz & Monahan (1999) Adaptive |   1.20|   28.60|    3.00|   145.80|
|     | QMC                            |       |        |        |         |
|     | Adaptive QMC                   |  22.20|   56.30|   46.00|    87.50|
| 8   | GHQ                            |       |        |        |         |
|     | AGHQ                           |   3.60|   28.00|  208.70|  1545.00|
|     | CDF                            |   5.80|   12.10|   23.00|    66.00|
|     | Genz & Monahan (1999)          |       |        |        |         |
|     | Genz & Monahan (1999) Adaptive |   2.20|    3.20|    5.60|     9.00|
|     | QMC                            |       |        |        |         |
|     | Adaptive QMC                   |   6.20|   23.10|   17.50|    44.20|
| 16  | GHQ                            |       |        |        |         |
|     | AGHQ                           |   4.80|   27.60|  169.20|  1032.10|
|     | CDF                            |  13.50|   35.00|   80.70|   185.50|
|     | Genz & Monahan (1999)          |       |        |        |         |
|     | Genz & Monahan (1999) Adaptive |   4.40|    6.60|   11.20|    17.20|
|     | QMC                            |       |        |        |         |
|     | Adaptive QMC                   |  11.20|   19.60|   30.40|    43.20|
| 32  | GHQ                            |       |        |        |         |
|     | AGHQ                           |   5.60|   24.40|  113.40|   451.90|
|     | CDF                            |  42.80|  120.90|  264.60|   641.80|
|     | Genz & Monahan (1999)          |       |        |        |         |
|     | Genz & Monahan (1999) Adaptive |   8.80|   12.80|   22.50|    29.40|
|     | QMC                            |       |        |        |         |
|     | Adaptive QMC                   |  21.40|   37.60|   59.60|    71.80|

``` r
show_run_times(meth = med_func, suffix = " (median)", na.rm = TRUE)
```

**NAs have been removed. Cells may not be comparable (median)**

| n   | method/p                       |      3|       4|       5|        6|
|:----|:-------------------------------|------:|-------:|-------:|--------:|
| 2   | AGHQ                           |   1.00|    7.40|   55.10|   403.00|
|     | CDF                            |   0.40|    0.40|    0.60|     2.60|
|     | Genz & Monahan (1999) Adaptive |   0.60|  113.50|  119.80|   380.80|
|     | Adaptive QMC                   |  21.10|   72.40|   70.20|   123.60|
| 4   | AGHQ                           |   2.00|   14.30|  106.80|   812.10|
|     | CDF                            |   0.60|    4.80|    7.60|    15.90|
|     | Genz & Monahan (1999) Adaptive |   1.20|   28.60|    3.00|   145.80|
|     | Adaptive QMC                   |  22.20|   56.30|   46.00|    87.50|
| 8   | AGHQ                           |   3.60|   28.00|  208.70|  1545.00|
|     | CDF                            |   5.80|   12.10|   23.00|    66.00|
|     | Genz & Monahan (1999) Adaptive |   2.20|    3.20|    5.60|     9.00|
|     | Adaptive QMC                   |   6.20|   23.10|   17.50|    44.20|
| 16  | AGHQ                           |   4.80|   27.60|  169.20|  1032.10|
|     | CDF                            |  13.50|   35.00|   80.70|   185.50|
|     | Genz & Monahan (1999) Adaptive |   4.40|    6.60|   11.20|    17.20|
|     | Adaptive QMC                   |  11.20|   19.60|   30.40|    43.20|
| 32  | AGHQ                           |   5.60|   24.40|  113.40|   451.90|
|     | CDF                            |  42.80|  120.90|  264.60|   641.80|
|     | Genz & Monahan (1999) Adaptive |   8.80|   12.80|   22.50|    29.40|
|     | Adaptive QMC                   |  21.40|   37.60|   59.60|    71.80|

``` r
show_run_times(meth = med_func, suffix = " (median)", TRUE)
```

**Only showing complete cases (median)**

| n   | method/p                       |      3|       4|       5|        6|
|:----|:-------------------------------|------:|-------:|-------:|--------:|
| 2   | GHQ                            |       |        |        |         |
|     | AGHQ                           |   1.00|    7.40|   55.10|   403.00|
|     | CDF                            |   0.40|    0.40|    0.60|     2.60|
|     | Genz & Monahan (1999)          |       |        |        |         |
|     | Genz & Monahan (1999) Adaptive |   0.60|  113.50|  119.80|   380.80|
|     | QMC                            |       |        |        |         |
|     | Adaptive QMC                   |  21.10|   72.40|   70.20|   123.60|
| 4   | GHQ                            |       |        |        |         |
|     | AGHQ                           |   2.00|   14.30|  106.80|   812.10|
|     | CDF                            |   0.60|    4.80|    7.60|    15.90|
|     | Genz & Monahan (1999)          |       |        |        |         |
|     | Genz & Monahan (1999) Adaptive |   1.20|   28.60|    3.00|   145.80|
|     | QMC                            |       |        |        |         |
|     | Adaptive QMC                   |  22.20|   56.30|   46.00|    87.50|
| 8   | GHQ                            |       |        |        |         |
|     | AGHQ                           |   3.60|   28.00|  208.70|  1545.00|
|     | CDF                            |   5.80|   12.10|   23.00|    66.00|
|     | Genz & Monahan (1999)          |       |        |        |         |
|     | Genz & Monahan (1999) Adaptive |   2.20|    3.20|    5.60|     9.00|
|     | QMC                            |       |        |        |         |
|     | Adaptive QMC                   |   6.20|   23.10|   17.50|    44.20|
| 16  | GHQ                            |       |        |        |         |
|     | AGHQ                           |   4.80|   27.60|  169.20|  1032.10|
|     | CDF                            |  13.50|   35.00|   80.70|   185.50|
|     | Genz & Monahan (1999)          |       |        |        |         |
|     | Genz & Monahan (1999) Adaptive |   4.40|    6.60|   11.20|    17.20|
|     | QMC                            |       |        |        |         |
|     | Adaptive QMC                   |  11.20|   19.60|   30.40|    43.20|
| 32  | GHQ                            |       |        |        |         |
|     | AGHQ                           |   5.60|   24.40|  113.40|   451.90|
|     | CDF                            |  42.80|  120.90|  264.60|   641.80|
|     | Genz & Monahan (1999)          |       |        |        |         |
|     | Genz & Monahan (1999) Adaptive |   8.80|   12.80|   22.50|    29.40|
|     | QMC                            |       |        |        |         |
|     | Adaptive QMC                   |  21.40|   37.60|   59.60|    71.80|

**Number of complete cases**

|     |    3|    4|    5|    6|
|-----|----:|----:|----:|----:|
| 2   |  100|  100|  100|  100|
| 4   |  100|  100|  100|  100|
| 8   |  100|  100|  100|  100|
| 16  |  100|  100|  100|  100|
| 32  |  100|  100|  100|  100|

``` r

# show quantiles instead
med_func <- function(x, prob = .75, ...)
  apply(x, 1, function(z) quantile(na.omit(z), probs = prob))
show_run_times(meth = med_func, suffix = " (75% quantile)", na.rm = TRUE)
```

**NAs have been removed. Cells may not be comparable (75% quantile)**

| n   | method/p                       |      3|       4|       5|        6|
|:----|:-------------------------------|------:|-------:|-------:|--------:|
| 2   | AGHQ                           |   1.20|    7.85|   56.80|   424.20|
|     | CDF                            |   1.00|    1.20|    2.45|     3.75|
|     | Genz & Monahan (1999) Adaptive |  12.85|  281.75|  380.50|  1121.05|
|     | Adaptive QMC                   |  37.80|  126.00|  125.10|   236.90|
| 4   | AGHQ                           |   2.20|   14.60|  109.05|   838.25|
|     | CDF                            |   2.40|    6.20|   12.05|    21.50|
|     | Genz & Monahan (1999) Adaptive |   2.65|  165.35|   82.85|   570.40|
|     | Adaptive QMC                   |  51.20|  104.65|   84.25|   148.80|
| 8   | AGHQ                           |   3.80|   28.60|  211.80|  1627.00|
|     | CDF                            |   8.45|   22.05|   35.60|    99.00|
|     | Genz & Monahan (1999) Adaptive |   2.40|    3.40|    5.80|    11.30|
|     | Adaptive QMC                   |  19.00|   40.70|   43.00|    83.05|
| 16  | AGHQ                           |   5.00|   28.20|  172.85|  1081.75|
|     | CDF                            |  22.80|   56.85|  109.55|   273.05|
|     | Genz & Monahan (1999) Adaptive |   4.60|    6.80|   11.60|    17.85|
|     | Adaptive QMC                   |  11.40|   20.00|   31.00|    44.80|
| 32  | AGHQ                           |   6.05|   24.80|  116.25|   454.05|
|     | CDF                            |  57.20|  196.65|  506.95|  1325.90|
|     | Genz & Monahan (1999) Adaptive |   9.20|   13.40|   23.25|    30.20|
|     | Adaptive QMC                   |  21.80|   38.05|   61.00|    72.40|

``` r
show_run_times(meth = med_func, suffix = " (75% quantile)", TRUE)
```

**Only showing complete cases (75% quantile)**

| n   | method/p                       |      3|       4|       5|        6|
|:----|:-------------------------------|------:|-------:|-------:|--------:|
| 2   | GHQ                            |       |        |        |         |
|     | AGHQ                           |   1.20|    7.85|   56.80|   424.20|
|     | CDF                            |   1.00|    1.20|    2.45|     3.75|
|     | Genz & Monahan (1999)          |       |        |        |         |
|     | Genz & Monahan (1999) Adaptive |  12.85|  281.75|  380.50|  1121.05|
|     | QMC                            |       |        |        |         |
|     | Adaptive QMC                   |  37.80|  126.00|  125.10|   236.90|
| 4   | GHQ                            |       |        |        |         |
|     | AGHQ                           |   2.20|   14.60|  109.05|   838.25|
|     | CDF                            |   2.40|    6.20|   12.05|    21.50|
|     | Genz & Monahan (1999)          |       |        |        |         |
|     | Genz & Monahan (1999) Adaptive |   2.65|  165.35|   82.85|   570.40|
|     | QMC                            |       |        |        |         |
|     | Adaptive QMC                   |  51.20|  104.65|   84.25|   148.80|
| 8   | GHQ                            |       |        |        |         |
|     | AGHQ                           |   3.80|   28.60|  211.80|  1627.00|
|     | CDF                            |   8.45|   22.05|   35.60|    99.00|
|     | Genz & Monahan (1999)          |       |        |        |         |
|     | Genz & Monahan (1999) Adaptive |   2.40|    3.40|    5.80|    11.30|
|     | QMC                            |       |        |        |         |
|     | Adaptive QMC                   |  19.00|   40.70|   43.00|    83.05|
| 16  | GHQ                            |       |        |        |         |
|     | AGHQ                           |   5.00|   28.20|  172.85|  1081.75|
|     | CDF                            |  22.80|   56.85|  109.55|   273.05|
|     | Genz & Monahan (1999)          |       |        |        |         |
|     | Genz & Monahan (1999) Adaptive |   4.60|    6.80|   11.60|    17.85|
|     | QMC                            |       |        |        |         |
|     | Adaptive QMC                   |  11.40|   20.00|   31.00|    44.80|
| 32  | GHQ                            |       |        |        |         |
|     | AGHQ                           |   6.05|   24.80|  116.25|   454.05|
|     | CDF                            |  57.20|  196.65|  506.95|  1325.90|
|     | Genz & Monahan (1999)          |       |        |        |         |
|     | Genz & Monahan (1999) Adaptive |   9.20|   13.40|   23.25|    30.20|
|     | QMC                            |       |        |        |         |
|     | Adaptive QMC                   |  21.80|   38.05|   61.00|    72.40|

**Number of complete cases**

|     |    3|    4|    5|    6|
|-----|----:|----:|----:|----:|
| 2   |  100|  100|  100|  100|
| 4   |  100|  100|  100|  100|
| 8   |  100|  100|  100|  100|
| 16  |  100|  100|  100|  100|
| 32  |  100|  100|  100|  100|

``` r

#####
# mean scaled RMSE table
show_scaled_mean_rmse <- function(remove_nas = FALSE, na.rm = FALSE){
  # get mean scaled RMSE for the methods and the configurations pairs
  res <- sapply(ex_output, function(x)
    sapply(x[!names(x) %in% c("n", "p")], `[[`, "vals_n_comp_time", 
           simplify = "array"), 
    simplify = "array")
  err <- res["rel_rmse", , , ]
  
  is_complete <- apply(err, 3, function(x){
    if(remove_nas){
      consider <- !apply(is.na(x), 1L, all)
      apply(!is.na(x[consider, , drop = FALSE]), 2, all)
    } else 
      rep(TRUE, NCOL(x))
  })
  dim(is_complete) <- dim(err)[2:3]
  
  err <- lapply(1:dim(err)[3], function(i){
    x <- err[, , i]
    x[, is_complete[, i]]
  })
  
  err <- sapply(err, rowMeans, na.rm = na.rm) * err_mult
  err[is.nan(err)] <- NA_real_
  err <- err[!apply(err, 1, function(x) all(is.na(x))), ]
  
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

| n   | method/p                       |      3|      4|      5|      6|
|:----|:-------------------------------|------:|------:|------:|------:|
| 2   | AGHQ                           |   6.84|   8.92|   8.41|   8.57|
|     | CDF                            |  19.45|  19.51|  21.59|  20.84|
|     | Genz & Monahan (1999) Adaptive |  17.06|  19.85|  18.37|  21.65|
|     | Adaptive QMC                   |  20.98|  18.21|  19.87|  19.21|
| 4   | AGHQ                           |   7.02|   8.00|   5.81|   6.60|
|     | CDF                            |  20.65|  21.94|  19.40|  17.81|
|     | Genz & Monahan (1999) Adaptive |  13.97|  19.32|  18.75|  18.69|
|     | Adaptive QMC                   |  19.08|  19.28|  19.24|  19.95|
| 8   | AGHQ                           |   7.99|   7.91|   7.19|   7.65|
|     | CDF                            |  20.73|  20.90|  19.91|  20.79|
|     | Genz & Monahan (1999) Adaptive |   7.21|  14.54|  12.26|  17.67|
|     | Adaptive QMC                   |  18.28|  20.67|  19.58|  20.24|
| 16  | AGHQ                           |  13.19|  12.50|  11.73|  11.04|
|     | CDF                            |  20.71|  18.74|  19.18|  18.82|
|     | Genz & Monahan (1999) Adaptive |   3.16|   5.25|   6.28|   8.85|
|     | Adaptive QMC                   |  12.37|  14.16|  12.68|  15.35|
| 32  | AGHQ                           |   8.53|   9.39|  11.98|  13.17|
|     | CDF                            |  19.12|  17.59|  18.75|  17.05|
|     | Genz & Monahan (1999) Adaptive |   1.61|   2.36|   1.76|   2.58|
|     | Adaptive QMC                   |   5.23|   5.88|   5.88|   5.61|

``` r
show_scaled_mean_rmse(na.rm = TRUE)
```

**NAs have been removed. Cells may not be comparable**

| n   | method/p                       |      3|      4|      5|      6|
|:----|:-------------------------------|------:|------:|------:|------:|
| 2   | AGHQ                           |   6.84|   8.92|   8.41|   8.57|
|     | CDF                            |  19.45|  19.51|  21.59|  20.84|
|     | Genz & Monahan (1999) Adaptive |  17.06|  19.85|  18.37|  21.65|
|     | Adaptive QMC                   |  20.98|  18.21|  19.87|  19.21|
| 4   | AGHQ                           |   7.02|   8.00|   5.81|   6.60|
|     | CDF                            |  20.65|  21.94|  19.40|  17.81|
|     | Genz & Monahan (1999) Adaptive |  13.97|  19.32|  18.75|  18.69|
|     | Adaptive QMC                   |  19.08|  19.28|  19.24|  19.95|
| 8   | AGHQ                           |   7.99|   7.91|   7.19|   7.65|
|     | CDF                            |  20.73|  20.90|  19.91|  20.79|
|     | Genz & Monahan (1999) Adaptive |   7.21|  14.54|  12.26|  17.67|
|     | Adaptive QMC                   |  18.28|  20.67|  19.58|  20.24|
| 16  | AGHQ                           |  13.19|  12.50|  11.73|  11.04|
|     | CDF                            |  20.71|  18.74|  19.18|  18.82|
|     | Genz & Monahan (1999) Adaptive |   3.16|   5.25|   6.28|   8.85|
|     | Adaptive QMC                   |  12.37|  14.16|  12.68|  15.35|
| 32  | AGHQ                           |   8.53|   9.39|  11.98|  13.17|
|     | CDF                            |  19.12|  17.59|  18.75|  17.05|
|     | Genz & Monahan (1999) Adaptive |   1.61|   2.36|   1.76|   2.58|
|     | Adaptive QMC                   |   5.23|   5.88|   5.88|   5.61|

``` r
show_scaled_mean_rmse(TRUE)
```

**Only showing complete cases**

| n   | method/p                       |      3|      4|      5|      6|
|:----|:-------------------------------|------:|------:|------:|------:|
| 2   | AGHQ                           |   6.84|   8.92|   8.41|   8.57|
|     | CDF                            |  19.45|  19.51|  21.59|  20.84|
|     | Genz & Monahan (1999) Adaptive |  17.06|  19.85|  18.37|  21.65|
|     | Adaptive QMC                   |  20.98|  18.21|  19.87|  19.21|
| 4   | AGHQ                           |   7.02|   8.00|   5.81|   6.60|
|     | CDF                            |  20.65|  21.94|  19.40|  17.81|
|     | Genz & Monahan (1999) Adaptive |  13.97|  19.32|  18.75|  18.69|
|     | Adaptive QMC                   |  19.08|  19.28|  19.24|  19.95|
| 8   | AGHQ                           |   7.99|   7.91|   7.19|   7.65|
|     | CDF                            |  20.73|  20.90|  19.91|  20.79|
|     | Genz & Monahan (1999) Adaptive |   7.21|  14.54|  12.26|  17.67|
|     | Adaptive QMC                   |  18.28|  20.67|  19.58|  20.24|
| 16  | AGHQ                           |  13.19|  12.50|  11.73|  11.04|
|     | CDF                            |  20.71|  18.74|  19.18|  18.82|
|     | Genz & Monahan (1999) Adaptive |   3.16|   5.25|   6.28|   8.85|
|     | Adaptive QMC                   |  12.37|  14.16|  12.68|  15.35|
| 32  | AGHQ                           |   8.53|   9.39|  11.98|  13.17|
|     | CDF                            |  19.12|  17.59|  18.75|  17.05|
|     | Genz & Monahan (1999) Adaptive |   1.61|   2.36|   1.76|   2.58|
|     | Adaptive QMC                   |   5.23|   5.88|   5.88|   5.61|

**Number of complete cases**

|     |    3|    4|    5|    6|
|-----|----:|----:|----:|----:|
| 2   |  100|  100|  100|  100|
| 4   |  100|  100|  100|  100|
| 8   |  100|  100|  100|  100|
| 16  |  100|  100|  100|  100|
| 32  |  100|  100|  100|  100|

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

| n   | quantile/p |    3|    4|    5|    6|
|:----|:-----------|----:|----:|----:|----:|

**Number of complete cases**

|     |    3|    4|    5|    6|
|-----|----:|----:|----:|----:|
| 2   |    0|    0|    0|    0|
| 4   |    0|    0|    0|    0|
| 8   |    0|    0|    0|    0|
| 16  |    0|    0|    0|    0|
| 32  |    0|    0|    0|    0|

``` r
show_n_nodes(TRUE)
```

**Only showing complete cases (Adaptive GHQ)**

| n   | quantile/p |     3|     4|     5|     6|
|:----|:-----------|-----:|-----:|-----:|-----:|
| 2   | 0%         |  5.00|  5.00|  5.00|  5.00|
|     | 25%        |  6.00|  6.00|  6.00|  6.00|
|     | 50%        |  6.00|  6.00|  6.00|  6.00|
|     | 75%        |  6.00|  6.00|  6.00|  6.00|
|     | 100%       |  7.00|  7.00|  7.00|  7.00|
| 4   | 0%         |  5.00|  4.00|  4.00|  4.00|
|     | 25%        |  6.00|  6.00|  6.00|  6.00|
|     | 50%        |  6.00|  6.00|  6.00|  6.00|
|     | 75%        |  6.00|  6.00|  6.00|  6.00|
|     | 100%       |  7.00|  7.00|  7.00|  7.00|
| 8   | 0%         |  4.00|  4.00|  4.00|  4.00|
|     | 25%        |  5.00|  5.00|  5.00|  5.00|
|     | 50%        |  6.00|  6.00|  6.00|  6.00|
|     | 75%        |  6.00|  6.00|  6.00|  6.00|
|     | 100%       |  7.00|  7.00|  7.00|  7.00|
| 16  | 0%         |  4.00|  4.00|  4.00|  4.00|
|     | 25%        |  4.00|  4.00|  5.00|  5.00|
|     | 50%        |  5.00|  5.00|  5.00|  5.00|
|     | 75%        |  5.00|  5.00|  5.00|  5.00|
|     | 100%       |  6.00|  6.00|  6.00|  6.00|
| 32  | 0%         |  4.00|  4.00|  4.00|  4.00|
|     | 25%        |  4.00|  4.00|  4.00|  4.00|
|     | 50%        |  4.00|  4.00|  4.00|  4.00|
|     | 75%        |  4.00|  4.00|  4.00|  4.00|
|     | 100%       |  6.00|  5.00|  4.00|  5.00|

**Number of complete cases**

|     |    3|    4|    5|    6|
|-----|----:|----:|----:|----:|
| 2   |  100|  100|  100|  100|
| 4   |  100|  100|  100|  100|
| 8   |  100|  100|  100|  100|
| 16  |  100|  100|  100|  100|
| 32  |  100|  100|  100|  100|

### Approximating the Inner Integral

The integrand with multinomial outcomes is intractable and requires an
approximation. To be more precise, we need an approximation of

![
\\begin{align\*}
h(\\vec u) &= \\int \\phi(a)\\prod\_{k = 1}^c
   \\Phi\\left(a + \\eta\_k + \\vec z\_k^\\top\\vec u\\right) du \\\\
&=\\int \\phi(a)\\prod\_{k = 1}^c
   \\Phi\\left(\\eta\_k (a, \\vec u)\\right) du
\\end{align\*}
](https://latex.codecogs.com/svg.latex?%0A%5Cbegin%7Balign%2A%7D%0Ah%28%5Cvec%20u%29%20%26%3D%20%5Cint%20%5Cphi%28a%29%5Cprod_%7Bk%20%3D%201%7D%5Ec%0A%20%20%20%5CPhi%5Cleft%28a%20%2B%20%5Ceta_k%20%2B%20%5Cvec%20z_k%5E%5Ctop%5Cvec%20u%5Cright%29%20du%20%5C%5C%0A%26%3D%5Cint%20%5Cphi%28a%29%5Cprod_%7Bk%20%3D%201%7D%5Ec%0A%20%20%20%5CPhi%5Cleft%28%5Ceta_k%20%28a%2C%20%5Cvec%20u%29%5Cright%29%20du%0A%5Cend%7Balign%2A%7D%0A "
\begin{align*}
h(\vec u) &= \int \phi(a)\prod_{k = 1}^c
   \Phi\left(a + \eta_k + \vec z_k^\top\vec u\right) du \\
&=\int \phi(a)\prod_{k = 1}^c
   \Phi\left(\eta_k (a, \vec u)\right) du
\end{align*}
")

with
![\\eta\_k (a, \\vec u) = a + \\eta\_k + \\vec z\_k^\\top\\vec u](https://latex.codecogs.com/svg.latex?%5Ceta_k%20%28a%2C%20%5Cvec%20u%29%20%3D%20a%20%2B%20%5Ceta_k%20%2B%20%5Cvec%20z_k%5E%5Ctop%5Cvec%20u "\eta_k (a, \vec u) = a + \eta_k + \vec z_k^\top\vec u").
Moreover, we need an approximations of the gradient and Hessian with
respect to
![\\vec u](https://latex.codecogs.com/svg.latex?%5Cvec%20u "\vec u") of
![\\log h(\\vec u)](https://latex.codecogs.com/svg.latex?%5Clog%20h%28%5Cvec%20u%29 "\log h(\vec u)").
We can easily compute the these if we have an approximations of the
gradient and Hessian with respect to
![x\_k = \\vec z\_k^\\top\\vec u](https://latex.codecogs.com/svg.latex?x_k%20%3D%20%5Cvec%20z_k%5E%5Ctop%5Cvec%20u "x_k = \vec z_k^\top\vec u").
Let
![e\_k(a) = \\eta\_k(a, \\vec u)](https://latex.codecogs.com/svg.latex?e_k%28a%29%20%3D%20%5Ceta_k%28a%2C%20%5Cvec%20u%29 "e_k(a) = \eta_k(a, \vec u)")
which implicitly depends on a given value of
![\\vec u](https://latex.codecogs.com/svg.latex?%5Cvec%20u "\vec u").
Then the latter derivatives are

![
\\begin{align\*}
\\frac{\\partial}{\\partial x\_i} \\log h &=
  \\frac{
  \\int \\phi(a)\\phi(e\_i(a))
  \\prod\_{k\\ne i}\\Phi(e\_k(a)) da}{
  \\int \\phi(a)\\prod\_{k = 1}^c\\Phi(e\_{k}(a)) da} \\\\
\\frac{\\partial^2}{\\partial x\_i^2}\\log h &=
 -\\bigg(
 \\left\[\\int \\phi(a)e\_i(a)\\phi(e\_{i}(a))
 \\prod\_{k\\ne i}\\Phi(e\_{k}(a))da\\right\]
 \\left\[\\int \\phi(a)\\prod\_{k = 1}^c\\Phi(e\_k(a)) da\\right\]\\\\
 &\\hspace{20pt}
 +\\left\[\\int \\phi(a)\\phi(e\_i(a))
 \\prod\_{k\\ne i}\\Phi(e\_k(a))da\\right\]^2\\bigg) \\\\
 &\\hspace{20pt}\\bigg/\\left\[\\int \\phi(a)
 \\prod\_{k = 1}^c\\Phi(e\_k(a)) da\\right\]^2 \\\\
\\frac{\\partial^2}{\\partial x\_i\\partial x\_j}\\log h &=
 \\bigg(
 \\left\[\\int \\phi(a)\\prod\_{k = 1}^c\\Phi(e\_k(a)) da\\right\]
 \\left\[\\int \\phi(a)\\phi(e\_i(a))\\phi(e\_j(a))
 \\prod\_{k\\ne i,j}\\Phi(e\_k(a))da\\right\] \\\\
 &\\hspace{20pt} -
 \\left\[\\int \\phi(a)\\phi(e\_i(a))
 \\prod\_{k\\ne i}\\Phi(e\_k(a))da\\right\]
 \\left\[\\int \\phi(a)\\phi(e\_j(a))
 \\prod\_{k\\ne j}\\Phi(e\_k(a))da\\right\]\\bigg) \\\\
 &\\hspace{20pt}\\bigg/
 \\left\[\\int \\phi(a)\\prod\_{k = 1}^c\\Phi(e\_k(a)) da\\right\]^2
\\end{align\*}
](https://latex.codecogs.com/svg.latex?%0A%5Cbegin%7Balign%2A%7D%0A%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20x_i%7D%20%5Clog%20h%20%26%3D%0A%20%20%5Cfrac%7B%0A%20%20%5Cint%20%5Cphi%28a%29%5Cphi%28e_i%28a%29%29%0A%20%20%5Cprod_%7Bk%5Cne%20i%7D%5CPhi%28e_k%28a%29%29%20da%7D%7B%0A%20%20%5Cint%20%5Cphi%28a%29%5Cprod_%7Bk%20%3D%201%7D%5Ec%5CPhi%28e_%7Bk%7D%28a%29%29%20da%7D%20%5C%5C%0A%5Cfrac%7B%5Cpartial%5E2%7D%7B%5Cpartial%20x_i%5E2%7D%5Clog%20h%20%26%3D%0A%20-%5Cbigg%28%0A%20%5Cleft%5B%5Cint%20%5Cphi%28a%29e_i%28a%29%5Cphi%28e_%7Bi%7D%28a%29%29%0A%20%5Cprod_%7Bk%5Cne%20i%7D%5CPhi%28e_%7Bk%7D%28a%29%29da%5Cright%5D%0A%20%5Cleft%5B%5Cint%20%5Cphi%28a%29%5Cprod_%7Bk%20%3D%201%7D%5Ec%5CPhi%28e_k%28a%29%29%20da%5Cright%5D%5C%5C%0A%20%26%5Chspace%7B20pt%7D%0A%20%2B%5Cleft%5B%5Cint%20%5Cphi%28a%29%5Cphi%28e_i%28a%29%29%0A%20%5Cprod_%7Bk%5Cne%20i%7D%5CPhi%28e_k%28a%29%29da%5Cright%5D%5E2%5Cbigg%29%20%5C%5C%0A%20%26%5Chspace%7B20pt%7D%5Cbigg%2F%5Cleft%5B%5Cint%20%5Cphi%28a%29%0A%20%5Cprod_%7Bk%20%3D%201%7D%5Ec%5CPhi%28e_k%28a%29%29%20da%5Cright%5D%5E2%20%5C%5C%0A%5Cfrac%7B%5Cpartial%5E2%7D%7B%5Cpartial%20x_i%5Cpartial%20x_j%7D%5Clog%20h%20%26%3D%0A%20%5Cbigg%28%0A%20%5Cleft%5B%5Cint%20%5Cphi%28a%29%5Cprod_%7Bk%20%3D%201%7D%5Ec%5CPhi%28e_k%28a%29%29%20da%5Cright%5D%0A%20%5Cleft%5B%5Cint%20%5Cphi%28a%29%5Cphi%28e_i%28a%29%29%5Cphi%28e_j%28a%29%29%0A%20%5Cprod_%7Bk%5Cne%20i%2Cj%7D%5CPhi%28e_k%28a%29%29da%5Cright%5D%20%5C%5C%0A%20%26%5Chspace%7B20pt%7D%20-%0A%20%5Cleft%5B%5Cint%20%5Cphi%28a%29%5Cphi%28e_i%28a%29%29%0A%20%5Cprod_%7Bk%5Cne%20i%7D%5CPhi%28e_k%28a%29%29da%5Cright%5D%0A%20%5Cleft%5B%5Cint%20%5Cphi%28a%29%5Cphi%28e_j%28a%29%29%0A%20%5Cprod_%7Bk%5Cne%20j%7D%5CPhi%28e_k%28a%29%29da%5Cright%5D%5Cbigg%29%20%5C%5C%0A%20%26%5Chspace%7B20pt%7D%5Cbigg%2F%0A%20%5Cleft%5B%5Cint%20%5Cphi%28a%29%5Cprod_%7Bk%20%3D%201%7D%5Ec%5CPhi%28e_k%28a%29%29%20da%5Cright%5D%5E2%0A%5Cend%7Balign%2A%7D%0A "
\begin{align*}
\frac{\partial}{\partial x_i} \log h &=
  \frac{
  \int \phi(a)\phi(e_i(a))
  \prod_{k\ne i}\Phi(e_k(a)) da}{
  \int \phi(a)\prod_{k = 1}^c\Phi(e_{k}(a)) da} \\
\frac{\partial^2}{\partial x_i^2}\log h &=
 -\bigg(
 \left[\int \phi(a)e_i(a)\phi(e_{i}(a))
 \prod_{k\ne i}\Phi(e_{k}(a))da\right]
 \left[\int \phi(a)\prod_{k = 1}^c\Phi(e_k(a)) da\right]\\
 &\hspace{20pt}
 +\left[\int \phi(a)\phi(e_i(a))
 \prod_{k\ne i}\Phi(e_k(a))da\right]^2\bigg) \\
 &\hspace{20pt}\bigg/\left[\int \phi(a)
 \prod_{k = 1}^c\Phi(e_k(a)) da\right]^2 \\
\frac{\partial^2}{\partial x_i\partial x_j}\log h &=
 \bigg(
 \left[\int \phi(a)\prod_{k = 1}^c\Phi(e_k(a)) da\right]
 \left[\int \phi(a)\phi(e_i(a))\phi(e_j(a))
 \prod_{k\ne i,j}\Phi(e_k(a))da\right] \\
 &\hspace{20pt} -
 \left[\int \phi(a)\phi(e_i(a))
 \prod_{k\ne i}\Phi(e_k(a))da\right]
 \left[\int \phi(a)\phi(e_j(a))
 \prod_{k\ne j}\Phi(e_k(a))da\right]\bigg) \\
 &\hspace{20pt}\bigg/
 \left[\int \phi(a)\prod_{k = 1}^c\Phi(e_k(a)) da\right]^2
\end{align*}
")

This requires an approximation of four different types of integrals and
is what we have implemented. Below, we consider an approximation
![h(\\vec u)](https://latex.codecogs.com/svg.latex?h%28%5Cvec%20u%29 "h(\vec u)").
We have implemented both an adaptive and non-adaptive version of GHQ.
Thus, we interested in comparing which version is fastest and a high
precision.

``` r
# define function to get test data for a given number of alternative 
# groups  
get_ex_data <- function(n_alt){
  Z <- Sigma <- diag(1., n_alt)
  Sigma[lower.tri(Sigma)] <- Sigma[upper.tri(Sigma)] <- -.1
  eta <- seq(-1, 1, length.out = n_alt)
  
  list(Z = Z, Sigma = Sigma, eta = eta)
}

# use the data to assign two functions to approximate the "inner" integral
dat <- get_ex_data(3L)
get_aprx_ghq <- function(dat, is_adaptive, u)
  function(n_nodes, n_times = 1L, order = 0L) 
    with(dat, drop(mixprobit:::multinomial_inner_integral(
      Z = Z, eta = eta, Sigma = Sigma, n_nodes = n_nodes, 
      is_adaptive = is_adaptive, n_times = n_times, u = u, 
      order = order)))

set.seed(1)
u <- drop(mvtnorm::rmvnorm(1L, sigma = dat$Sigma))
# the adaptive version 
adap   <- get_aprx_ghq(dat, TRUE , u)
# the non-adaptive version
n_adap <- get_aprx_ghq(dat, FALSE, u)
adap  (10L)
#> [1] 0.2352
n_adap(10L)
#> [1] 0.2352

# plot one example (open circle: AGHQ; filled circle: GHQ)
ns <- 3:30
par(mar = c(5, 5, 1, 1), mfcol = c(1, 2))
vals <- cbind(sapply(ns, adap), sapply(ns, n_adap))
matplot(ns[1:7], vals[1:7, ], pch = c(1, 16), col = "black",
        xlab = "Number of nodes", ylab = "Integral aprx.")
abline(h = tail(vals, 1L)[, 1], lty = 3)
matplot(ns[-(1:7)], vals[-(1:7), ], pch = c(1, 16), col = "black",
        xlab = "Number of nodes", ylab = "Integral aprx.")
abline(h = tail(vals, 1L)[, 1], lty = 3)
```

![](man/figures/README-aprx_mult_inner-1.png)

``` r

# compare approximation time
microbenchmark::microbenchmark(
  `AGHQ 3`  = adap  (3L , n_times = 1000L),
  `AGHQ 7`  = adap  (7L , n_times = 1000L), 
  ` GHQ 3`  = n_adap(3L , n_times = 1000L), 
  ` GHQ 7`  = n_adap(7L , n_times = 1000L),
  ` GHQ 21` = n_adap(21L, n_times = 1000L))
#> Unit: microseconds
#>     expr    min     lq   mean median     uq    max neval
#>   AGHQ 3  892.3  927.4  940.7  929.5  933.6 1200.3   100
#>   AGHQ 7 1730.2 1773.8 1795.4 1778.7 1784.3 2114.5   100
#>    GHQ 3  604.2  617.9  625.2  622.3  623.4  754.4   100
#>    GHQ 7 1396.7 1432.3 1449.0 1435.0 1437.5 1747.4   100
#>   GHQ 21 4156.2 4256.3 4282.6 4261.9 4283.3 4600.5   100
```

The adaptive version is much more precise. Moreover, the it seems that 5
nodes is about sufficient. As of this writing, it takes about 1.9
milliseconds to do 1000 evaluations of the integrand. This implies about
1.9 microseconds per integrand evaluation which, unfortunately, will add
when we have to marginalize over the random effects,
![\\vec u](https://latex.codecogs.com/svg.latex?%5Cvec%20u "\vec u").

Similar to what we do above, we consider approximating the gradient and
Hessian of
![\\log h(\\vec u)](https://latex.codecogs.com/svg.latex?%5Clog%20h%28%5Cvec%20u%29 "\log h(\vec u)")
with respect to
![\\vec u](https://latex.codecogs.com/svg.latex?%5Cvec%20u "\vec u")
below.

``` r
#####
# the gradient
adap  (10L, order = 1L)
#> [1] -0.47024 -0.47285 -0.02483
n_adap(10L, order = 1L)
#> [1] -0.47049 -0.47313 -0.02473

# check precision. We plot the errors now with the black being the 
# adaptive version and gray being the non-adaptive version
va <- t(sapply(ns,   adap, order = 1L))
vn <- t(sapply(ns, n_adap, order = 1L))
est <- rep(drop(tail(va, 1)), each = length(ns))
va <- va - est
vn <- vn - est
matplot(
  ns[1:10], cbind(va, vn)[1:10, ], pch = rep(as.character(1:NCOL(va)), 2), 
  xlab = "Number of nodes", ylab = "Gradient aprx. (error)", 
  col = rep(c("black", "darkgray"), each = NCOL(va)), type = "b", 
  lty = rep(c(1, 2), each = NCOL(va)))
```

![](man/figures/README-grads_aprx_mult_inner-1.png)

``` r

# compare approximation time
microbenchmark::microbenchmark(
  `AGHQ 3`  = adap  (3L , n_times = 1000L, order = 1L),
  `AGHQ 7`  = adap  (7L , n_times = 1000L, order = 1L), 
  ` GHQ 3`  = n_adap(3L , n_times = 1000L, order = 1L), 
  ` GHQ 7`  = n_adap(7L , n_times = 1000L, order = 1L),
  ` GHQ 21` = n_adap(21L, n_times = 1000L, order = 1L))
#> Unit: microseconds
#>     expr    min     lq   mean median     uq    max neval
#>   AGHQ 3 1200.8 1232.4 1234.7 1234.3 1236.3 1305.7   100
#>   AGHQ 7 2395.9 2456.5 2459.0 2459.5 2464.0 2553.9   100
#>    GHQ 3  858.3  880.7  884.3  881.9  884.2  984.9   100
#>    GHQ 7 2014.4 2065.0 2073.9 2070.4 2075.0 2283.1   100
#>   GHQ 21 6068.0 6216.4 6230.2 6221.2 6231.5 6513.6   100
```

``` r
#####
# the Hessian
adap  (10L, order = 2L)
#>          [,1]     [,2]     [,3]
#> [1,] -0.38453  0.13987  0.01607
#> [2,]  0.13987 -0.34386  0.01722
#> [3,]  0.01607  0.01722 -0.04677
n_adap(10L, order = 2L)
#>          [,1]     [,2]     [,3]
#> [1,] -0.38482  0.13960  0.01599
#> [2,]  0.13960 -0.34424  0.01714
#> [3,]  0.01599  0.01714 -0.04666

# check precision. We plot the errors now with the black being the 
# adaptive version and gray being the non-adaptive version
va <- t(sapply(ns, adap, order = 2L))
vn <- t(sapply(ns, n_adap, order = 2L))
keep <- which(lower.tri(matrix(nc = 3, nr = 3), diag = TRUE))
va <- va[, keep]
vn <- vn[, keep]
est <- rep(drop(tail(va, 1)), each = length(ns))
va <- va - est
vn <- vn - est
matplot(
  ns[1:10], cbind(va, vn)[1:10, ], pch = rep(as.character(1:NCOL(va)), 2), 
  xlab = "Number of nodes", ylab = "Hessian aprx. (error)", 
  col = rep(c("black", "darkgray"), each = NCOL(va)), type = "b", 
  lty = rep(c(1, 2), each = NCOL(va)))
```

![](man/figures/README-hess_aprx_mult_inner-1.png)

``` r

# compare approximation time
microbenchmark::microbenchmark(
  `AGHQ 3`  = adap  (3L , n_times = 1000L, order = 2L),
  `AGHQ 7`  = adap  (7L , n_times = 1000L, order = 2L), 
  ` GHQ 3`  = n_adap(3L , n_times = 1000L, order = 2L), 
  ` GHQ 7`  = n_adap(7L , n_times = 1000L, order = 2L),
  ` GHQ 21` = n_adap(21L, n_times = 1000L, order = 2L))
#> Unit: microseconds
#>     expr     min      lq    mean  median      uq   max neval
#>   AGHQ 3  1195.5  1241.1  1255.6  1243.5  1261.8  1412   100
#>   AGHQ 7  2405.2  2456.8  2493.5  2461.9  2504.0  2837   100
#>    GHQ 3   905.8   929.5   940.8   931.7   937.8  1172   100
#>    GHQ 7  2009.4  2060.4  2086.6  2063.5  2083.0  2363   100
#>   GHQ 21 15607.2 15947.1 16122.0 16036.8 16227.4 16916   100
```

It does not take much more time and using an adaptive method only seems
more attractive as the overhead from finding a mode is relatively
smaller.

As another example, we use the simulation function we defined before and
compute the average absolute error using a number of data sets and the
computation where we fix the covariance matrix.

<!-- knitr::opts_knit$set(output.dir = ".") -->
<!-- knitr::load_cache("rig_aprx_mult_inner", path = "README_cache/markdown_github/") -->
``` r
sim_one_integrand_aprx <- function(n, p, n_nodes){ 
  Sigma <- diag(p)
  dat <- get_sim_dat(n, p, Sigma = Sigma)
  dat$Sigma <- Sigma
  n_times <- 10000L
  
  . <- function(n){
    ti <- system.time(
      out <- func(n_nodes = n, n_times = n_times, order = 0L))
    c(Estimate = out, Time = unname(ti["elapsed"]) * 1000L / n_times)
  }
  
  func <- get_aprx_ghq(dat = dat, is_adaptive = TRUE, u = dat$u)
  ada <- sapply(n_nodes, .)
  func <- get_aprx_ghq(dat = dat, is_adaptive = FALSE, u = dat$u)
  n_ada <- sapply(n_nodes, .)
  
  truth <- ada["Estimate", which.max(n_nodes)]
  . <- function(x)
    rbind(x, Error = x["Estimate", ] - truth)
  
  ada <- .(ada)
  n_ada <- .(n_ada)
  array(c(ada, n_ada), dim = c(dim(ada), 2L), 
        dimnames = list(
          Entity = rownames(ada), 
          Property = n_nodes, 
          Method = c("Adaptive", "Non-Adaptive")))
} 

n_nodes <- c(c(seq(2L, 20L, by = 2L)), 30L)
n_sim <- 25L
gr <- expand.grid(n = 2^(1:4), p = 2:4)

set.seed(1L)
integrand_dat <- mapply(function(n, p){
  all_dat <- replicate(
    n_sim, sim_one_integrand_aprx(n = n, p = p, n_nodes = n_nodes), 
    simplify = "array")
  out <- all_dat
  out["Error", , , ] <- abs(out["Error", , , ])
  sd_est <- apply(out["Error", , , ], 1:2, sd) / sqrt(n_sim)
  out <- apply(out, 1:3, mean)
  library(abind)
  
  out <- abind(out, sd_est, along = 1)
  dimnames(out)[[1]][3:4] <- c("Abs(error)",  "Sd abs(error)")
  
  list(n = n, p = p, all_dat = all_dat, out = aperm(out, 3:1))
}, n = gr$n, p = gr$p, SIMPLIFY = FALSE)
```

We plot the average evaluation time and error below using a different
number of nodes for different cluster sizes, number of clusters and
random effects.

``` r
to_print <- lapply(integrand_dat, `[[`, "out")
names(to_print) <- lapply(integrand_dat, function(x)
  sprintf("n: %2d; p: %2d", x$n, x$p))

do_plot <- function(what, do_abline = FALSE, n_choice){
  dat <- sapply(to_print, function(x) x[, , what], simplify = "array")
  ns <- sapply(integrand_dat, `[[`, "n")
  ps <- sapply(integrand_dat, `[[`, "p")
  
  keep <- ns %in% n_choice
  ns <- ns[keep]
  ps <- ps[keep]
  dat <- dat[, , keep]
  
  col <- gray.colors(length(unique(ps)), start = .5, end = 0)[
    as.integer(factor(ps))]
  lty <- as.integer(factor(ns))
  
  if(min(dat) <= 0)
    dat <- pmax(dat, min(dat[dat != 0]))
  plot(1, 1, xlim = range(n_nodes), ylim = range(dat), ylab = what, 
       xlab = "# nodes", bty = "l", log = "y")
  for(i in seq_len(dim(dat)[[3]])){
    matlines(n_nodes, t(dat[, , i]), lty = lty[i], col = col[i])
    matpoints(n_nodes, t(dat[, , i]), pch = c(1, 16) + (i %% 3), 
              col = col[i])
  }
  if(do_abline)
    for(x in 6:16)
      abline(h = 10^(-x), lty = 3)
}

par(mfcol = c(2, 2), mar = c(5, 5, 1, 1))
for(n_choice in unique(gr$n)){
  cat(sprintf("n_choice %2d\n", n_choice))
  do_plot("Abs(error)", do_abline = TRUE, n_choice = n_choice)
  do_plot("Time", n_choice = n_choice)
  do_plot("Sd abs(error)", do_abline = TRUE, n_choice = n_choice)
  do_plot("Estimate", n_choice = n_choice)
}
#> n_choice  2
```

![](man/figures/README-show_rig_aprx_mult_inner_res-1.png)

    #> n_choice  4

![](man/figures/README-show_rig_aprx_mult_inner_res-2.png)

    #> n_choice  8

![](man/figures/README-show_rig_aprx_mult_inner_res-3.png)

    #> n_choice 16

![](man/figures/README-show_rig_aprx_mult_inner_res-4.png)

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
