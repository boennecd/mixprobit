Mixed Models with Probit Link
=============================

We make a comparison below of making an approximation of a marignal log-likelihood term that is typical in many mixed effect models with a probit link funciton.

TODO: make a better description.

``` r
options(digits = 4)
set.seed(2)

#####
# parameters to change
n <- 10L                             # cluster size
p <- 4L                              # number of random effects
b <- 30L                             # number of nodes to use with Gaussian
                                     # Hermite quadrature
maxpts <- p * 10000L                 # factor to set the (maximum) number of
                                     # evaluations of  the integrand with
                                     # the other methods

#####
# variables used in simulation
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

#####
# use Gaussian Hermite quadrature (GHQ)
library(fastGHQuad)
rule <- fastGHQuad::gaussHermiteData(b)
f <- function(x)
  sum(mapply(pnorm, q = eta + sqrt(2) * drop(x %*% S_chol %*% Z),
             lower.tail = y, log.p = TRUE))

idx <- do.call(expand.grid, replicate(p, 1:b, simplify = FALSE))

xs <- local({
  args <- list(FUN = c, SIMPLIFY = FALSE)
  do.call(mapply, c(args, lapply(idx, function(i) rule$x[i])))
})
ws_log <- local({
  args <- list(FUN = prod)
  log(do.call(mapply, c(args, lapply(idx, function(i) rule$w[i]))))
})

# function that makes the approximation
f1 <- function()
  sum(exp(ws_log + vapply(xs, f, numeric(1L)))) / pi^(p / 2)

f  <- compiler::cmpfun(f)
f1 <- compiler::cmpfun(f1)

# same function but written in C++
invisible(mixprobit:::set_GH_rule_cached(b))
f1_cpp <- function()
  mixprobit:::aprx_binary_mix_ghq(y = y, eta = eta, Z = Z, Sigma = S,
                                  b = b)

#####
# function that returns the CDF approximation like in Pawitan et al. (2004)
library(mvtnorm)
f2 <- function(){
  dum_vec <- ifelse(y, 1, -1)
  Z_tilde <- Z * rep(dum_vec, each = p)
  SMat <- crossprod(Z_tilde , S %*% Z_tilde)
  diag(SMat) <- diag(SMat) + 1
  pmvnorm(upper = dum_vec * eta, mean = rep(0, n), sigma = SMat,
          algorithm = GenzBretz(maxpts = maxpts, abseps = 1e-5))
}
f2 <- compiler::cmpfun(f2)

# same function but written in C++
f2_cpp <- function(mxpts = maxpts, abseps = 1e-5)
  mixprobit:::aprx_binary_mix_cdf(
    y = y, eta = eta, Z = Z, Sigma = S, maxpts = mxpts,
    abseps = abseps, releps = -1)

#####
# use method from Genz & Monahan (1998)
f3 <- function(key, mxpts = 3L * maxpts, abseps = 1e-5)
  mixprobit:::aprx_binary_mix(
    y = y, eta = eta, Z = Z, Sigma = S,
    mxvals = mxpts, key = key, epsabs = abseps, epsrel = -1)

#####
# compare results. Start with the simulation based methods with a lot of
# samples
capital_T_truth_maybe1 <- f2_cpp(mxpts = 1e7, abseps = 1e-11)
capital_T_truth_maybe2 <- f3(key = 2L, mxpts = 1e7, abseps = 1e-11)
dput(capital_T_truth_maybe1)
#> structure(0.00928768943662851, inform = 1L, error = 1.26643795888541e-08)
dput(capital_T_truth_maybe2)
#> structure(0.0092893494603688, error = 2.76532166158425e-06, inform = 1L, inivls = 9999991L)
all.equal(c(capital_T_truth_maybe1), c(capital_T_truth_maybe2))
#> [1] "Mean relative difference: 0.0001787"
capital_T_truth_maybe <- c(capital_T_truth_maybe1)

# compare with using fewer samples and GHQ
all.equal(capital_T_truth_maybe,   f1())
#> [1] "Mean relative difference: 0.0007289"
all.equal(capital_T_truth_maybe,   f1_cpp())
#> [1] "Mean relative difference: 0.0007289"
all.equal(capital_T_truth_maybe, c(f2()))
#> [1] "Mean relative difference: 1.097e-05"
all.equal(capital_T_truth_maybe, c(f2_cpp()))
#> [1] "Mean relative difference: 5.156e-05"
all.equal(capital_T_truth_maybe, c(f3(1L)))
#> [1] "Mean relative difference: 0.005451"
all.equal(capital_T_truth_maybe, c(f3(2L)))
#> [1] "Mean relative difference: 0.001244"
all.equal(capital_T_truth_maybe, c(f3(3L)))
#> [1] "Mean relative difference: 0.0006818"
all.equal(capital_T_truth_maybe, c(f3(4L)))
#> [1] "Mean relative difference: 0.00296"

# compare computations times
system.time(f1()) # way too slow (seconds!). Use C++ method instead
#>    user  system elapsed 
#>   21.16    0.00   21.16
microbenchmark::microbenchmark(
  `GHQ (C++)` = f1_cpp(),
  `CDF` = f2(), `CDF (C++)` = f2_cpp(),
  `Genz & Monahan (1)` = f3(1L), `Genz & Monahan (2)` = f3(2L),
  `Genz & Monahan (3)` = f3(3L), `Genz & Monahan (4)` = f3(4L),
  times = 10)
#> Unit: milliseconds
#>                expr    min     lq   mean median     uq    max neval
#>           GHQ (C++) 627.68 629.58 632.25 631.58 635.44 636.76    10
#>                 CDF  20.86  20.95  20.97  20.97  20.98  21.13    10
#>           CDF (C++)  20.32  20.35  20.44  20.37  20.52  20.67    10
#>  Genz & Monahan (1)  92.92  93.36  94.35  94.19  95.13  97.50    10
#>  Genz & Monahan (2)  92.55  92.98  95.05  93.70  94.73 106.79    10
#>  Genz & Monahan (3)  88.45  89.40  90.15  89.76  91.23  92.61    10
#>  Genz & Monahan (4)  87.65  87.78  89.73  88.14  89.30 101.37    10
```

References
----------

Genz, Alan, and Frank Bretz. 2002. “Comparison of Methods for the Computation of Multivariate T Probabilities.” *Journal of Computational and Graphical Statistics* 11 (4). Taylor & Francis: 950–71. doi:[10.1198/106186002394](https://doi.org/10.1198/106186002394).

Genz, Alan., and John. Monahan. 1998. “Stochastic Integration Rules for Infinite Regions.” *SIAM Journal on Scientific Computing* 19 (2): 426–39. doi:[10.1137/S1064827595286803](https://doi.org/10.1137/S1064827595286803).

Liu, Qing, and Donald A. Pierce. 1994. “A Note on Gauss-Hermite Quadrature.” *Biometrika* 81 (3). \[Oxford University Press, Biometrika Trust\]: 624–29. <http://www.jstor.org/stable/2337136>.

Ochi, Y., and Ross L. Prentice. 1984. “Likelihood Inference in a Correlated Probit Regression Model.” *Biometrika* 71 (3). \[Oxford University Press, Biometrika Trust\]: 531–43. <http://www.jstor.org/stable/2336562>.

Pawitan, Y., M. Reilly, E. Nilsson, S. Cnattingius, and P. Lichtenstein. 2004. “Estimation of Genetic and Environmental Factors for Binary Traits Using Family Data.” *Statistics in Medicine* 23 (3): 449–65. doi:[10.1002/sim.1603](https://doi.org/10.1002/sim.1603).
