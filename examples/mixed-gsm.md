# Mixed Generalized Survival Models

We simulate at estimate models from

![
\\begin{align\*}
-\\Phi^{-1}(S(t\\mid \\vec x\_{ij}, \\vec z\_{ij}, \\vec u_i)) 
  &= \\vec x\_{ij}(t)^\\top\\vec\\beta + \\vec z\_{ij}^\\top\\vec u_i \\\\
\\vec U_i &\\sim N(\\vec 0, \\Sigma) 
\\end{align\*}
](https://latex.codecogs.com/svg.latex?%0A%5Cbegin%7Balign%2A%7D%0A-%5CPhi%5E%7B-1%7D%28S%28t%5Cmid%20%5Cvec%20x_%7Bij%7D%2C%20%5Cvec%20z_%7Bij%7D%2C%20%5Cvec%20u_i%29%29%20%0A%20%20%26%3D%20%5Cvec%20x_%7Bij%7D%28t%29%5E%5Ctop%5Cvec%5Cbeta%20%2B%20%5Cvec%20z_%7Bij%7D%5E%5Ctop%5Cvec%20u_i%20%5C%5C%0A%5Cvec%20U_i%20%26%5Csim%20N%28%5Cvec%200%2C%20%5CSigma%29%20%0A%5Cend%7Balign%2A%7D%0A "
\begin{align*}
-\Phi^{-1}(S(t\mid \vec x_{ij}, \vec z_{ij}, \vec u_i)) 
  &= \vec x_{ij}(t)^\top\vec\beta + \vec z_{ij}^\top\vec u_i \\
\vec U_i &\sim N(\vec 0, \Sigma) 
\end{align*}
")

subject to independent censoring. The conditional hazard of the model is

![
h(t\\mid  \\vec x\_{ij}, \\vec z\_{ij}, \\vec u_i) = 
  \\vec x\_{ij}'(t)^\\top\\vec\\beta
  \\frac{\\phi(-\\vec x\_{ij}(t)^\\top\\vec\\beta - \\vec z\_{ij}^\\top\\vec u_i)}
       {\\Phi(-\\vec x\_{ij}(t)^\\top\\vec\\beta - \\vec z\_{ij}^\\top\\vec u_i)}
](https://latex.codecogs.com/svg.latex?%0Ah%28t%5Cmid%20%20%5Cvec%20x_%7Bij%7D%2C%20%5Cvec%20z_%7Bij%7D%2C%20%5Cvec%20u_i%29%20%3D%20%0A%20%20%5Cvec%20x_%7Bij%7D%27%28t%29%5E%5Ctop%5Cvec%5Cbeta%0A%20%20%5Cfrac%7B%5Cphi%28-%5Cvec%20x_%7Bij%7D%28t%29%5E%5Ctop%5Cvec%5Cbeta%20-%20%5Cvec%20z_%7Bij%7D%5E%5Ctop%5Cvec%20u_i%29%7D%0A%20%20%20%20%20%20%20%7B%5CPhi%28-%5Cvec%20x_%7Bij%7D%28t%29%5E%5Ctop%5Cvec%5Cbeta%20-%20%5Cvec%20z_%7Bij%7D%5E%5Ctop%5Cvec%20u_i%29%7D%0A "
h(t\mid  \vec x_{ij}, \vec z_{ij}, \vec u_i) = 
  \vec x_{ij}'(t)^\top\vec\beta
  \frac{\phi(-\vec x_{ij}(t)^\top\vec\beta - \vec z_{ij}^\top\vec u_i)}
       {\Phi(-\vec x_{ij}(t)^\top\vec\beta - \vec z_{ij}^\top\vec u_i)}
")

``` r
# computes the time-varying fixed effects
x <- \(v) { v <- log(v); cbind(1, v, v^3) }
xp <- \(v) { v_org <- v; v <- log(v); cbind(0, 1, 3 * v^2) / v_org }

# generates the time-invariant covariate
gen_cov <- \(n) cbind(rnorm(n), runif(n) > .5)

# the fixed effects coefficients (beta)
beta <- c(-1, .25, .4, .5, 1)

admin_cens <- 5 # the administrative censoring time

# plot of the hazard
par(mar = c(5, 5, 1, 1))
seq(1e-2, admin_cens, length.out = 1000) |>
  (\(vs)
   plot(vs, 
        {
          beta_use <- head(beta, -2)
          eta <- x(vs) %*% beta_use
          eta_p <- xp(vs) %*% beta_use
          eta_p * exp(dnorm(-eta, log = TRUE) - pnorm(-eta, log = TRUE))
        }, 
        xlim = c(1e-2, admin_cens), xlab = "Time", ylab = "Hazard", bty = "l", 
        xaxs = "i", yaxs = "i", type = "l")
  )()

grid()
```

![](fig-mgsm/assing_sim_dat-1.png)

``` r
# generates the random effect covariates
gen_rng_cov <- \(n) cbind(1, rnorm(n))

# the random effect covariance matrix
Sigma <- structure(c(0.96725, -0.1505, -0.1505, 0.27875), .Dim = c(2L, 2L))

# simulates a given number of clusters
sim_dat <- \(n_clusters)
  lapply(seq_len(n_clusters), \(id) {
    n_members <- sample.int(9L, 1L) + 2L
    U <- drop(mvtnorm::rmvnorm(1, sigma = Sigma))
    X <- gen_cov(n_members)
    Z <- gen_rng_cov(n_members)
    
    # find the event time
    offset <- X %*% tail(beta, NCOL(X)) + Z %*% U
    
    beta_use <- head(beta, -NCOL(X))
    y <- sapply(offset, \(o){
      rng <- runif(1)
      res <- uniroot(\(ti) rng - pnorm(-o - x(ti) %*% beta_use), 
                     c(1e-32, 10000), 
                     tol = 1e-10)
      res$root
    })
    
    cens <- pmin(admin_cens, runif(n_members, 0, 2 * admin_cens))
    
    colnames(X) <- paste0("X", 1:NCOL(X))
    colnames(Z) <- paste0("Z", 1:NCOL(Z))
    
    out <- list(event = y < cens, y = pmin(y, cens), X = X, Z = Z, 
                id = rep(id, n_members))
    
    c(out, list(df = do.call(cbind, out)))
  })
```

``` r
# simulate the data
set.seed(8401834)
dat <- sim_dat(2000L)
dat_full <- do.call(rbind, lapply(dat, `[[`, "df")) |> data.frame()

mean(dat_full$event) # fraction of observed events
#> [1] 0.6672
NROW(dat_full) # number of observations
#> [1] 13923
# quantiles of the observed event times
subset(dat_full, event > 0)$y |> 
  quantile(probs = seq(0, 1, length.out = 11))
#>      0%     10%     20%     30%     40%     50%     60%     70%     80%     90% 
#> 0.07917 0.20883 0.26261 0.32910 0.42824 0.69067 1.74864 2.55597 3.24702 4.01113 
#>    100% 
#> 4.99897

# fit the model 
library(mixprobit)
#> Loading required package: survival
system.time(
  res <- fit_mgsm(
    formula = Surv(y, event) ~ X1 + X2, data = dat_full, id = id,
    rng_formula = ~ Z2, maxpts = c(1000L, 10000L), df = 8L))
#>    user  system elapsed 
#> 215.898   0.023 215.932
```

``` r
# the estimates are shown below
rbind(Estimate = res$beta_fixef, 
      Truth = tail(beta, 2))
#>            [,1]   [,2]
#> Estimate 0.4882 0.9919
#> Truth    0.5000 1.0000

res$Sigma # estimated covariance matrix
#>         [,1]    [,2]
#> [1,]  0.9799 -0.1634
#> [2,] -0.1634  0.2835
Sigma # the true covariance matrix
#>         [,1]    [,2]
#> [1,]  0.9673 -0.1505
#> [2,] -0.1505  0.2787

# plot of the estimated hazard
library(splines)
Xt_spline <- \(v) splineDesign(res$A_knots, log(v), outer.ok = TRUE)
Xt_spline_prime <- \(v) splineDesign(
  res$A_knots, log(v), derivs = 1, outer.ok = TRUE) / v

vs <- seq(1e-2, admin_cens, length.out = 1000)
haz_truth <- sapply(vs, \(v){
  beta_use <- head(beta, -2)
    eta <- x(v) %*% beta_use
    eta_p <- xp(v) %*% beta_use
    eta_p * exp(dnorm(-eta, log = TRUE) - pnorm(-eta, log = TRUE))
})
haz_est <- sapply(vs, \(v){
  eta <- Xt_spline(v) %*% res$beta_spline
  eta_p <- Xt_spline_prime(v) %*% res$beta_spline
  eta_p * exp(dnorm(-eta, log = TRUE) - pnorm(-eta, log = TRUE))
})

par(mar = c(5, 5, 1, 1))
matplot(
  vs, cbind(haz_truth, haz_est), type = "l", bty = "l", lty = 1:2, 
  col = "Black", xlab = "Time", ylab = "Hazard", xaxs = "i", yaxs = "i")
grid()
```

![](fig-mgsm/show_fit_ex-1.png)

``` r
# the maximum likelihood
res$logLik
#> [1] -12537

# can be compared with say a Weibull model
survreg(Surv(y, event) ~ X1 + X2, data = dat_full) |> logLik()
#> 'log Lik.' -18255 (df=4)
```
