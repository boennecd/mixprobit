#' Fits a mixed generalized survival model with the probit link
#'
#' @param formula formula for the fixed effects. Must contain the intercept.
#' @param data a data.frame with the needed variables.
#' @param id the cluster identifier for each individual.
#' @param rng_formula formula for the design matrix for the random effects.
#' @param df the degrees of freedom to use for the spline in the generalized
#' survival model.
#' @param maxpts a vector of integer with successive number of samples to use
#' when fitting the model.
#' @param key,abseps,releps parameters for the Monte Carlo method.
#' @param seed the fixed seed to use.
#'
#' @importFrom stats  model.frame model.response terms model.matrix quantile optim
#' @importFrom splines splineDesign
#' @export
fit_mgsm <- function(
  formula, data, id, rng_formula, df = 4L, maxpts = c(1000L, 10000L), key = 3L, abseps = 0, releps = 1e-3, seed = 1L){
  mf_X <- model.frame(formula, data)
  y <- model.response(mf_X)
  X_fixef <- model.matrix(terms(mf_X), mf_X)
  stopifnot("(Intercept)" %in% colnames(X_fixef))
  X_fixef <- X_fixef[, !colnames(X_fixef) == "(Intercept)", drop = FALSE]
  id <- eval(substitute(id), data, parent.frame())

  mf_Z <- model.frame(rng_formula, data)
  Z <- model.matrix(terms(mf_Z), mf_Z)

  df <- max(df, 2L)
  event <- y[, 2] > 0
  obs_time <- y[, 1]
  log_obs_time <- log(obs_time)
  knots <- quantile(log_obs_time[event], seq(0, 1, length.out = df - 1L))
  bd_knots <- range(log_obs_time)
  ik_knots <- knots[-c(1L, length(knots))]

  ord <- 4L
  A_knots <- sort(c(rep(bd_knots, ord), ik_knots))

  # construct the basis for the time-varying effect
  X_vary <- splineDesign(A_knots, log_obs_time, ord)
  X_vary_prime <-
    splineDesign(A_knots, log_obs_time, ord, derivs = 1L) / obs_time

  X <- cbind(X_fixef, X_vary)
  X_prime <- cbind(matrix(0, NROW(X_fixef), NCOL(X_fixef)), X_vary_prime)

  # fit a log normal model on the complete data to get the starting values
  lm_res <- lm(log_obs_time[event] ~ X_fixef[event, ])
  lm_sum <- summary(lm_res)
  beta_start_fixef <- -unname(coef(lm_res)[-1]) / lm_sum$sigma

  # fit the spline to match the linear effect
  lm_lin_effect <- lm((log_obs_time / lm_sum$sigma) ~ X_vary - 1)
  beta_start_vary <- unname(coef(lm_lin_effect))
  stopifnot(all(diff(beta_start_vary) >= 0))
  beta_start <- c(beta_start_fixef, beta_start_vary)

  # fit the model for the observed data without the random effects
  n_par <- length(beta_start)
  n_constraints <- NCOL(X_vary) - 1L
  idx_constrainted <- NCOL(X_fixef) + 1L + seq_len(n_constraints)
  ui <- matrix(0, n_par, n_par)
  for(i in 1:n_constraints)
    ui[i + NCOL(X_fixef) + 1L , NCOL(X_fixef) + 0:1 + i] <- c(-1, 1)
  for(i in seq_len(NCOL(X_fixef) + 1L ))
    ui[i, i] <- 1

  theta_start <- ui %*% beta_start
  theta_start[idx_constrainted] <- log(theta_start[idx_constrainted])

  get_beta <- function(theta){
    theta[idx_constrainted] <- exp(theta[idx_constrainted])
    solve(ui, theta)
  }

  d_theta <- function(d_beta, theta){
    out <- solve(t(ui), d_beta)
    out[idx_constrainted] <-
      exp(theta[idx_constrainted]) * out[idx_constrainted]
    out
  }

  # extend the matrix to transform the parameters
  n_rng <- NCOL(Z)
  n_par <- length(beta_start) + (n_rng * (n_rng + 1L)) / 2L
  ui_new <- diag(n_par)
  ui_new[seq_along(beta_start), seq_along(beta_start)] <- ui
  ui <- ui_new

  # create the data to pass to C++
  X <- t(X)
  Z <- t(Z)
  X_prime <- t(X_prime)
  cpp_data <- tapply(1:NROW(dat_full), id, function(indices)
    list(X = X[, indices], X_prime = X_prime[, indices],
         Z = Z[, indices], y = obs_time[indices],
         event = as.numeric(event[indices])),
    simplify = FALSE)

  ptr <- get_gsm_ptr(cpp_data)

  sig <- numeric((n_rng * (n_rng + 1L)) %/% 2L)
  par <- c(theta_start, sig[lower.tri(sig, TRUE)])

  fn <- function(theta, maxpts){
    set.seed(seed)
    par <- get_beta(theta)
    beta <- head(par, length(beta_start))
    sig <- tail(par, -length(beta_start))
    res <- try(gsm_eval(
      ptr = ptr, beta = beta, sig = sig, maxpts = maxpts, key = key, abseps = abseps,
      releps = releps), silent = TRUE)
    if(inherits(res, "try-error")) NA else -res
  }
  gr <- function(theta, maxpts){
    set.seed(seed)
    par <- get_beta(theta)
    beta <- head(par, length(beta_start))
    sig <- tail(par, -length(beta_start))
    res <- try(gsm_gr(
      ptr = ptr, beta = beta, sig = sig, maxpts = maxpts, key = key,
      abseps = abseps, releps = releps), silent = TRUE)
    if(inherits(res, "try-error"))
      return(rep(NA, length(par)))

    d_theta(-res, theta)
  }

  # fit the model
  maxpts <- sort(maxpts)
  fits <- vector("list", length(maxpts))
  opt_func <- function(par, maxpts)
    optim(par, fn, gr, method = "BFGS", control = list(maxit = 10000L),
          maxpts = maxpts)

  fits[[1L]] <- opt_func(par, maxpts[1])
  for(i in seq_len(length(maxpts) - 1L))
    fits[[i + 1L]] <- opt_func(fits[[i]]$par, maxpts[i + 1L])

  # format the result and return
  fit <- fits[[length(fits)]]
  par_org <- get_beta(fit$par)
  beta_est <- head(par_org, length(beta_start))
  beta_fixef <- head(beta_est, length(beta_start_fixef))
  beta_spline <- tail(beta_est, -length(beta_start_fixef))

  Sigma <- matrix(0, n_rng, n_rng)
  Sigma[lower.tri(Sigma, TRUE)] <- tail(par_org, -length(theta_start))
  diag(Sigma) <- exp(diag(Sigma))
  Sigma <- tcrossprod(Sigma)

  list(beta_fixef = beta_fixef, beta_spline = beta_spline, Sigma = Sigma,
       get_beta = get_beta, fn = fn, gr = gr, logLik = -fit$value,
       optim = fit, fits = fits, knots = knots, A_knots = A_knots)
}
