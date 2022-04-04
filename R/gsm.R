.check_valid_gsm_method <- function(method)
  stopifnot(method %in% c("adaptive_spherical_radial",  "cdf_approach",
                              "spherical_radial"))

bs_spline_w_constraints <- function(event, obs_time, df, ord = 4L){
  df <- max(df, 2L)
  log_obs_time <- log(obs_time)
  bd_knots <- range(log_obs_time)
  knots <- quantile(log_obs_time[event], seq(0, 1, length.out = df - 1L))
  ik_knots <- knots[-c(1L, length(knots))]

  # create the constraint
  ui <- matrix(0, df, df + 1L)
  for(i in seq_len(df))
    ui[i, i + 0:1] <- c(-1, 1)
  ui <- ui[, -1, drop = FALSE]

  # assign the functions to compute the basis
  A_knots <- sort(c(rep(bd_knots, ord), ik_knots))
  offset <- numeric(df)
  basis <- function(x){
    bas <- splineDesign(A_knots, log(x), ord)
    bas[, -1, drop = FALSE] - rep(offset, each = length(x))
  }
  d_basis <- function(x){
    bas <- splineDesign(A_knots, log(x), ord, derivs = 1L) / x
    bas[, -1, drop = FALSE]
  }

  offset <- colMeans(basis(obs_time))

  # TODO: clean up

  structure(
    list(basis = basis, d_basis = d_basis, constraints = ui),
    bd_knots = bd_knots, ik_knots = ik_knots, offset = offset)
}

#' @importFrom stats lm lm.fit sd
gsm_beta_start <- function(obs_time, event, X_fixef, X_vary){
  # we match the mean and standard deviation of the log of time of the observed
  # outcomes
  obs_time <- obs_time[event]
  X_fixef <- X_fixef[event, , drop = FALSE]
  X_vary <- X_vary[event, , drop = FALSE]

  obs_time <- log(obs_time)
  time_sd <- sd(obs_time)
  intercept <- mean(obs_time) / time_sd
  slope <- 1 / time_sd

  # match the intercept with the fixed effects
  comb_inter <- lm.fit(X_fixef, rep(1, NROW(X_fixef)))$coef

  # fit the spline to match the linear effect
  comb_slope <- lm.fit(cbind(1, X_vary), obs_time)$coef

  beta_start_fixef <- (-intercept + comb_slope[1] * slope) * comb_inter
  beta_start_vary <- slope * comb_slope[-1]

  c(beta_start_fixef, beta_start_vary)
}

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
#' @param method character with the method to use to approximate the
#' intractble part of the marginal likelihood. \code{"spherical_radial"} and
#' \code{"adaptive_spherical_radial"} yields (adaptive) stochastic
#' spherical-radial rules and \code{"spherical_radial"} yields the CDF approach.
#' @param trace,max_it arguments passed to \code{\link{psqn_bfgs}}.
#'
#' @importFrom stats model.frame model.response terms model.matrix quantile optim
#' @importFrom splines splineDesign
#' @importFrom utils head tail
#' @importFrom psqn psqn_bfgs
#' @export
fit_mgsm <- function(
  formula, data, id, rng_formula, df = 4L, maxpts = c(1000L, 10000L),
  key = 3L, abseps = 0, releps = 1e-3, seed = 1L,
  method = c("adaptive_spherical_radial",  "cdf_approach",
                 "spherical_radial"),
  trace = 0L, max_it = 10000L){
  method <- method[1]
  .check_valid_gsm_method(method)

  mf_X <- model.frame(formula, data)
  y <- model.response(mf_X)
  X_fixef <- model.matrix(terms(mf_X), mf_X)
  id <- eval(substitute(id), data, parent.frame())

  mf_Z <- model.frame(rng_formula, data)
  Z <- model.matrix(terms(mf_Z), mf_Z)

  event <- y[, 2] > 0
  obs_time <- y[, 1]

  spline <-
    bs_spline_w_constraints(event = event, obs_time = obs_time, df = df)

  # construct the basis for the time-varying effect
  X_vary <- spline$basis(obs_time)
  X_vary_prime <- spline$d_basis(obs_time)

  # find the starting values
  n_fixef <- NCOL(X_fixef)
  n_vary <- NCOL(X_vary)
  X <- cbind(X_fixef, X_vary)
  X_prime <- cbind(matrix(0, NROW(X_fixef), NCOL(X_fixef)), X_vary_prime)

  beta_start <-
    gsm_beta_start(
      obs_time = obs_time, event = event, X_fixef = X_fixef, X_vary = X_vary)

  # handle the variable transformation
  n_rng <- NCOL(Z)
  sig <- diag(log(1), n_rng)
  par <- c(beta_start, sig[lower.tri(sig, TRUE)])

  n_par <- length(par)
  idx_varying <- n_fixef + seq_len(n_vary)

  trans_vars <- diag(n_par)
  trans_vars[idx_varying, idx_varying] <- spline$constraints
  par <- trans_vars %*% par
  par[idx_varying] <- log(par[idx_varying])

  get_beta <- function(theta){
    theta[idx_varying] <- exp(theta[idx_varying])
    drop(solve(trans_vars, theta))
  }

  d_theta <- function(d_beta, theta){
    out <- solve(t(trans_vars), d_beta)
    out[idx_varying] <-
      exp(theta[idx_varying]) * out[idx_varying]
    out
  }

  # create the data to pass to C++
  X <- t(X)
  Z <- t(Z)
  X_prime <- t(X_prime)
  cpp_data <- tapply(1:NROW(data), id, function(indices)
    list(X = X[, indices], X_prime = X_prime[, indices],
         Z = Z[, indices], y = obs_time[indices],
         event = as.numeric(event[indices])),
    simplify = FALSE)

  ptr <- get_gsm_ptr(cpp_data)

  fn <- function(theta, maxpts, key, abseps, releps, method, seed,
                 silent = TRUE){
    set.seed(seed)
    par <- get_beta(theta)
    beta <- head(par, length(beta_start))
    sig <- tail(par, -length(beta_start))
    res <- try(gsm_eval(
      ptr = ptr, beta = beta, sig = sig, maxpts = maxpts, key = key,
      abseps = abseps, releps = releps, method_use = method),
      silent = silent)
    if(inherits(res, "try-error")) NA_real_ else -res
  }

  gr <- function(theta, maxpts, key, abseps, releps, method, seed,
                 silent = TRUE){
    set.seed(seed)
    par <- get_beta(theta)
    beta <- head(par, length(beta_start))
    sig <- tail(par, -length(beta_start))
    res <- try(gsm_gr(
      ptr = ptr, beta = beta, sig = sig, maxpts = maxpts, key = key,
      abseps = abseps, releps = releps, method_use = method),
      silent = silent)
    if(inherits(res, "try-error"))
      return(rep(NA_real_, length(par)))

    d_theta(-res, theta)
  }

  to_set <- c("maxpts", "key", "abseps", "releps", "method", "seed")
  formals(fn)[to_set] <- formals(gr)[to_set] <-
    list(max(maxpts), key, abseps, releps, method, seed)

  # fit the model
  maxpts <- sort(maxpts)
  fits <- vector("list", length(maxpts) + 1L)
  fn_scale <- length(unique(id))
  opt_func <- function(par, maxpts){
    out <- psqn_bfgs(
      par, fn = function(x) fn(x, maxpts = maxpts) / fn_scale,
      gr = function(x)
        structure(gr(x, maxpts = maxpts) / fn_scale,
                  value = fn(x, maxpts = maxpts) / fn_scale),
      max_it = max_it, rel_eps = 1e-8, gr_tol = 1e-2,
      trace = trace)
    out$value <- fn(out$par, maxpts = maxpts)
    out
  }

  fits[[1L]] <- opt_func(par, maxpts[1])
  for(i in seq_len(length(maxpts) - 1L))
    fits[[i + 1L]] <- opt_func(fits[[i]]$par, maxpts[i + 1L])

  # end with optim
  opt_func_optim <- function(par, maxpts){
    optim(
      par, fn = fn, gr = gr,
      method = "BFGS", maxpts = maxpts,
      control = list(maxit = 1000L, fnscale = fn_scale, trace = trace > 0))
  }

  fits[[length(fits)]] <- opt_func_optim(
    fits[[length(fits) - 1L]]$par, max(maxpts))

  # format the result and return
  fit <- fits[[length(fits)]]
  par_org <- get_beta(fit$par)
  beta_est <- head(par_org, length(beta_start))
  beta_fixef <- head(beta_est, n_fixef)
  beta_spline <- tail(beta_est, -n_fixef)

  Sigma <- matrix(0, n_rng, n_rng)
  Sigma[lower.tri(Sigma, TRUE)] <- tail(fit$par, -length(beta_start))
  diag(Sigma) <- exp(diag(Sigma))
  Sigma <- tcrossprod(Sigma)

  # TODO: clean up

  list(beta_fixef = beta_fixef, beta_spline = beta_spline, Sigma = Sigma,
       fn = fn, gr = gr, logLik = -fit$value, get_beta = get_beta,
       optim = fit, fits = fits, spline = spline)
}

#' @inheritParams fit_mgsm
#' @importFrom utils head tail
#' @export
fit_mgsm_pedigree <- function(
  data, df = 4L, maxpts = c(1000L, 10000L), key = 3L, abseps = 0,
  releps = 1e-3, seed = 1L,
  method = c("adaptive_spherical_radial",  "cdf_approach",
                 "spherical_radial"),
  trace = 0L, max_it = 10000L){
  method <- method[1]
  .check_valid_gsm_method(method)

  stopifnot(is.list(data))

  event <- unlist(lapply(data, `[[`, "event")) > 0
  obs_time <- unlist(lapply(data, `[[`, "y"))
  stopifnot(length(event) == length(obs_time))

  spline <-
    bs_spline_w_constraints(event = event, obs_time = obs_time, df = df)

  # construct the basis for the time-varying effect
  dat_pass <- lapply(data, function(x){
    X_vary <- spline$basis(x$y)
    X_vary_prime <- spline$d_basis(x$y)

    within(x, {
      X_fixef <- X
      X_vary <- X_vary
      X <- cbind(X, X_vary)
      X_prime <- cbind(matrix(0., NROW(X_fixef), NCOL(X_fixef)), X_vary_prime)
      colnames(X_prime) <- colnames(X)
    })
  })

  # find the starting values
  X_fixef <- do.call(rbind, lapply(dat_pass, `[[`, "X_fixef"))
  X_vary <- do.call(rbind, lapply(dat_pass, `[[`, "X_vary"))

  beta_start <-
    gsm_beta_start(
      obs_time = obs_time, event = event, X_fixef = X_fixef,
      X_vary = X_vary)

  # handle the variable transformation
  n_fixef <- NCOL(X_fixef)
  n_vary <- NCOL(X_vary)
  n_scales <- length(dat_pass[[1L]]$scale_mats)
  par <- c(beta_start, numeric(n_scales))

  n_par <- length(par)
  idx_varying <- n_fixef + seq_len(n_vary)
  idx_scale <- length(beta_start) + seq_len(n_scales)

  trans_vars <- diag(n_par)
  trans_vars[idx_varying, idx_varying] <- spline$constraints
  par <- drop(trans_vars %*% par)
  par[idx_varying] <- log(par[idx_varying])

  get_beta <- function(theta){
    theta[idx_varying] <- exp(theta[idx_varying])
    theta <- drop(solve(trans_vars, theta))
    theta[idx_scale] <- exp(theta[idx_scale])
    theta
  }

  d_theta <- function(d_beta, theta){
    out <- solve(t(trans_vars), d_beta)
    out[idx_varying] <-
      exp(theta[idx_varying]) * out[idx_varying]
    out[idx_scale] <- exp(theta[idx_scale]) * out[idx_scale]
    out
  }

  # create the data to pass to C++
  dat_pass <- lapply(dat_pass, function(x){
    within(x, {
      X <- t(X)
      X_prime <- t(X_prime)
    })
  })

  ptr <- get_gsm_ptr_pedigree(dat_pass)

  fn <- function(theta, maxpts, key, abseps, releps, method, seed,
                 silent = TRUE){
    set.seed(seed)
    par <- get_beta(theta)
    beta <- head(par, length(beta_start))
    sigs <- tail(par, -length(beta_start))
    res <- try(gsm_eval_pedigree(
      ptr = ptr, beta = beta, sigs = sigs, maxpts = maxpts, key = key,
      abseps = abseps, releps = releps, method_use = method),
      silent = silent)
    if(inherits(res, "try-error")) NA_real_ else -res
  }

  gr <- function(theta, maxpts, key, abseps, releps, method, seed,
                 silent = TRUE){
    set.seed(seed)
    par <- get_beta(theta)
    beta <- head(par, length(beta_start))
    sigs <- tail(par, -length(beta_start))
    res <- try(gsm_gr_pedigree(
      ptr = ptr, beta = beta, sigs = sigs, maxpts = maxpts, key = key,
      abseps = abseps, releps = releps, method_use = method),
      silent = silent)
    if(inherits(res, "try-error"))
      return(rep(NA_real_, length(par)))

    d_theta(-res, theta)
  }

  to_set <- c("maxpts", "key", "abseps", "releps", "method", "seed")
  formals(fn)[to_set] <- formals(gr)[to_set] <-
    list(max(maxpts), key, abseps, releps, method, seed)

  # fit the model
  maxpts <- sort(maxpts)
  fits <- vector("list", length(maxpts) + 1L)
  fn_scale <- length(data)
  opt_func <- function(par, maxpts){
    out <- psqn_bfgs(
      par, fn = function(x) fn(x, maxpts = maxpts) / fn_scale,
      gr = function(x)
        structure(gr(x, maxpts = maxpts) / fn_scale,
                  value = fn(x, maxpts = maxpts) / fn_scale),
      max_it = max_it, rel_eps = 1e-8, gr_tol = 1e-2,
      trace = trace)
    out$value <- fn(out$par, maxpts = maxpts)
    out
  }

  fits[[1L]] <- opt_func(par, maxpts[1])
  for(i in seq_len(length(maxpts) - 1L))
    fits[[i + 1L]] <- opt_func(fits[[i]]$par, maxpts[i + 1L])

  # end with optim
  opt_func_optim <- function(par, maxpts){
    optim(
      par, fn = fn, gr = gr,
      method = "BFGS", maxpts = maxpts,
      control = list(maxit = 1000L, fnscale = fn_scale, trace = trace > 0))
  }

  fits[[length(fits)]] <- opt_func_optim(
    fits[[length(fits) - 1L]]$par, max(maxpts))

  # format the result and return
  fit <- fits[[length(fits)]]
  par_org <- get_beta(fit$par)
  beta_est <- head(par_org, length(beta_start))
  beta_fixef <- head(beta_est, n_fixef)
  beta_spline <- tail(beta_est, -n_fixef)

  sigs <- drop(exp(tail(fit$par, n_scales)))

  list(beta_fixef = beta_fixef, beta_spline = beta_spline, sigs = sigs,
       fn = fn, gr = gr, logLik = -fit$value, get_beta = get_beta,
       optim = fit, fits = fits, spline = spline)
}
