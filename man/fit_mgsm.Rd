% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/gsm.R
\name{fit_mgsm}
\alias{fit_mgsm}
\title{Fits a mixed generalized survival model with the probit link}
\usage{
fit_mgsm(
  formula,
  data,
  id,
  rng_formula,
  df = 4L,
  maxpts = c(1000L, 10000L),
  key = 3L,
  abseps = 0,
  releps = 0.001,
  seed = 1L,
  method = c("adaptive_spherical_radial", "cdf_approach", "spherical_radial"),
  trace = 0L,
  max_it = 10000L
)
}
\arguments{
\item{formula}{formula for the fixed effects. Must contain the intercept.}

\item{data}{a data.frame with the needed variables.}

\item{id}{the cluster identifier for each individual.}

\item{rng_formula}{formula for the design matrix for the random effects.}

\item{df}{the degrees of freedom to use for the spline in the generalized
survival model.}

\item{maxpts}{a vector of integer with successive number of samples to use
when fitting the model.}

\item{key, abseps, releps}{parameters for the Monte Carlo method.}

\item{seed}{the fixed seed to use.}

\item{method}{character with the method to use to approximate the
intractble part of the marginal likelihood. \code{"spherical_radial"} and
\code{"adaptive_spherical_radial"} yields (adaptive) stochastic
spherical-radial rules and \code{"spherical_radial"} yields the CDF approach.}

\item{trace, max_it}{arguments passed to \code{\link{psqn_bfgs}}.}
}
\description{
Fits a mixed generalized survival model with the probit link
}
