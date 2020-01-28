#include "mvtnorm-wrapper.h"
#include "integrand-binary.h"

// [[Rcpp::export]]
Rcpp::List pmvnorm_cpp(arma::vec const &lower, arma::vec const &upper,
                       arma::vec const &mean, arma::mat const &cov,
                       int const maxpts, double const abseps,
                       double const releps){
  using Rcpp::Named;
  auto res = pmvnorm::cdf(lower, upper, mean, cov, maxpts, abseps, releps);

  return Rcpp::List::create(Named("value")  = res.value,
                            Named("error")  = res.error,
                            Named("inform") = res.inform);
}

// [[Rcpp::export]]
Rcpp::NumericVector aprx_binary_mix(
    arma::ivec const &y, arma::vec const &eta, arma::mat const &Z,
    arma::mat const &Sigma, int const mxvals, int const key,
    double const epsabs, double const epsrel){
  using namespace ranrth_aprx;
  using Rcpp::NumericVector;

  set_integrand(std::unique_ptr<integrand>(
      new mix_binary(y, eta, Z, Sigma)));
  auto res = integral_arpx(mxvals, key, epsabs, epsrel);

  NumericVector out = NumericVector::create(res.value);
  out.attr("error") = res.err;
  out.attr("inform") = res.inform;
  out.attr("inivls") = res.inivls;

  return out;
}
