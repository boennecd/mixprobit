#include "mvtnorm-wrapper.h"


/* arma::vec lower, arma::vec upper, arma::vec mean,
 arma::mat const &cov, int const maxpts,
 double const abseps, double const releps */

// [[Rcpp::export]]
Rcpp::List pmvnorm_cpp(arma::vec const &lower, arma::vec const &upper,
                   arma::vec const &mean, arma::mat const &cov,
                   int const maxpts, double const abseps,
                   double const releps){
  using Rcpp::List;
  using Rcpp::Named;
  auto res = pmvnorm::cdf(lower, upper, mean, cov, maxpts, abseps, releps);

  return List::create(Named("value") = res.value,
                      Named("error") = res.error,
                      Named("inform") = res.inform);
}
