#include "gaus-Hermite.h"
#include "mvtnorm-wrapper.h"
#include "integrand-binary.h"
#include "threat-safe-random.h"
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::export]]
Rcpp::List pmvnorm_cpp(arma::vec const &lower, arma::vec const &upper,
                       arma::vec const &mean, arma::mat const &cov,
                       int const maxpts, double const abseps,
                       double const releps){
  parallelrng::set_rng_seeds(1L);

  using Rcpp::Named;
  auto const res = pmvnorm::cdf(
    lower, upper, mean, cov, maxpts, abseps, releps);

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

  parallelrng::set_rng_seeds(1L);
  set_integrand(std::unique_ptr<integrand>(
      new mix_binary(y, eta, Z, Sigma)));
  auto const res = integral_arpx(
    mxvals, key, epsabs, epsrel);

  NumericVector out = NumericVector::create(res.value);
  out.attr("error") = res.err;
  out.attr("inform") = res.inform;
  out.attr("inivls") = res.inivls;

  return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector aprx_binary_mix_cdf(
    arma::ivec const &y, arma::vec eta, arma::mat Z,
    arma::mat const &Sigma, int const maxpts, double const abseps,
    double const releps){
  using Rcpp::NumericVector;

  parallelrng::set_rng_seeds(1L);
  arma::uword const n = y.n_elem,
                    p = Z.n_rows;
  assert(eta.n_elem == n);
  assert(Z.n_cols == n);
  assert(Sigma.n_cols == p);
  assert(Sigma.n_rows == p);

  {
    arma::rowvec dum_vec(n, arma::fill::ones);
    dum_vec.elem(arma::find(y < 1L)).fill(-1);
    Z.each_row() %= dum_vec;
    eta %= dum_vec.t();
  }

  arma::mat S = Z.t() * (Sigma * Z);
  S.diag() += 1.;
  arma::vec const mean(n, arma::fill::zeros);
  arma::vec lower(n);
  lower.fill(-std::numeric_limits<double>::infinity());

  auto const res =
    pmvnorm::cdf(lower, eta, mean, S, maxpts, abseps, releps);

  NumericVector out = NumericVector::create(res.value);
  out.attr("inform") = res.inform;
  out.attr("error")  = res.error;

  return out;
}

inline arma::mat get_mat_from_sexp(SEXP X){
  Rcpp::NumericMatrix Xm(X);
  arma::mat out(Xm.begin(), Xm.nrow(), Xm.ncol());
  return out;
}

/* very specialized function just to get a grasp of the computation times
 * one can achieve. */
// [[Rcpp::export]]
double aprx_binary_mix_cdf_salamander
  (Rcpp::List data, arma::vec const &beta, arma::vec const &log_sds,
   unsigned const n_threads, int const maxpts, double const abseps,
   double const releps){
  /* setup data */
  arma::vec const vars = arma::exp(2 * log_sds);

  struct cluster_data {
    arma::ivec const y, var_idx;
    arma::mat const X, Z;
    arma::uword const p = Z.n_rows, n = Z.n_cols;

    cluster_data(Rcpp::List data):
      y      (Rcpp::as<Rcpp::IntegerVector>(data["y"])),
      var_idx(Rcpp::as<Rcpp::IntegerVector>(data["var_idx"])),
      X      (get_mat_from_sexp(data["X"])),
      Z      (get_mat_from_sexp(data["Z"]))
    {
      assert(X.n_rows == n);
      assert(y.n_elem == n);
      assert(var_idx.n_elem == p);
      assert(var_idx.min() >= 0L);
    }
  };

  unsigned const n_clusters = data.size();
  std::vector<cluster_data> const comp_dat = ([&](){
    std::vector<cluster_data> out;
    out.reserve(n_clusters);
    for(unsigned i = 0; i < n_clusters; ++i)
      out.emplace_back(Rcpp::as<Rcpp::List>(data[i]));
    return out;
  })();

  /* approximate log-likelihood */
  {
    int const n_use = std::max(unsigned(1L), n_threads);
#ifdef _OPENMP
    omp_set_num_threads(n_use);
#endif
    parallelrng::set_rng_seeds(n_use);
  }

  double out(0.);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:out)
#endif
  for(unsigned i = 0; i < n_clusters; ++i){
    cluster_data const &my_data = comp_dat[i];
    unsigned const n = my_data.n,
                   p = my_data.p;
    assert(vars.n_elem > (arma::uword)my_data.var_idx.max());

    arma::vec eta = my_data.X * beta;
    arma::mat Z(my_data.Z);

    {
      arma::rowvec dum_vec(n, arma::fill::ones);
      dum_vec.elem(arma::find(my_data.y < 1L)).fill(-1);
      Z.each_row() %= dum_vec;
      eta %= dum_vec.t();
    }

    arma::vec Sigma_diag(p);
    {
      unsigned j(0L);
      for(auto idx : my_data.var_idx)
        Sigma_diag[j++] = vars[idx];
    }
    arma::mat Sigma = arma::diagmat(Sigma_diag);

    arma::mat S = Z.t() * (Sigma * Z);
    S.diag() += 1.;
    arma::vec const mean(n, arma::fill::zeros);
    arma::vec lower(n);
    lower.fill(-std::numeric_limits<double>::infinity());

    out += std::log(
      pmvnorm::cdf(lower, eta, mean, S, maxpts, abseps, releps).value);
  }

  return out;
}

/* use to set cached values to avoid computation cost from computing the
 * weights */
// [[Rcpp::export]]
Rcpp::List set_GH_rule_cached(unsigned const b){
  auto const &res = GaussHermite::gaussHermiteDataCached(b);
  return Rcpp::List::create(
    Rcpp::Named("x") = res.x, Rcpp::Named("w") = res.w);
}

// [[Rcpp::export]]
double aprx_binary_mix_ghq(
    arma::ivec const &y, arma::vec eta, arma::mat Z,
    arma::mat const &Sigma, unsigned const b){
  auto const &rule = GaussHermite::gaussHermiteDataCached(b);
  mix_binary integrand(y, eta, Z, Sigma);

  return GaussHermite::approx(rule, integrand);
}

/* brute force MC estimate */
// [[Rcpp::export]]
double aprx_binary_mix_brute(
    arma::ivec const &y, arma::vec eta, arma::mat Z,
    arma::mat const &Sigma, unsigned const n_sim,
    unsigned const n_threads = 1L){
  std::size_t const p = Sigma.n_cols;
  mix_binary integrand(y, eta, Z, Sigma);
  arma::vec par_vec(p);

  {
    int const n_use = std::max(unsigned(1L), n_threads);
#ifdef _OPENMP
    omp_set_num_threads(n_use);
#endif
    parallelrng::set_rng_seeds(n_use);
  }

  double out(0.);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:out) firstprivate(par_vec, integrand)
#endif
  for(unsigned i = 0; i < n_sim; ++i){
    for(unsigned j = 0; j < p; ++j)
      par_vec[j] = rngnorm_wrapper();

    out += integrand(par_vec.begin());
  }

  out /= (double)n_sim;
  return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector for_rngnorm_wrapper_test
  (unsigned const n, unsigned const n_threads){
  {
    int const n_use = std::max(unsigned(1L), n_threads);
#ifdef _OPENMP
    omp_set_num_threads(n_use);
#endif
    parallelrng::set_rng_seeds(n_use);
  }

  Rcpp::NumericVector out(n);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(unsigned i = 0; i < n; ++i)
    out[i] =  rngnorm_wrapper();

  return out;
}
