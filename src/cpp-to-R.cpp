#include "gaus-Hermite.h"
#include "mvtnorm-wrapper.h"
#include "integrand-binary.h"
#include "restrict-cdf.h"
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
Rcpp::List pmvnorm_cpp_restrict(
    arma::vec const &mean, arma::mat const &cov,
    int const maxpts, double const abseps, double const releps,
    bool const gradient = false){
  parallelrng::set_rng_seeds(1L);

  using Rcpp::Named;
  auto const res = ([&]{
    if(gradient)
      return restrictcdf::cdf<restrictcdf::deriv>
        (mean, cov).approximate(maxpts, abseps, releps);

    return restrictcdf::cdf<restrictcdf::likelihood>
      (mean, cov).approximate(maxpts, abseps, releps);
  })();

  return Rcpp::List::create(Named("value")  = res.finest,
                            Named("error")  = res.abserr,
                            Named("inform") = res.inform);
}

// [[Rcpp::export]]
Rcpp::NumericVector aprx_binary_mix(
    arma::ivec const &y, arma::vec const &eta, arma::mat const &Z,
    arma::mat const &Sigma, int const maxpts, double const abseps,
    double const releps, int const key = 2L){
  using namespace ranrth_aprx;
  using Rcpp::NumericVector;

  parallelrng::set_rng_seeds(1L);
  set_integrand(std::unique_ptr<integrand>(
      new mix_binary(y, eta, Z, Sigma)));
  auto const res = integral_arpx(
    maxpts, key, abseps, releps);

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

/* class to approximate the marginal log-likelihood where the unconditional
 * covariance is a diagonal matrix potentially with shared parameters. */
class aprx_binary_mix_cdf_structured_diag {
  /* class to holder data for each cluster */
  struct cluster_data {
    arma::vec const dum_vec;
    arma::ivec const var_idx;
    arma::mat const X, Z;
    arma::uword const p = Z.n_rows, n = Z.n_cols;
    arma::vec const mean = arma::vec(n, arma::fill::zeros),
                   lower = ([&](){
                     arma::vec out(n);
                     out.fill(-std::numeric_limits<double>::infinity());
                     return out;
                   })();

    cluster_data(Rcpp::List data):
      dum_vec(([&](){
        arma::ivec const y = Rcpp::as<Rcpp::IntegerVector>(data["y"]);
        arma::vec out(y.n_elem, arma::fill::ones);
        out.elem(arma::find(y < 1L)).fill(-1);
        return out;
      })()),
      var_idx(Rcpp::as<Rcpp::IntegerVector>(data["var_idx"])),
      X(([&](){
        arma::mat out(get_mat_from_sexp(data["X"]));
        out.each_col() %= dum_vec;
        return out;
      })()),
      Z(([&](){
        arma::mat out(get_mat_from_sexp(data["Z"]));
        out.each_row() %= dum_vec.t();
        return out;
      })())
    {
      assert(X.n_rows == n);
      assert(dum_vec.n_elem == n);
      assert(var_idx.n_elem == p);
      assert(var_idx.min() >= 0L);
    }
  };

  std::vector<cluster_data> const comp_dat;
  unsigned const n_threads,
                n_clusters = comp_dat.size();

public:
  aprx_binary_mix_cdf_structured_diag
  (Rcpp::List data, unsigned const n_threads):
  comp_dat(([&](){
    unsigned const n = data.size();
    std::vector<cluster_data> out;
    out.reserve(n);
    for(unsigned i = 0; i < n; ++i)
      out.emplace_back(Rcpp::as<Rcpp::List>(data[i]));
    return out;
  })()), n_threads(n_threads)
  { }

  /* approximates the log-likelihood */
  double operator()
  (arma::vec const &beta, arma::vec const &log_sds, int const maxpts,
   double const abseps, double const releps){
    /* set the threads and seeds */
    {
      int const n_use = std::max(unsigned(1L), n_threads);
#ifdef _OPENMP
      omp_set_num_threads(n_use);
#endif
      parallelrng::set_rng_seeds(n_use);
    }

    /* compute variance parameters and then approximate the
     * log-likelihood. */
    arma::vec const vars = arma::exp(2 * log_sds);
    double out(0.);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:out)
#endif
    for(unsigned i = 0; i < n_clusters; ++i){
      cluster_data const &my_data = comp_dat[i];
      unsigned const p = my_data.p;
      assert(vars.n_elem > (arma::uword)my_data.var_idx.max());

      arma::vec const eta = my_data.X * beta;
      arma::mat const &Z = my_data.Z;

      arma::vec Sigma_diag(p);
      {
        unsigned j(0L);
        for(auto idx : my_data.var_idx)
          Sigma_diag[j++] = vars[idx];
      }
      arma::mat Sigma = arma::diagmat(Sigma_diag);

      arma::mat S = Z.t() * (Sigma * Z);
      S.diag() += 1.;

      out += std::log(
        pmvnorm::cdf(my_data.lower, eta, my_data.mean, S,
                     maxpts, abseps, releps).value);
    }

    return out;
  }
};

/* very specialized function just to get a grasp of the computation times
 * one can achieve. First call this function to create a pointer to a
 * functor. Then use the next function to approximate the log-likelihood at
 * a given point. */
// [[Rcpp::export]]
SEXP aprx_binary_mix_cdf_get_ptr
  (Rcpp::List data, unsigned const n_threads){
  using dat_T = aprx_binary_mix_cdf_structured_diag;
  return Rcpp::XPtr<dat_T>(new dat_T(data, n_threads), true);
}

// [[Rcpp::export]]
double aprx_binary_mix_cdf_eval
  (SEXP ptr, arma::vec const &beta, arma::vec const &log_sds,
   int const maxpts,double const abseps, double const releps){
  Rcpp::XPtr<aprx_binary_mix_cdf_structured_diag> functor(ptr);

  return functor->operator()(beta, log_sds, maxpts, abseps, releps);
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
