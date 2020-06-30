#include "gaus-Hermite.h"
#include "mvtnorm-wrapper.h"
#include "integrand-binary.h"
#include "restrict-cdf.h"
#include "threat-safe-random.h"
#include "welfords.h"
#include "my_pmvnorm.h"
#include "sobol.h"

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
    double const releps, int const key = 2L,
    bool const is_adaptive = false){
  using namespace ranrth_aprx;
  using namespace integrand;
  using Rcpp::NumericVector;

  parallelrng::set_rng_seeds(1L);
  auto const res = ([&](){
    if(is_adaptive){
      mix_binary bin(y, eta, Z, Sigma);
      mvn<mix_binary > mix_bin(bin);

      set_integrand(std::unique_ptr<base_integrand>(
          new adaptive<mvn<mix_binary> > (mix_bin)));

      return integral_arpx(maxpts, key, abseps, releps);
    }

    set_integrand(std::unique_ptr<base_integrand>(
        new mix_binary(y, eta, Z, Sigma)));

    return integral_arpx(maxpts, key, abseps, releps);
  })();

  NumericVector out = NumericVector::create(res.value);
  out.attr("error") = res.err;
  out.attr("inform") = res.inform;
  out.attr("inivls") = res.inivls;

  return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector aprx_jac_binary_mix(
    arma::ivec const &y, arma::vec const &eta, arma::mat const &X,
    arma::mat const &Z, arma::mat const &Sigma, int const maxpts,
    double const abseps, double const releps, int const key = 2L,
    bool const is_adaptive = false){
  using namespace ranrth_aprx;
  using namespace integrand;
  using Rcpp::NumericVector;

  parallelrng::set_rng_seeds(1L);
  auto const res = ([&](){
    if(is_adaptive){
      mix_binary bin(y, eta, Z, Sigma, &X);
      mvn<mix_binary > mix_bin(bin);

      set_integrand(std::unique_ptr<base_integrand>(
          new adaptive<mvn<mix_binary> > (mix_bin)));

      return jac_arpx(maxpts, key, abseps, releps);
    }

    set_integrand(std::unique_ptr<base_integrand>(
        new mix_binary(y, eta, Z, Sigma, &X)));

    return jac_arpx(maxpts, key, abseps, releps);
  })();

  NumericVector out(Rcpp::wrap(res.value));
  out.attr("error") = Rcpp::wrap(res.err);
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

#ifdef _OPENMP
/* openMP reductions */
#pragma omp declare reduction(+: arma::vec: omp_out += omp_in) \
  initializer (omp_priv = omp_orig)
#endif

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
  bool const gradient;

public:
  aprx_binary_mix_cdf_structured_diag
  (Rcpp::List data, unsigned const n_threads, bool const gradient):
  comp_dat(([&](){
    unsigned const n = data.size();
    std::vector<cluster_data> out;
    out.reserve(n);
    for(unsigned i = 0; i < n; ++i)
      out.emplace_back(Rcpp::as<Rcpp::List>(data[i]));
    return out;
  })()), n_threads(n_threads), gradient(gradient)
  { }

  /* approximates the log-likelihood */
  arma::vec operator()
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
    arma::vec out(1L + gradient * (beta.n_elem + log_sds.n_elem),
                  arma::fill::zeros);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:out)
#endif
    for(unsigned i = 0; i < n_clusters; ++i){
      cluster_data const &my_data = comp_dat[i];
      unsigned const p = my_data.p,
                     n = my_data.n;
      assert(vars.n_elem > (arma::uword)my_data.var_idx.max());

      arma::vec const eta = my_data.X * beta;
      arma::mat const &Z = my_data.Z;

      arma::mat const S = ([&](){
        arma::vec Sigma_diag(p);
        {
          unsigned j(0L);
          for(auto idx : my_data.var_idx)
            Sigma_diag[j++] = vars[idx];
        }

        arma::mat const Sigma = arma::diagmat(Sigma_diag);
        arma::mat S_out = Z.t() * (Sigma * Z);
        S_out.diag() += 1.;

        return S_out;
      })();

      if(gradient){
        auto output = restrictcdf::cdf<restrictcdf::deriv>
          (-eta, S).approximate(maxpts, abseps, releps);

        double const phat = output.finest[0L];
        arma::vec d_eta(output.finest.memptr() + 1L, n, false),
                  d_S  (output.finest.memptr() + 1L + n,
                        (n * (n + 1L)) / 2L, false);

        d_eta *= -1 / phat;
        d_S /= phat;

        /* log-probability */
        out[0L] += std::log(phat);

        /* derivatives w.r.t. coefs */
        out.subvec(1L, beta.n_elem) += arma::trans(d_eta.t() * my_data.X);

        /* derivatives w.r.t. log standard deviations */
        arma::vec d_log_sds(out.memptr() + 1L + beta.n_elem,
                            log_sds.n_elem, false);
        unsigned pi(0L);
        /* TODO: maybe do something smarter... */
        arma::vec zi(n);
        for(auto const idx : my_data.var_idx){
          zi = Z.row(pi++).t();

          double nv(0.);
          double const *d_Si = d_S.memptr();
          for(unsigned c = 0; c < n; c++){
            double const zic = zi[c];
            for(unsigned r = 0; r < c; r++)
              nv += 2. * *d_Si++ * zi[r] * zic;
            nv   +=      *d_Si++ * zic   * zic;

          }

          d_log_sds[idx] += 2. * vars[idx] * nv;
        }

      } else
        out += arma::log(
          restrictcdf::cdf<restrictcdf::likelihood>
          (-eta, S).approximate(maxpts, abseps, releps).finest);
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
  (Rcpp::List data, unsigned const n_threads, bool const gradient = false){
  using dat_T = aprx_binary_mix_cdf_structured_diag;
  return Rcpp::XPtr<dat_T>(new dat_T(data, n_threads, gradient), true);
}

// [[Rcpp::export]]
arma::vec aprx_binary_mix_cdf_eval
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
    arma::mat const &Sigma, unsigned const b,
    bool const is_adaptive = false){
  auto const &rule = GaussHermite::gaussHermiteDataCached(b);
  integrand::mix_binary integrand(y, eta, Z, Sigma);

  return GaussHermite::approx(rule, integrand, is_adaptive);
}

/* brute force MC estimate */
// [[Rcpp::export]]
Rcpp::NumericVector aprx_binary_mix_brute(
    arma::ivec const &y, arma::vec eta, arma::mat Z,
    arma::mat const &Sigma, unsigned const n_sim,
    unsigned const n_threads = 1L, bool const is_is = true){
  using namespace integrand;

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

  welfords out;

  if(is_is){
    struct integrand_worker {
      mix_binary const bin;
      mvn<mix_binary> const mvn_obj = mvn<mix_binary>(bin);
      adaptive<mvn<mix_binary> > const ada =
        adaptive<mvn<mix_binary> >(mvn_obj);

      integrand_worker(mix_binary const &bin): bin(bin) { }
      integrand_worker(integrand_worker const &other):
        integrand_worker(other.bin) { }
    };
    integrand_worker const worker(integrand);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(welPlus:out) firstprivate(par_vec, worker)
#endif
    for(unsigned i = 0; i < n_sim; ++i){
      for(unsigned j = 0; j < p; ++j)
        par_vec[j] = rngnorm_wrapper();

      out += antithetic<base_integrand>(worker.ada)(par_vec);
    }

  } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(welPlus:out) firstprivate(par_vec, integrand)
#endif
    for(unsigned i = 0; i < n_sim; ++i){
      for(unsigned j = 0; j < p; ++j)
        par_vec[j] = rngnorm_wrapper();

      out += antithetic<base_integrand>(integrand)(par_vec);
    }
  }

  Rcpp::NumericVector R_out(1);
  R_out[0] = out.mean();
  R_out.attr("SE") = std::sqrt(out.var() / (double)n_sim);
  return R_out;
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

// [[Rcpp::export]]
Rcpp::NumericVector my_pmvnorm_cpp
  (arma::vec const &mean_in, arma::mat const &sigma_in,
   unsigned const nsim, double const eps) {
  auto const out = my_pmvnorm(mean_in, sigma_in, nsim, eps);

  Rcpp::NumericVector ret_val(1L);
  ret_val[0L] = out.est;
  ret_val.attr("error") = out.se;
  ret_val.attr("n_sim") = out.nsim;
  return ret_val;
}

// [[Rcpp::export(rng = false)]]
SEXP get_sobol_obj
  (int const dimen, int const scrambling = 0L, int const seed = 4711L){
  return Rcpp::XPtr<sobol_gen>(new sobol_gen(dimen, scrambling, seed));
}

// [[Rcpp::export(rng = false)]]
arma::mat eval_sobol(unsigned const n, SEXP ptr){
  Rcpp::XPtr<sobol_gen> functor(ptr);
  size_t const dimen = functor->dimen;

  if(n > 0L){
    arma::mat out(dimen, n);
    arma::vec wrk(dimen);
    for(size_t i = 0; i < n; ++i){
      functor->operator()(wrk);
      for(size_t j = 0; j < dimen; ++j)
        out.at(j, i) = wrk[j];
    }
    return out;
  }

  return arma::mat(functor->dimen, 0L);
}
