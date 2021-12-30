#include "gaus-Hermite.h"
#include "mvtnorm-wrapper.h"
#include "integrand-binary.h"
#include "restrict-cdf.h"
#include "threat-safe-random.h"
#include "welfords.h"
#include "my_pmvnorm.h"
#include "sobol.h"
#include "qmc.h"
#include "integrand-multinomial.h"
#include <vector>
#include "gsm.h"

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
                            Named("inform") = res.inform,
                            Named("intvls") = res.intvls);
}

// [[Rcpp::export]]
Rcpp::List pmvnorm_cpp_restrict(
    arma::vec const &mean, arma::mat const &cov,
    int const maxpts, double const abseps, double const releps,
    bool const gradient = false, int const minvls = 0L){
  parallelrng::set_rng_seeds(1L);

  using Rcpp::Named;
  restrictcdf::cdf<restrictcdf::deriv>::set_working_memory
    (mean.size(), 1L);
  auto const res = ([&]{
    if(gradient)
      return restrictcdf::cdf<restrictcdf::deriv>
        (mean, cov, true).approximate(maxpts, abseps, releps, minvls);

    return restrictcdf::cdf<restrictcdf::likelihood>
      (mean, cov, true).approximate(maxpts, abseps, releps, minvls);
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
          new adaptive<mvn<mix_binary> > (mix_bin, true)));

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
Rcpp::NumericVector aprx_mult_mix(
    unsigned const n_alt, arma::vec const &eta, arma::mat const &Z,
    arma::mat const &Sigma, int const maxpts, double const abseps,
    double const releps, int const key = 2L,
    bool const is_adaptive = false){
  using namespace ranrth_aprx;
  using namespace integrand;
  using Rcpp::NumericVector;

  parallelrng::set_rng_seeds(1L);
  auto const res = ([&](){
    if(is_adaptive){
      multinomial_group obj(n_alt, Z, eta, Sigma);
      mvn<multinomial_group > mix_obj(obj);

      set_integrand(std::unique_ptr<base_integrand>(
          new adaptive<mvn<multinomial_group> > (mix_obj, true)));

      return integral_arpx(maxpts, key, abseps, releps);
    }

    set_integrand(std::unique_ptr<base_integrand>(
        new multinomial_group(n_alt, Z, eta, Sigma)));

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
    mix_binary bin(y, eta, Z, Sigma, &X);
    if(is_adaptive){
      mvn<mix_binary > mix_bin(bin);
      set_integrand(std::unique_ptr<base_integrand>(
          new adaptive<mvn<mix_binary> > (mix_bin, true)));

    } else
      set_integrand(std::unique_ptr<base_integrand>(
          new mix_binary(y, eta, Z, Sigma, &X)));

    auto res = jac_arpx(maxpts, key, abseps, releps);
    return res;
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
  out.attr("intvls") = res.intvls;

  return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector aprx_mult_mix_cdf(
    unsigned const n_alt, arma::vec const &eta, arma::mat const &Z,
    arma::mat const &Sigma, int const maxpts, double const abseps,
    double const releps){
  using Rcpp::NumericVector;

  parallelrng::set_rng_seeds(1L);
  arma::uword const n = eta.n_elem / n_alt,
                    p = Z.n_rows;
  assert(n * n_alt == eta.n_elem);
  assert(Z.n_cols == eta.n_elem);
  assert(Sigma.n_cols == p);
  assert(Sigma.n_rows == p);

  arma::mat S = Z.t() * (Sigma * Z);
  {
    arma::mat dum(n_alt, n_alt, arma::fill::ones);
    dum.diag() += 1.;
    for(size_t i = 0; i < n; ++i){
      size_t const start = i * n_alt;
      arma::span const indices(start, start +  n_alt - 1L);
      S(indices, indices) += dum;
    }
  }
  arma::vec const mean(eta.n_elem, arma::fill::zeros);
  arma::vec lower(eta.n_elem);
  lower.fill(-std::numeric_limits<double>::infinity());

  auto const res =
    pmvnorm::cdf(lower, eta, mean, S, maxpts, abseps, releps);

  NumericVector out = NumericVector::create(res.value);
  out.attr("inform") = res.inform;
  out.attr("error")  = res.error;
  out.attr("intvls") = res.intvls;

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
  size_t const max_cdf_dim;
  bool const gradient, do_reorder;
  size_t const minvls;

public:
  aprx_binary_mix_cdf_structured_diag
  (Rcpp::List data, unsigned const n_threads, bool const gradient,
   size_t const minvls, bool const do_reorder):
  comp_dat(([&](){
    unsigned const n = data.size();
    std::vector<cluster_data> out;
    out.reserve(n);
    for(unsigned i = 0; i < n; ++i)
      out.emplace_back(Rcpp::as<Rcpp::List>(data[i]));
    return out;
  })()), n_threads(n_threads),
  max_cdf_dim(([&](){
    size_t out(0);
    for(auto &x : comp_dat)
      if(x.n > out)
        out = x.n;
    return out;
  })()), gradient(gradient), do_reorder(do_reorder), minvls(minvls)
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
      restrictcdf::cdf<restrictcdf::deriv>::set_working_memory
        (max_cdf_dim, n_use);
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
          (-eta, S, do_reorder).approximate(maxpts, abseps, releps,
           static_cast<int>(minvls));

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
          (-eta, S, do_reorder).approximate(maxpts, abseps, releps,
           static_cast<int>(minvls)).finest);
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
  (Rcpp::List data, unsigned const n_threads, bool const gradient = false,
   unsigned const minvls = 0L, bool const do_reorder = true){
  using dat_T = aprx_binary_mix_cdf_structured_diag;
  return Rcpp::XPtr<dat_T>(
    new dat_T(data, n_threads, gradient, 0L, do_reorder), true);
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

// [[Rcpp::export]]
double aprx_mult_mix_ghq(
    unsigned const n_alt, arma::vec const &eta, arma::mat const &Z,
    arma::mat const &Sigma, unsigned const b,
    bool const is_adaptive = false){
  auto const &rule = GaussHermite::gaussHermiteDataCached(b);
  integrand::multinomial_group integrand(
      n_alt, Z, eta, Sigma);

  return GaussHermite::approx(rule, integrand, is_adaptive);
}

// [[Rcpp::export(rng = false)]]
Rcpp::NumericVector aprx_binary_mix_qmc(
    arma::ivec const &y, arma::vec eta, arma::mat Z,
    arma::mat const &Sigma, unsigned const n_max,
    arma::ivec const &seeds, double const releps,
    bool const is_adaptive = false){
  using Rcpp::NumericVector;
  integrand::mix_binary integrand(y, eta, Z, Sigma);

  auto const res = qmc::approx(
    integrand, is_adaptive, n_max, seeds, releps);
  NumericVector out = NumericVector::create(res.value);
  out.attr("intvls") = res.intvls;
  out.attr("error")  = res.err;

  return out;
}

// [[Rcpp::export(rng = false)]]
Rcpp::NumericVector aprx_mult_mix_qmc(
    unsigned const n_alt, arma::vec const &eta, arma::mat const &Z,
    arma::mat const &Sigma, unsigned const n_max,
    arma::ivec const &seeds, double const releps,
    bool const is_adaptive = false){
  using Rcpp::NumericVector;
  integrand::multinomial_group integrand(n_alt, Z, eta, Sigma);

  auto const res = qmc::approx(
    integrand, is_adaptive, n_max, seeds, releps);
  NumericVector out = NumericVector::create(res.value);
  out.attr("intvls") = res.intvls;
  out.attr("error")  = res.err;

  return out;
}

template<class T>
Rcpp::NumericVector aprx_brute(
    std::vector<T> const &int_objs, unsigned const n_sim, bool const is_is){
  using namespace integrand;
  if(int_objs.size() == 0L)
    throw std::invalid_argument("aprx_brute(): invalid int_obj");
  size_t const p = int_objs[0].get_n_par(),
       n_threads = int_objs.size();
  arma::vec par_vec(p);

  {
  int const n_use = std::max(static_cast<size_t>(1L), n_threads);
#ifdef _OPENMP
  omp_set_num_threads(n_use);
#endif
  parallelrng::set_rng_seeds(n_use);
  }

  welfords out;

  if(is_is){
    struct integrand_worker {
      T const &bin;
      mvn<T> const mvn_obj = mvn<T>(bin);
      adaptive<mvn<T> > const ada =
        adaptive<mvn<T> >(mvn_obj, true);

      integrand_worker(T const &bin): bin(bin) { }
      integrand_worker(integrand_worker const &other):
        integrand_worker(other.bin) { }
    };

    std::vector<integrand_worker> workers;
    workers.reserve(int_objs.size());
    for(auto &x : int_objs)
      workers.emplace_back(x);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(welPlus:out) firstprivate(par_vec)
#endif
    for(unsigned i = 0; i < n_sim; ++i){
      for(unsigned j = 0; j < p; ++j)
        par_vec[j] = rngnorm_wrapper();

#ifdef _OPENMP
      size_t const idx = omp_get_thread_num();
#else
      size_t const idx = 0L;
#endif
      out += antithetic<base_integrand>(workers[idx].ada)(par_vec);
    }

  } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(welPlus:out) firstprivate(par_vec)
#endif
    for(unsigned i = 0; i < n_sim; ++i){
      for(unsigned j = 0; j < p; ++j)
        par_vec[j] = rngnorm_wrapper();

#ifdef _OPENMP
      size_t const idx = omp_get_thread_num();
#else
      size_t const idx = 0L;
#endif
      out += antithetic<base_integrand>(int_objs[idx])(par_vec);
    }
  }

  Rcpp::NumericVector R_out(1);
  R_out[0] = out.mean();
  R_out.attr("SE") = std::sqrt(out.var() / (double)n_sim);
  return R_out;
}

/* brute force MC estimate */
// [[Rcpp::export]]
Rcpp::NumericVector aprx_binary_mix_brute(
    arma::ivec const &y, arma::vec eta, arma::mat Z,
    arma::mat const &Sigma, unsigned const n_sim,
    unsigned const n_threads = 1L, bool const is_is = true){
  std::vector<integrand::mix_binary> objs;
  objs.reserve(n_threads);
  for(size_t i = 0; i < n_threads; ++i)
    objs.emplace_back(y, eta, Z, Sigma);

  return aprx_brute(objs, n_sim, is_is);
}

// [[Rcpp::export]]
Rcpp::NumericVector aprx_mult_mix_brute(
    unsigned const n_alt, arma::vec const &eta, arma::mat const &Z,
    arma::mat const &Sigma, unsigned const n_sim,
    unsigned const n_threads = 1L, bool const is_is = true){
  std::vector<integrand::multinomial_group> objs;
  objs.reserve(n_threads);
  for(size_t i = 0; i < n_threads; ++i)
    objs.emplace_back(n_alt, Z, eta, Sigma);

  return aprx_brute(objs, n_sim, is_is);
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
    for(size_t i = 0; i < n; ++i)
      functor->operator()(out.colptr(i));
    return out;
  }

  return arma::mat(functor->dimen, 0L);
}

/**
 Evaluates the inner integral needed for multinomial outcomes. I.e. the
 conditional density given the random effects. The `n_times` argument is
 added to compare the run times.
 */
// [[Rcpp::export(rng = false)]]
arma::mat multinomial_inner_integral
  (arma::mat const &Z, arma::vec const &eta, arma::mat const &Sigma,
   unsigned const n_nodes, bool const is_adaptive, unsigned const n_times,
   arma::vec const &u, unsigned const order = 0L){
  std::unique_ptr<double[]> wk_mem(new double[Z.n_rows + 3L * Z.n_cols]);

  integrand::multinomial obj(
      Z, eta, arma::chol(Sigma), wk_mem.get(), n_nodes, is_adaptive);

  if(order == 0L){
    double out(0.);
    for(size_t i = 0; i < n_times; ++i)
      out = obj(u.memptr(), false);

    arma::mat m(1, 1);
    m[0] = out;
    return m;

  } else if(order == 1L){
    arma::vec out;

    for(size_t i = 0; i < n_times; ++i)
      out = obj.gr(u.memptr());

    return out;
  }

  arma::mat out;
  for(size_t i = 0; i < n_times; ++i)
    out = obj.Hessian(u.memptr());

  return out;
}

using gsm_vec_ptr = Rcpp::XPtr<std::vector<mixed_gsm_cluster> >;

/**
 * returns a pointer to compute the log marginal likelihood and the gradient for
 * a mixed GSM model.
 */
// [[Rcpp::export(rng = false)]]
SEXP get_gsm_ptr(Rcpp::List data){
  gsm_vec_ptr out(new std::vector<mixed_gsm_cluster>());
  out->reserve(data.size());

  for(SEXP dat : data){
    Rcpp::List dat_list = dat;
    out->emplace_back(
      Rcpp::as<arma::mat>(dat_list["X"]),
      Rcpp::as<arma::mat>(dat_list["X_prime"]),
      Rcpp::as<arma::mat>(dat_list["Z"]),
      Rcpp::as<arma::vec>(dat_list["y"]),
      Rcpp::as<arma::vec>(dat_list["event"]));
  }

  return out;
}

// [[Rcpp::export()]]
Rcpp::NumericVector gsm_eval
  (SEXP ptr, arma::vec const &beta, arma::vec const &sig, int const maxpts,
   int const key, double const abseps, double const releps){
  gsm_vec_ptr comp_obj(ptr);
  double out{};
  unsigned n_fails{};

  arma::uword const n_rng =
    std::lround((std::sqrt(8 * sig.n_elem + 1.) - 1.) / 2);
  arma::mat L(n_rng, n_rng), Sigma;
  get_pd_mat(sig.memptr(), L, Sigma);

  parallelrng::set_rng_seeds(1);
  for(mixed_gsm_cluster &dat : *comp_obj){
    auto res = dat(beta, Sigma, maxpts, key, abseps, releps);
    out += res.log_like;
    n_fails += res.inform != 0;
  }

  Rcpp::NumericVector out_vec{out};
  out_vec.attr("n_fails") = n_fails;

  return out_vec;
}

// [[Rcpp::export()]]
Rcpp::NumericVector gsm_gr
  (SEXP ptr, arma::vec const &beta, arma::vec const &sig, int const maxpts,
   int const key, double const abseps, double const releps){
  gsm_vec_ptr comp_obj(ptr);
  double ll{};
  unsigned n_fails{};
  arma::vec out, tmp;

  arma::uword const n_rng =
    std::lround((std::sqrt(8 * sig.n_elem + 1.) - 1.) / 2);
  arma::mat L(n_rng, n_rng), Sigma;
  get_pd_mat(sig.memptr(), L, Sigma);

  parallelrng::set_rng_seeds(1);
  for(mixed_gsm_cluster &dat : *comp_obj){
    auto res = dat.grad(tmp, beta, Sigma, maxpts, key, abseps, releps);
    ll += res.log_like;
    n_fails += res.inform != 0;
    if(tmp.size() != out.size())
      out = tmp;
    else
      out += tmp;
  }

  arma::mat d_Sigma(n_rng, n_rng);
  {
    double const * d_sig_ij{out.memptr() + beta.n_elem};
    for(arma::uword j = 0; j < n_rng; ++j, ++d_sig_ij){
      for(arma::uword i = 0; i < j; ++i, ++d_sig_ij){
        d_Sigma(i, j) = *d_sig_ij / 2;
        d_Sigma(j, i) = *d_sig_ij / 2;
      }
      d_Sigma(j, j) = *d_sig_ij;
    }
  }

  std::unique_ptr<double[]> wk_mem{new double[dpd_mat::n_wmem(n_rng)]};
  std::fill(out.begin() + beta.n_elem, out.end(), 0);
  dpd_mat::get(L, out.memptr() + beta.n_elem, d_Sigma.memptr(), wk_mem.get());

  Rcpp::NumericVector out_vec(out.n_elem);
  std::copy(out.begin(), out.end(), &out_vec[0]);
  out_vec.attr("n_fails") = n_fails;
  out_vec.attr("logLik") = ll;

  return out_vec;
}
