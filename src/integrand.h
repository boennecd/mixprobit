#ifndef INTEGRAND_H
#define INTEGRAND_H
#include "arma-wrap.h"
#include "mix-optim.h"
#include "utils.h"

namespace integrand {
/* base class to use for the integral approximations like with ranrth */
class base_integrand {
public:
  /* returns the integrand value at a given point */
  virtual double operator()
  (double const*, bool const ret_log = false) const = 0;

  /* returns the graient of the log integrand at a given point */
  virtual arma::vec gr(double const*) const {
    throw std::logic_error("Function not yet implemented");

    return arma::vec();
  }

  /* returns the Hessian of the log integrand at a given point */
  virtual arma::mat Hessian(double const*) const {
    throw std::logic_error("Function not yet implemented");

    return arma::mat();
  }

  /* returns the dimension of the integral */
  virtual std::size_t get_n_par() const = 0;

  /* returns the integral approximation as the first element and the
   * Jacobian w.r.t. the model parameters in the remaining elements */
  virtual void Jacobian(double const*, arma::vec&) const {
    throw std::logic_error("Function not yet implemented");
  }

  /* returns the dimension of the Jacobian */
  virtual std::size_t get_n_jac() const {
    throw std::logic_error("Function not yet implemented");

    return 0L;
  }

  ~base_integrand() = default;
};

/* adds an iid normal distiribution factor to the integrand */
template <typename other_integrand>
class mvn final : public base_integrand {
  other_integrand const &other_terms;

  double log_integrand_factor(double const *par) const {
    static double const log2pi = std::log(2.0 * M_PI);

    std::size_t const n_par = other_terms.get_n_par();
    double out(-(double)n_par/2.0 * log2pi);
    for(unsigned i = 0; i < n_par; ++i)
      out -= *(par + i) * *(par + i) / 2.;

    return out;
  }

public:
  mvn(other_integrand const &other_terms): other_terms(other_terms) { }

  double operator()
  (double const *par, bool const ret_log = false) const {
    double out = log_integrand_factor(par);
    out += other_terms(par, true);

    if(ret_log)
      return out;

    return std::exp(out);
  }

  arma::vec gr(double const *par) const {
    arma::vec out = other_terms.gr(par);
    std::size_t const n_par = other_terms.get_n_par();
    for(unsigned i = 0; i < n_par; ++i)
      out[i] -= *(par + i);

    return out;
  }

  arma::mat Hessian(double const *par) const {
    arma::mat out =  other_terms.Hessian(par);
    out.diag() -= 1;

    return out;
  }

  std::size_t get_n_par() const {
    return other_terms.get_n_par();
  }

  void Jacobian(double const *par, arma::vec &jac) const {
    double const fac = std::exp(log_integrand_factor(par));
    other_terms.Jacobian(par, jac);
    jac *= fac;
  }

  std::size_t get_n_jac() const {
    return other_terms.get_n_jac();
  }
};

/* yields a new integrand of the form
 * pvnorm(x, mode, -Hessian^-1) * f(x) / pvnorm(x, mode, -Hessian^-1) =
     pvnorm(x) * f(mode - Hessian^-1/2 * x) / pvnorm(x) */
template <typename other_integrand>
class adaptive final : public base_integrand {

  /* compute terms from multivariate normal cumulative distribution
   * function and transform the x vector for subsequent calls */
  double log_integrand_factor(arma::vec &x) const {
    double out(arma::dot(x, x) / 2. + dat.constant);

    inplace_tri_mat_mult(x, dat.neg_hes_inv_chol);
    x += dat.mode;

    return out;
  }

public:
  other_integrand const &other_terms;

  struct opt_data {
    typedef adaptive<other_integrand>::opt_data my_kind;
    other_integrand const &other_terms;
    arma::vec mode;
    arma::mat neg_hes_inv_chol;
    double constant;
    bool success = true;

    /* functions used in the optimization */
    static double optim_obj(int npar, double *point, void *data) {
      my_kind dat = *(my_kind*)data;

      return -dat.other_terms(point, true);
    }
    static void optim_gr(int npar, double *point, double *grad, void *data){
      my_kind dat = *(my_kind*)data;
      arma::vec gr(grad, npar, false, true);
      gr = -dat.other_terms.gr(point);
    }

    /* mode, negative Hessian, etc. are set in the constructor */
    opt_data(other_integrand const &other_terms, int const max_it = 10000L,
             double const abstol = -1, double const reltol = 1e-5):
      other_terms(other_terms) {
      arma::uword const n_par = other_terms.get_n_par();
      arma::vec start(n_par, arma::fill::zeros);

      {
        double const integrand_start_val =
          optim_obj((int)n_par, start.begin(), (void*)this);
        if(std::isnan(integrand_start_val)){
          Rcpp::warning("adaptive: invalid starting value");
          success = false;
          return;
        }
      }

      auto const opt_res = optimizers::bfgs(
        start, optim_obj, optim_gr, (void*)this, max_it, 0L, abstol, reltol);

      if(opt_res.fail != 0L)
        throw std::runtime_error("integrand::adaptive: fail != 0L");

      mode = opt_res.par;
      if(!arma::chol(
          neg_hes_inv_chol,
          arma::inv(-other_terms.Hessian(mode.memptr())))){
        Rcpp::warning("adaptive: Cholesky decomposition failed");
        success = false;
        return;

      }

      static double const log2pi = std::log(2.0 * M_PI);
      constant = (double)neg_hes_inv_chol.n_cols / 2. * log2pi;
      for(unsigned i = 0; i < n_par; ++i)
        constant += std::log(neg_hes_inv_chol.at(i, i));
    }
  };
  opt_data const dat;

  adaptive(other_integrand const &other_terms, int const max_it = 10000L,
           double const abstol = -1, double const reltol = 1e-5):
    other_terms(other_terms), dat(other_terms, max_it, abstol, reltol) { }

  double operator()
    (double const *par, bool const ret_log = false) const {
    if(!dat.success)
      return other_terms(par, ret_log);

    arma::vec x(par, other_terms.get_n_par());
    double out = log_integrand_factor(x);
    out += other_terms(x.memptr(), true);

    if(ret_log)
      return out;
    return std::exp(out);
  }

  std::size_t get_n_par() const {
    return other_terms.get_n_par();
  }

  void Jacobian(double const *par, arma::vec &jac) const {
    arma::vec x(par, other_terms.get_n_par());
    double const fac = std::exp(log_integrand_factor(x));
    other_terms.Jacobian(x.memptr(), jac);
    jac *= fac;
  }

  std::size_t get_n_jac() const {
    return other_terms.get_n_jac();
  }
};
} // namespace integrand
#endif
