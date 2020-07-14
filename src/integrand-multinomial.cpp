#include "integrand-multinomial.h"
#include "Brent_fmin.h"
#include "pnorm.h"

using std::log;
using std::exp;
using std::abs;

namespace integrand {
static double const norm_const = 1. / std::sqrt(M_PI * 2.),
                norm_const_log = log(norm_const);

inline double my_log_dnorm(double const x){
  return norm_const_log - .5 * x * x;
}

/** the class assumes that the `lp` object has been set before calling its
    member functions. It then computes the log of the "inner" integrand and
    its second derivative. */
class multinomial_mode_helper {
  multinomial const &int_obj;
  arma::vec const &lp = int_obj.lp;

  /*
   CDF approximation from

   xup <- 0
   dput(off <- pnorm(xup, log = TRUE))
   xs <- seq(-8, xup, length.out = 1000)
   xt <- xs - xup
   fit <- lm(pnorm(xs, log.p = TRUE) ~ poly(xt, raw = TRUE, degree = 3) - 1,
   offset = rep(off, length(xs)))
   xhat <- seq(-9, xup, length.out = 1000)
   matplot(xhat, cbind(pnorm(xhat, log.p = TRUE),
   predict(fit, newdata = data.frame(xt = xhat - xup))),
   type = "l")
   xhat <- seq(xup - 1, xup, length.out = 1000)
   matplot(xhat, cbind(pnorm(xhat, log.p = TRUE),
   predict(fit, newdata = data.frame(xt = xhat - xup))),
   type = "l")
   max(abs(fit$residuals))
   dput(coef(fit))

   xs <- seq(xup, 8, length.out = 1000)
   xt <- xs - xup
   fit <- lm(pnorm(xs, log.p = TRUE) ~ poly(xt, raw = TRUE, degree = 5) - 1,
   offset = rep(off, length(xs)))
   xhat <- seq(xup, 8, length.out = 1000)
   matplot(xhat, cbind(pnorm(xhat, log.p = TRUE),
   predict(fit, newdata = data.frame(xt = xhat - xup))),
   type = "l")
   max(abs(fit$residuals))
   dput(coef(fit))

   */
  inline double norm_cdf_aprx(double const x) const {
    constexpr double const xup = 0,
                         inter = -0.693147180559945;

    if(x > -8 and x < 8){
      if(x >= xup) {
        double const xd = x - xup;
        double out(inter),
                xp = xd;
        out += 0.811401689963717 * xp;
        xp *= xd;
        out -= 0.36464495501633 * xp;
        xp *= xd;
        out += 0.0784688043503829 * xp;
        xp *= xd;
        out -= 0.00809951215678632 * xp;
        xp *= xd;
        out += 0.000321901567031593 * xp;
        return out;

      } else {
        double const xd = x - xup;
        double out(inter),
                xp = xd;
        out += 0.7633873031817 * xp;
        xp *= xd;
        out -= 0.365089939925538 * xp;
        xp *= xd;
        out += 0.0146225647154304 * xp;
        xp *= xd;
        out += 0.000646820246088735 * xp;
        return out;

      }
    }

    return pnorm_std(x, 1L, 1L);
  }

public:
  multinomial_mode_helper(multinomial const &int_obj): int_obj(int_obj) { }

  /** computes the log of the integrand */
  double fn(double const a) const {
    double out = - a * a / 2.; // R::dnorm4(a, 0, 1, 1L);
    for(auto l : lp)
      out += norm_cdf_aprx(l + a);
    return out;
  }

  /** computes the second order derivative of log of the integrand */
  double he(double const a) const {
    double out = -1;
    for(auto l : lp){
      double const val = l + a,
              dnrm_log = my_log_dnorm(val), // R::dnorm4(val, 0, 1,     1L),
              pnrm_log = norm_cdf_aprx(val),
                   rat = std::exp(dnrm_log - pnrm_log);
      out -= rat * (val + rat);
    }

    return out;
  }
};

inline double multinomial_optfunc(double a, void *ptr){
  multinomial_mode_helper *obj =
    static_cast<multinomial_mode_helper*>(ptr);

  return -obj->fn(a);
}

/* TODO: we can cache the value in case we make multiple calls to the
         function w/ the same argument */
multinomial::mode_res multinomial::find_mode(double const *par) const {
  if(!is_adaptive)
    return multinomial::mode_res();

  memcpy(par_vec.begin(), par, sizeof(double) * n_par);
  for(unsigned i = 0; i < n_alt; ++i)
    lp[i] = eta[i] - colvecdot(Z, i, par_vec);
  multinomial_mode_helper helper(*this);

  /* settings for Brent */
  double delta = delta_old;
  double x_min = mode_old - delta / 2,
         x_max = mode_old + delta / 2,
         res;
  double const tol = 1e-3;

  /* reset default values */
  mode_old = 0.;
  delta_old = 5.16;

  /* perform minimization */
  constexpr size_t const  max_it = 30L;
  size_t it;
  for(it = 0; it < max_it; ++it){
    res =  Brent_fmin(x_min, x_max, multinomial_optfunc,
                      static_cast<void*>(&helper), tol);

    double const eps =
      std::numeric_limits<double>::epsilon() * abs(res) + tol / 2.;
    if       (abs(res - x_min) < eps){
      x_max = x_min + .01 * delta;
      delta *= 5;
      x_min = x_max - delta;

    } else if(abs(res - x_max) < eps){
      x_min = x_max - .01 * delta;
      delta *= 5;
      x_max = x_min + delta;

    } else
      break;
  }

  if(it >= max_it)
    return multinomial::mode_res();

  mode_old = res;
  delta_old = 1;

  /* compute second order derivative and return */
  double const he = helper.he(res);
  if(he >= 0)
    return multinomial::mode_res();

  return { res, std::sqrt(-1. / he), true };
}

double multinomial::operator()
(double const *par, bool const ret_log) const {
  auto const m_res = find_mode(par);
  double const sqrt_2 = std::sqrt(2),
                scale = m_res.found_mode ? sqrt_2 *  m_res.scale : sqrt_2,
             location = m_res.found_mode ? m_res.location : 0;
  memcpy(par_vec.begin(), par, sizeof(double) * n_par);

  /* set the fixed part of the linear predictor */
  for(unsigned i = 0; i < n_alt; ++i)
    lp[i] = eta[i] - colvecdot(Z, i, par_vec);

  double out(-std::numeric_limits<double>::infinity());
  for(unsigned i = 0; i < n_nodes; ++i){
    double const a = scale * x[i] + location;

    double new_term = w_log[i] - a * a / 2. + x[i] * x[i];
    for(unsigned j = 0; j < n_alt; ++j)
      new_term += pnorm_std(a + lp[j], 1L, 1L);

    out = log_exp_add(out, new_term);
  }
  out += norm_const_log + log(scale);

  if(ret_log)
    return out;
  return exp(out);
}

/**
 Let e_i(a) = a + \eta_i + x_i. We use that

 \begin{align*}
 h(\vec x) &= \log \int \phi(a)\prod_{k' \ne k}\Phi(e_{k'}(a)) da \\
 \frac{\partial}{\partial x_i}h(\vec x) &=
 \frac{
 \int \phi(a)\phi(e_i(a))
 \prod_{k'\ne k,i}\Phi(e_{k'}(a)) da}{
 \int \phi(a)\prod_{k' \ne k}\Phi(e_{k'}(a)) da}
 \end{align*}

 */
arma::vec multinomial::gr(double const *par) const {
  auto const m_res = find_mode(par);
  double const sqrt_2 = std::sqrt(2),
                scale = m_res.found_mode ? sqrt_2 *  m_res.scale : sqrt_2,
             location = m_res.found_mode ? m_res.location : 0;

  memcpy(par_vec.begin(), par, sizeof(double) * n_par);
  arma::vec gr(n_par, arma::fill::zeros);

  /* we use these to store partial derivatives */
  arma::vec &partial = wk1,
      &partial_inner = wk2;
  partial.fill(-std::numeric_limits<double>::infinity());

  /* set the fixed part of the linear predictor */
  for(unsigned i = 0; i < n_alt; ++i)
    lp[i] = eta[i] - colvecdot(Z, i, par_vec);

  double denom(-std::numeric_limits<double>::infinity());
  for(unsigned i = 0; i < n_nodes; ++i){
    double const a = scale * x[i] + location,
         start_val = w_log[i] - a * a / 2. + x[i] * x[i];

    double new_term_denom(start_val);
    for(unsigned j = 0; j < n_alt; ++j){
      double const lpj = a + lp[j],
              dnrm_log = my_log_dnorm(lpj),
              pnrm_log = pnorm_std(lpj, 1L, 1L);

      new_term_denom += pnrm_log;
      partial_inner[j] = dnrm_log - pnrm_log;
    }

    denom = log_exp_add(denom, new_term_denom);
    for(unsigned k = 0; k < n_alt; ++k)
      partial[k] = log_exp_add(
        partial[k], partial_inner[k] + new_term_denom);
  }

  partial -= denom;
  for(unsigned i = 0; i < n_alt; ++i)
    gr -= exp(partial[i]) * Z.col(i);

  return gr;
}

/**
 For the diagonal elements simply use that

   d^2/dx^2 log(f(-x)) = (f(-x) f''(-x) - f'(x)^2) / f(x)^2
                       = f'(x) (-x f(-x) - f'(x)) / f(x)^2

 to show

 \begin{align*}
 h(\vec x) &= \log \int \phi(a)\prod_{k' \ne k}\Phi(e_{k'}(a))) da \\
 \frac{\partial^2}{\partial x_i^2}h(\vec x) &=
 \bigg(
 -\left[\int \phi(a)e_i(a)\phi(e_{i}(a))
 \prod_{k'\ne k,i}\Phi(e_{k'}(a))da\right]
 \left[\int \phi(a)\prod_{k' \ne k}\Phi(e_{k'}(a)) da\right]\\
 &\hspace{20pt}
 -\left[\int \phi(a)\phi(e_i(a))
 \prod_{k''\ne k,i}\Phi(e_{k'}(a))da\right]^2\bigg) \\
 &\hspace{20pt}\bigg/\left[\int \phi(a)\prod_{k' \ne k}\Phi(e_{k'}(a)) da\right]^2
 \end{align*}

 For the off-diagonal elements we use that

   d / dx f(x) / g(x) = (g(x)f'(x) - f(x)g'(x)) / g(x)^2

 To find that,

 \begin{align*}
 h(\vec x) &= \log \int \phi(a)\prod_{k' \ne k}\Phi(e_{k'}(a)) da \\
 \frac{\partial}{\partial x_i}h(\vec x) &=
 \frac{
 \int \phi(a)\phi(e_i(a))
 \prod_{k'\ne k,i}\Phi(e_{k'}(a)) da}{
 \int \phi(a)\prod_{k' \ne k}\Phi(e_{k'}(a)) da} \\
 \frac{\partial^2}{\partial x_i\partial x_j}h(\vec x) &=
 \bigg(
 \left[\int \phi(a)\prod_{k' \ne k}\Phi(e_{k'}(a)) da\right]
 \left[\int \phi(a)\phi(e_i(a))\phi(e_j(a))
 \prod_{k'\ne k,i,j}\Phi(e_{k'}(a))da\right] \\
 &\hspace{20pt} -
 \left[\int \phi(a)\phi(e_i(a))
 \prod_{k'\ne k,i}\Phi(e_{k'}(a))da\right]
 \left[\int \phi(a)\phi(e_j(a))
 \prod_{k'\ne k,j}\Phi(e_{k'}(a))da\right]\bigg) \\
 &\hspace{20pt}\bigg/\left[\int \phi(a)\prod_{k' \ne k}\Phi(e_{k'}(a)) da\right]^2
 \end{align*}

 Thus, we can do the following. Make one pass storing

   f: \int \phi(a)\prod_{k' \ne k}\Phi(e_{k'}(a)) da
      (one double)
   fp: \int \phi(a)\phi(e_i(a))\prod_{k'\ne k,i}\Phi(e_{k'}(a))da
       (K doubles)
   fppc: \int \phi(a)e_i(a)\phi(e_{i}(a))\prod_{k'\ne k,i}\Phi(e_{k'}(a))da
         (K doubles)
   fppd: \int \phi(a)\phi(e_i(a))\phi(e_j(a))\prod_{k'\ne k,i,j}\Phi(e_{k'}(a))da
         (K(K - 1) / 2 doubles)

 where K is the number of categories less one. Then fill in a K x K matrix
 in the lower triangular. Copy to the upper part and compute the Hessian
 use Z. This will not require too much memory when K is small.

 */
arma::mat multinomial::Hessian(double const *par) const {
  auto const m_res = find_mode(par);
  double const sqrt_2 = std::sqrt(2),
                scale = m_res.found_mode ? sqrt_2 *  m_res.scale : sqrt_2,
             location = m_res.found_mode ? m_res.location : 0;

  memcpy(par_vec.begin(), par, sizeof(double) * n_par);

  /* set the fixed part of the linear predictor */
  for(unsigned i = 0; i < n_alt; ++i)
    lp[i] = eta[i] - colvecdot(Z, i, par_vec);

  /* compute intermediaries */
  arma::vec &fp = wk1,
            &fppc = wk2,
            dnrms_log (n_alt),
            pnrms_log (n_alt);
  arma::mat wk_mat = arma::mat(n_alt, n_alt, arma::fill::zeros);

  fp.zeros();
  fppc.zeros();
  double denom(0.);
  for(unsigned i = 0; i < n_nodes; ++i){
    double const a = scale * x[i] + location;

    /* compute dnorm and pnorms */
    for(unsigned j = 0; j < n_alt; ++j){
      double const lpj = a + lp[j];
      dnrms_log[j] = my_log_dnorm(lpj);
      pnrms_log[j] = pnorm_std(lpj, 1L, 1L);
    }

    double new_term_denom(w[i] * exp(- a * a / 2. + x[i] * x[i]));
    for(unsigned j = 0; j < n_alt; ++j)
      new_term_denom *= exp(pnrms_log[j]);
    denom += new_term_denom;

    for(unsigned k = 0; k < n_alt; ++k){
      double const fac_k = exp(dnrms_log[k] - pnrms_log[k]);

      /* fill in the kth column */
      for(unsigned j = k + 1; j < n_alt; ++j){
        double const fac_j = exp(dnrms_log[j] - pnrms_log[j]);
        wk_mat.at(j, k) += fac_k * fac_j * new_term_denom;
      }

      double const lpk = a + lp[k];
      fp  [k] +=       fac_k * new_term_denom;
      fppc[k] += lpk * fac_k * new_term_denom;
    }
  }

  for(unsigned k = 0; k < n_alt; ++k){
    /* the diagonal entry */
    double const F = denom,
              fp_k = fp  [k],
            fppc_k = fppc[k];
    wk_mat.at(k, k) = -(fppc_k * F + fp_k * fp_k) / F / F;

    /* the off-diagonal entries */
    for(unsigned j = k + 1; j < n_alt; ++j){
      double const fp_j = fp[j];
      wk_mat.at(k, j) = (F * wk_mat.at(j, k) - fp_k * fp_j) / F / F;
    }
  }

  return (Z * arma::symmatu(wk_mat)) * Z.t();
}


double multinomial_group::operator()
  (double const *par, bool const ret_log) const {
  double out(0.);
  for(auto &x : factors)
    out += x(par, true);

  if(ret_log)
    return out;
  return exp(out);
}

arma::vec multinomial_group::gr(double const *par) const {
  arma::vec out(n_par, arma::fill::zeros);
  for(auto &x : factors)
    out += x.gr(par);
  return out;
}

arma::mat multinomial_group::Hessian(double const *par) const {
  arma::mat out(n_par, n_par, arma::fill::zeros);
  for(auto &x : factors)
    out += x.Hessian(par);
  return out;
}

} // namespace integrand
