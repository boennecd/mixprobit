Multivariate Normal CDFs and Their Derivatives
==============================================

Setup.

``` r
set.seed(1L)
options(digits = 4)

# sample data
get_sim <- function(){
  within(list(), {
    p <- 4L
    sigma <- drop(rWishart(1, 2L * p, diag(p)))
    mu <- drop(crossprod(chol(sigma), rnorm(p)))
  })
}

invisible(list2env(get_sim(), envir = environment()))
```

``` r
#####
# define function to approximate rectangular multivariate normal CDFs
# where the lower bound are -Inf and upper bounds are zero
my_pmvnorm <- function(mean, sigma, nsim, h = function(...) 1){
  #####
  # setup
  n <- length(mean)
  sigma_chol <- chol(sigma)
  draw_trunc_mean <- function(b, u)
    qnorm(pnorm(b) * u)
  nm1 <- n - 1L

  #####
  # simulate
  sims <- replicate(nsim, {
    u <- runif(n)
    draw <- numeric(n)
    out <- 1.

    for(i in 1:n){
      b <- if(i == 1L)
        -mean[i]
      else
        -mean[i] - drop(sigma_chol[1:(i - 1L), i] %*% draw[1:(i - 1L)])
      b <- b / sigma_chol[i, i]

      draw[i] <- draw_trunc_mean(b, u[i])

      out <- out * pnorm(b)
    }

    h(draw, mean, sigma_chol) * out
  }, simplify = "array")

  # return MC estimate
  if(is.vector(sims))
    return(mean(sims))
  rowMeans(sims)
}
```

Approximate CDF.

``` r
library(mvtnorm)
my_pmvnorm(mu, sigma, 10000L)
```

    ## [1] 0.1959

``` r
pmvnorm(upper = rep(0, p), mean = mu, sigma = sigma)
```

    ## [1] 0.195
    ## attr(,"error")
    ## [1] 0.0005886
    ## attr(,"msg")
    ## [1] "Normal Completion"

Approximate derivative of CDF w.r.t. the mean.

``` r
my_pmvnorm(mu, sigma, 10000L, h = function(draw, mean, sigma_chol)
  forwardsolve(sigma_chol, draw, upper.tri = TRUE))
```

    ## [1] -0.081467 -0.021322 -0.007201 -0.034268

``` r
library(numDeriv)
drop(jacobian(function(x){
  set.seed(31958429L)
  pmvnorm(upper = rep(0, p), mean = x, sigma = sigma)
}, mu))
```

    ## [1] -0.081226 -0.021237 -0.007224 -0.034509

Approximate derivative of log CDF w.r.t. the mean.

``` r
out <- my_pmvnorm(mu, sigma, 10000L, h = function(draw, mean, sigma_chol)
  c(1, forwardsolve(sigma_chol, draw, upper.tri = TRUE)))
out[-1] / out[1]
```

    ## [1] -0.42344 -0.10468 -0.04087 -0.17686

``` r
drop(jacobian(function(x){
  set.seed(31958429L)
  log(pmvnorm(upper = rep(0, p), mean = x, sigma = sigma))
}, mu))
```

    ## [1] -0.41640 -0.10887 -0.03703 -0.17691

Computation Times
-----------------

Make C++ example and compare computations times.

``` cpp
#include <stdexcept>
#include <array>
#include <limits>

// [[Rcpp::depends("RcppArmadillo")]]
#include <RcppArmadillo.h>

template<bool draw>
inline std::array<double, 2> draw_trunc_mean
  (double const b, const double u){
  double const qb = R::pnorm5(b, 0, 1, 1L, 0L);
  if(draw)
    return { qb, R::qnorm5(qb * u, 0, 1, 1L, 0L) };
  return { qb, std::numeric_limits<double>::quiet_NaN() };
}

// [[Rcpp::export]]
Rcpp::NumericVector my_pmvnorm_cpp(
    arma::vec const &mean_in, arma::mat const &sigma_in, 
    unsigned const nsim, double const eps){
  arma::uword const p = mean_in.n_elem;
  if(sigma_in.n_cols != p or sigma_in.n_rows != p)
    throw std::invalid_argument("invalid sigma");
  
  /* re-scale */
  arma::vec const sds = arma::sqrt(arma::diagvec(sigma_in));
  arma::vec const mean = mean_in / sds;
  
  arma::mat const sigma = ([&](){
    arma::mat out = sigma_in;
    out.each_row() /= sds.t();
    out.each_col() /= sds;
    
    return out;
  })();
  
  arma::mat const sigma_chol = arma::chol(sigma);
  
  double out(0.), di(0.), M(0.);
  arma::vec draw(p), u(p);
  unsigned i;
  unsigned const min_run(p * 25L);
  constexpr double alpha(2.576);
  for(i = 0; i < nsim;){
    unsigned const i_max = i + min_run;
    for(; i < i_max; ++i){
      auto func = [&](arma::vec const &u){
        double w(1.);
        unsigned j;
        for(j = 0; j < p - 1; ++j){
          double b(-mean[j]);
          for(unsigned k = 0; k < j; ++k)
            b -= sigma_chol.at(k, j) * draw[k];
          b /= sigma_chol.at(j, j);
          
          auto const draw_n_p = draw_trunc_mean<true>(b, u[j]);
          w       *= draw_n_p[0];
          draw[j]  = draw_n_p[1];
        }
        
        double b(-mean[j]);
        for(unsigned k = 0; k < j; ++k)
          b -= sigma_chol.at(k, j) * draw[k];
        b /= sigma_chol.at(j, j);
        
        auto const draw_n_p = draw_trunc_mean<false>(b, u[j]);
        w *= draw_n_p[0];
        
        return w;
      };
      
      u.for_each([](double &val) { return val = unif_rand(); });
      double w = func(u);
      u.for_each([](double &val) { return val = 1 - val; });
      w += func(u);
      w /= 2;
      
      /* use Welfords online algorithm */
      double const old_diff = (w - out);
      di  += 1.;
      out += old_diff / di;
      M   += old_diff * (w - out);
      
    }
    
    if(alpha * std::sqrt(M / di / di) < eps)
      break;
  }
  
  Rcpp::NumericVector ret_val(1L);
  ret_val[0L] = out;
  ret_val.attr("error") = std::sqrt(M / di / di);
  ret_val.attr("n_sim") = i;
  return ret_val;
}
```

We also use the implementation in the package. Test the C++ function.

``` r
# approximate CDF
library(mvtnorm)
library(mixprobit)
abs_eps <- 1e-4
n_sim <- 10000000L

my_wrap <- function()
  my_pmvnorm_cpp(mu, sigma, n_sim, abs_eps)
cm_wrap <- function()
  pmvnorm(
    upper = rep(0, p), mean = mu, sigma = sigma, 
    algorithm = GenzBretz(abseps = abs_eps, releps = -1, 
                          maxpts = n_sim))
# differs by using R's pnorm and qnorm
my_wrap_2 <- function()
  mixprobit:::my_pmvnorm_cpp(mean_in = mu, sigma_in = sigma, 
                             eps = abs_eps, nsim = n_sim)
cm_wrap_2 <- function()
  mixprobit:::pmvnorm_cpp(
    lower = rep(-Inf, p), 
    upper = rep(0, p), mean = mu, cov = sigma, 
    abseps = abs_eps, releps = -1, maxpts = n_sim)

print(my_wrap  (), digits = 6)
```

    ## [1] 0.195295
    ## attr(,"error")
    ## [1] 3.88132e-05
    ## attr(,"n_sim")
    ## [1] 143700

``` r
print(my_wrap_2(), digits = 6)
```

    ## [1] 0.19528
    ## attr(,"error")
    ## [1] 3.88137e-05
    ## attr(,"n_sim")
    ## [1] 75400

``` r
print(cm_wrap  (), digits = 6)
```

    ## [1] 0.195207
    ## attr(,"error")
    ## [1] 7.94508e-05
    ## attr(,"msg")
    ## [1] "Normal Completion"

``` r
print(cm_wrap_2(), digits = 6)
```

    ## $value
    ## [1] 0.195243
    ## 
    ## $error
    ## [1] 3.1716e-05
    ## 
    ## $inform
    ## [1] 0

``` r
sd(om  <- replicate(30L, my_wrap  ()))
```

    ## [1] 4.889e-05

``` r
sd(om2 <- replicate(30L, my_wrap_2()))
```

    ## [1] 4.051e-05

``` r
sd(cm  <- replicate(30L, cm_wrap()))
```

    ## [1] 2.622e-05

``` r
sd(cm2 <- replicate(30L, cm_wrap_2()$value))
```

    ## [1] 2.524e-05

``` r
cat(sprintf("%.8f\n%.8f\n%.8f\n%.8f\n%.8f\n%.8f\n%.8f\n%.8f\n", 
            mean(om), mean(om2), mean(cm), mean(cm2), 
            log(mean(om)), log(mean(om2)), log(mean(cm)), log(mean(cm2))))
```

    ## 0.19524147
    ## 0.19524973
    ## 0.19524527
    ## 0.19524364
    ## -1.63351819
    ## -1.63347589
    ## -1.63349873
    ## -1.63350708

``` r
# run benchmark 
set.seed(2)
bench_data <- replicate(100, {
  list2env(get_sim(), parent.env(environment()))

  sapply(list(
    `pmvnorm`        = cm_wrap, 
    `pmvnorm pkg`    = cm_wrap_2, 
    `my pmvnorm`     = my_wrap, 
    `my pmvnorm pkg` = my_wrap_2
  ), function(func){
    system.time(replicate(10L, func())) / 5L
  })
})

# show computation times
apply(bench_data["user.self", , ], 1, function(x)
  c(mean = mean(x), sd = sd(x), quantile(x, seq(0, 1, length.out = 5))))
```

    ##       pmvnorm pmvnorm pkg my pmvnorm my pmvnorm pkg
    ## mean 0.002130   0.0010100    0.09235        0.02871
    ## sd   0.001411   0.0007797    0.28950        0.12158
    ## 0%   0.001600   0.0006000    0.00000        0.00000
    ## 25%  0.001800   0.0008000    0.00040        0.00020
    ## 50%  0.001800   0.0008000    0.00640        0.00130
    ## 75%  0.001800   0.0010000    0.05380        0.01535
    ## 100% 0.014200   0.0076000    1.98780        1.09320

Clearly, do not directly use the GHK method used by Hajivassiliou, McFadden, and Ruud (1996) or Genz (1992) (two very similar and maybe independent suggestions). The main difference between `my pmvnorm` and `my pmvnorm pkg` is that the latter reorders the variables. Use what Alan Genz has implemented in `mvtnorm::pmvt`. See Niederreiter (1972), Keast (1973), Cranley and Patterson (1976) and the `MVKBRV` subroutine in the `mvt.f` file. Using this subroutine to approximate the derivatives should be fairly straight forward.

References
----------

Cranley, R., and T. N. L. Patterson. 1976. “Randomization of Number Theoretic Methods for Multiple Integration.” *SIAM Journal on Numerical Analysis* 13 (6). Society for Industrial; Applied Mathematics: 904–14. <http://www.jstor.org/stable/2156452>.

Genz, Alan. 1992. “Numerical Computation of Multivariate Normal Probabilities.” *Journal of Computational and Graphical Statistics* 1 (2). Taylor & Francis: 141–49. doi:[10.1080/10618600.1992.10477010](https://doi.org/10.1080/10618600.1992.10477010).

Hajivassiliou, Vassilis, Daniel McFadden, and Paul Ruud. 1996. “Simulation of Multivariate Normal Rectangle Probabilities and Their Derivatives Theoretical and Computational Results.” *Journal of Econometrics* 72 (1): 85–134. doi:[https://doi.org/10.1016/0304-4076(94)01716-6](https://doi.org/https://doi.org/10.1016/0304-4076(94)01716-6).

Keast, P. 1973. “Optimal Parameters for Multidimensional Integration.” *SIAM Journal on Numerical Analysis* 10 (5): 831–38. doi:[10.1137/0710068](https://doi.org/10.1137/0710068).

Niederreiter, Harald. 1972. “On a Number-Theoretical Integration Method.” *Aequationes Mathematicae* 8 (3). Springer: 304–11.
