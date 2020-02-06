Multivariate Normal CDFs and Their Derivatives
==============================================

Setup.

``` r
set.seed(1L)
options(digits = 4)

# sample data
p <- 4L
sigma <- drop(rWishart(1, 2L * p, diag(p)))
mu <- rnorm(p, sd = .1)
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

    ## [1] 0.0678

``` r
pmvnorm(upper = rep(0, p), mean = mu, sigma = sigma)
```

    ## [1] 0.06746
    ## attr(,"error")
    ## [1] 0.0001655
    ## attr(,"msg")
    ## [1] "Normal Completion"

Approximate derivative of CDF w.r.t. the mean.

``` r
my_pmvnorm(mu, sigma, 10000L, h = function(draw, mean, sigma_chol)
  forwardsolve(sigma_chol, draw, upper.tri = TRUE))
```

    ## [1] -0.02723 -0.01722 -0.02594 -0.01064

``` r
library(numDeriv)
drop(jacobian(function(x){
  set.seed(31958429L)
  pmvnorm(upper = rep(0, p), mean = x, sigma = sigma)
}, mu))
```

    ## [1] -0.02721 -0.01730 -0.02598 -0.01097

Approximate derivative of log CDF w.r.t. the mean.

``` r
out <- my_pmvnorm(mu, sigma, 10000L, h = function(draw, mean, sigma_chol)
  c(1, forwardsolve(sigma_chol, draw, upper.tri = TRUE)))
out[-1] / out[1]
```

    ## [1] -0.4065 -0.2534 -0.3874 -0.1623

``` r
drop(jacobian(function(x){
  set.seed(31958429L)
  log(pmvnorm(upper = rep(0, p), mean = x, sigma = sigma))
}, mu))
```

    ## [1] -0.4027 -0.2560 -0.3845 -0.1623

Computation Times
-----------------

Make C++ example and compare computations times.

``` cpp
#include <stdexcept>
#include <array>
#include <limits>

// [[Rcpp::depends("RcppArmadillo")]]
#include <RcppArmadillo.h>

inline std::array<double, 2> draw_trunc_mean
  (double const b, const double u, const bool draw){
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
  arma::vec draw(p);
  unsigned i;
  unsigned const min_run = p * 10L;
  constexpr double alpha(2.5);
  for(i = 0; i < nsim;){
    unsigned const i_max = i + min_run;
    for(; i < i_max; i += 2L){
      auto func = [&](arma::vec const &u){
        double w(1.);
        for(unsigned j = 0; j < p; ++j){
          double b(-mean[j]);
          for(unsigned k = 0; k < j; ++k)
            b -= sigma_chol.at(k, j) * draw[k];
          b /= sigma_chol.at(j, j);
          
          auto const draw_n_p = draw_trunc_mean(b, u[j], j + 1 < p);
          w       *= draw_n_p[0];
          draw[j]  = draw_n_p[1];
        }
        
        /* use Welfords online algorithm */
        double const old_diff = (w - out);
        di  += 1.;
        out += old_diff / di;
        M   += old_diff * (w - out);
      };
      
      arma::vec u(p);
      u.transform([](double val) { return unif_rand(); });
      func(u);
      func(1 - u);
      
    }
    
    if(alpha * std::sqrt(M / di / di) < eps)
      break;
  }
  
  Rcpp::NumericVector ret_val(1L);
  ret_val[0L] = out;
  ret_val.attr("error") = alpha * std::sqrt(M / di / di);
  ret_val.attr("n_sim") = i;
  return ret_val;
}
```

Test the C++ function.

``` r
# approximate CDF
library(mvtnorm)
library(mixprobit)
my_wrap <- function(n_sim)
  my_pmvnorm_cpp(mu, sigma, n_sim, 25e-5)
cm_wrap <- function()
  pmvnorm(
    upper = rep(0, p), mean = mu, sigma = sigma, 
    algorithm = GenzBretz(abseps = 1e-3, releps = -1))
# differs by using R's pnorm and qnorm
cm_wrap_2 <- function()
  mixprobit:::pmvnorm_cpp(
    lower = rep(-Inf, p), 
    upper = rep(0, p), mean = mu, cov = sigma, 
    abseps = 1e-3, releps = -1, maxpts = 25000L)

my_wrap(100000L)
```

    ## [1] 0.06751
    ## attr(,"error")
    ## [1] 0.0002499
    ## attr(,"n_sim")
    ## [1] 51040

``` r
cm_wrap()
```

    ## [1] 0.06759
    ## attr(,"error")
    ## [1] 0.0001496
    ## attr(,"msg")
    ## [1] "Normal Completion"

``` r
cm_wrap_2()
```

    ## $value
    ## [1] 0.0676
    ## 
    ## $error
    ## [1] 0.0001504
    ## 
    ## $inform
    ## [1] 0

``` r
sd(om  <- replicate(1000L, my_wrap(100000L)))
```

    ## [1] 5.251e-05

``` r
sd(cm  <- replicate(1000L, cm_wrap()))
```

    ## [1] 5.675e-05

``` r
sd(cm2 <- replicate(1000L, cm_wrap_2()$value))
```

    ## [1] 5.844e-05

``` r
cat(sprintf("%.8f\n%.8f\n%.8f\n%.8f\n%.8f\n%.8f\n", 
            mean(om), mean(cm), mean(cm2), 
            log(mean(om)), log(mean(cm)), log(mean(cm2))))
```

    ## 0.06749068
    ## 0.06748494
    ## 0.06748627
    ## -2.69576579
    ## -2.69585082
    ## -2.69583107

``` r
microbenchmark::microbenchmark(
  pmvnorm                 = cm_wrap(), 
  `pmvnorm (R funcs)`     = cm_wrap_2(), 
  `my_pmvnorm_cpp 10000`  = my_wrap(10000L ),
  `my_pmvnorm_cpp 100000` = my_wrap(100000L),
  times = 100L)
```

    ## Unit: microseconds
    ##                   expr     min      lq    mean  median      uq     max neval
    ##                pmvnorm   805.9   816.9   853.1   835.6   864.8  1262.9   100
    ##      pmvnorm (R funcs)   301.8   306.7   318.3   313.4   325.0   420.1   100
    ##   my_pmvnorm_cpp 10000  2470.5  2479.2  2530.2  2498.1  2553.1  2855.9   100
    ##  my_pmvnorm_cpp 100000 12442.1 12647.8 12823.0 12765.4 12926.5 13992.8   100

Clearly, do not directly use the GHK method used by Hajivassiliou, McFadden, and Ruud (1996) or Genz (1992) (two very similar and maybe independent suggestions). Use what Alan Genz has implemented in `mvtnorm::pmvt`. See Niederreiter (1972), Keast (1973), Cranley and Patterson (1976) and the `MVKBRV` subroutine in the `mvt.f` file. Using this subroutine to approximate the derivatives should be fairly straight forward.

References
----------

Cranley, R., and T. N. L. Patterson. 1976. “Randomization of Number Theoretic Methods for Multiple Integration.” *SIAM Journal on Numerical Analysis* 13 (6). Society for Industrial; Applied Mathematics: 904–14. <http://www.jstor.org/stable/2156452>.

Genz, Alan. 1992. “Numerical Computation of Multivariate Normal Probabilities.” *Journal of Computational and Graphical Statistics* 1 (2). Taylor & Francis: 141–49. doi:[10.1080/10618600.1992.10477010](https://doi.org/10.1080/10618600.1992.10477010).

Hajivassiliou, Vassilis, Daniel McFadden, and Paul Ruud. 1996. “Simulation of Multivariate Normal Rectangle Probabilities and Their Derivatives Theoretical and Computational Results.” *Journal of Econometrics* 72 (1): 85–134. doi:[https://doi.org/10.1016/0304-4076(94)01716-6](https://doi.org/https://doi.org/10.1016/0304-4076(94)01716-6).

Keast, P. 1973. “Optimal Parameters for Multidimensional Integration.” *SIAM Journal on Numerical Analysis* 10 (5): 831–38. doi:[10.1137/0710068](https://doi.org/10.1137/0710068).

Niederreiter, Harald. 1972. “On a Number-Theoretical Integration Method.” *Aequationes Mathematicae* 8 (3). Springer: 304–11.
