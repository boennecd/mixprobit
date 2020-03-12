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

    ## [1] 0.0675398
    ## attr(,"error")
    ## [1] 3.87991e-05
    ## attr(,"n_sim")
    ## [1] 46300

``` r
print(my_wrap_2(), digits = 6)
```

    ## [1] 0.0675357
    ## attr(,"error")
    ## [1] 3.88178e-05
    ## attr(,"n_sim")
    ## [1] 120600

``` r
print(cm_wrap  (), digits = 6)
```

    ## [1] 0.0674869
    ## attr(,"error")
    ## [1] 6.78552e-05
    ## attr(,"msg")
    ## [1] "Normal Completion"

``` r
print(cm_wrap_2(), digits = 6)
```

    ## $value
    ## [1] 0.0674821
    ## 
    ## $error
    ## [1] 3.5969e-05
    ## 
    ## $inform
    ## [1] 0

``` r
sd(om  <- replicate(30L, my_wrap  ()))
```

    ## [1] 3.496e-05

``` r
sd(om2 <- replicate(30L, my_wrap_2()))
```

    ## [1] 3.664e-05

``` r
sd(cm  <- replicate(30L, cm_wrap()))
```

    ## [1] 2.954e-05

``` r
sd(cm2 <- replicate(30L, cm_wrap_2()$value))
```

    ## [1] 2.685e-05

``` r
cat(sprintf("%.8f\n%.8f\n%.8f\n%.8f\n%.8f\n%.8f\n%.8f\n%.8f\n", 
            mean(om), mean(om2), mean(cm), mean(cm2), 
            log(mean(om)), log(mean(om2)), log(mean(cm)), log(mean(cm2))))
```

    ## 0.06748464
    ## 0.06749476
    ## 0.06749380
    ## 0.06748732
    ## -2.69585525
    ## -2.69570529
    ## -2.69571957
    ## -2.69581548

``` r
microbenchmark::microbenchmark(
  pmvnorm              = cm_wrap(), 
  `pmvnorm pkg`        = cm_wrap_2(), 
  `my_pmvnorm_cpp`     = my_wrap  (),
  `my_pmvnorm_cpp pkg` = my_wrap_2(),
  times = 100L)
```

    ## Unit: microseconds
    ##                expr     min      lq    mean  median      uq   max neval
    ##             pmvnorm   861.5  1804.5  2151.8  1874.6  1977.5  3413   100
    ##         pmvnorm pkg   304.4   759.7   905.9   768.9   821.5  1530   100
    ##      my_pmvnorm_cpp 22044.3 22346.8 22563.0 22516.0 22694.4 23651   100
    ##  my_pmvnorm_cpp pkg 49397.2 50081.3 50369.6 50309.9 50639.1 51717   100

Clearly, do not directly use the GHK method used by Hajivassiliou, McFadden, and Ruud (1996) or Genz (1992) (two very similar and maybe independent suggestions). Use what Alan Genz has implemented in `mvtnorm::pmvt`. See Niederreiter (1972), Keast (1973), Cranley and Patterson (1976) and the `MVKBRV` subroutine in the `mvt.f` file. Using this subroutine to approximate the derivatives should be fairly straight forward.

References
----------

Cranley, R., and T. N. L. Patterson. 1976. “Randomization of Number Theoretic Methods for Multiple Integration.” *SIAM Journal on Numerical Analysis* 13 (6). Society for Industrial; Applied Mathematics: 904–14. <http://www.jstor.org/stable/2156452>.

Genz, Alan. 1992. “Numerical Computation of Multivariate Normal Probabilities.” *Journal of Computational and Graphical Statistics* 1 (2). Taylor & Francis: 141–49. doi:[10.1080/10618600.1992.10477010](https://doi.org/10.1080/10618600.1992.10477010).

Hajivassiliou, Vassilis, Daniel McFadden, and Paul Ruud. 1996. “Simulation of Multivariate Normal Rectangle Probabilities and Their Derivatives Theoretical and Computational Results.” *Journal of Econometrics* 72 (1): 85–134. doi:[https://doi.org/10.1016/0304-4076(94)01716-6](https://doi.org/https://doi.org/10.1016/0304-4076(94)01716-6).

Keast, P. 1973. “Optimal Parameters for Multidimensional Integration.” *SIAM Journal on Numerical Analysis* 10 (5): 831–38. doi:[10.1137/0710068](https://doi.org/10.1137/0710068).

Niederreiter, Harald. 1972. “On a Number-Theoretical Integration Method.” *Aequationes Mathematicae* 8 (3). Springer: 304–11.
