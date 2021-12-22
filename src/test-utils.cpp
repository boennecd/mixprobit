#include <testthat.h>
#include <limits>
#include "utils.h"

context("utils tests") {
  test_that("dcond_vcov works as expected") {
    /*
     set.seed(1)
     n <- 5L
     k <- 2L
     Sigma <- drop(rWishart(1, 2L * n, diag(n)))
     Z <- matrix(rnorm(n * k), k)

     dput(Sigma)
     dput(crossprod(Z))

     get_H <- \(x) crossprod(Z) + solve(matrix(x, n, n))
     g <- \(x) sum(exp(x))
     f <- \(x) get_H(x) |> solve() |> g()

     f(Sigma)
     library(numDeriv)
     dF <- grad(f, Sigma)

     H_inv <- get_H(Sigma) |> solve()
     dH_inv <- grad(g, H_inv)
     dput(dH_inv)

     H <- get_H(Sigma)
     comp_res <- matrix(dH_inv, n) |>
     solve(a = H) |>
     solve(a = Sigma) |>
     t() |>
     solve(a = H) |>
     solve(a = Sigma)

     all.equal(c(comp_res), dF)

     dput(comp_res)
     */
    constexpr arma::uword n{5};

    arma::mat Sigma{6.53840376244045, 3.25364028156331, -3.93769885961481, 1.47229034065976, -1.58853096462909, 3.25364028156331, 15.8224564695579, -5.45900271436491, -0.418287500907006, -9.13710658458838, -3.93769885961481, -5.45900271436491, 11.8710928151963, 3.83994392481692, 6.31928920340362, 1.47229034065976, -0.418287500907006, 3.83994392481692, 7.73275702834141, 1.91859396027026, -1.58853096462909, -9.13710658458838, 6.31928920340362, 1.91859396027026, 12.8671054809489},
                  K{0.891088917219998, 0.547249785327857, 0.723330076893113, -1.87882939304271, -0.0630114787615107, 0.547249785327857, 1.02712303062224, 1.21919547775656, -1.12024425578792, 0.475679108874113, 0.723330076893113, 1.21919547775656, 1.45625660244923, -1.4874206438287, 0.525705511955469, -1.87882939304271, -1.12024425578792, -1.4874206438287, 3.96308010657873, 0.157877099749777, -0.0630114787615107, 0.475679108874113, 0.525705511955469, 0.157877099749777, 0.387334393154351},
             dH_inv{484.444947170403, 16.4789307714358, 0.0883098397142331, 16.3264143375452, 0.634059475655878, 16.4789307714358, 152115.657221066, 0.00493508350902881, 17.166830926361, 0.000365883567707474, 0.088309839714233, 0.00493508350902882, 90.1916462857063, 0.395603576291986, 4.00766354829859, 16.3264143375452, 17.166830926361, 0.395603576291987, 7.92756177857831, 0.112148676957492, 0.634059475655869, 0.000365883567707474, 4.00766354829859, 0.112148676957492, 10318.546358946},
        true_derivs{18655.1616963039, -33367.5623913086, 23174.7753743168, -38005.4937320917, 6381.80831452317, -33367.5623913086, 61645.621069475, -43355.6381075457, 69729.7302571997, -10074.655975296, 23174.7753743168, -43355.6381075457, 31265.7694272985, -48303.5189011489, 4981.5941609135, -38005.4937320916, 69729.7302571997, -48303.5189011489, 79580.7455091153, -13402.319153896, 6381.80831452315, -10074.655975296, 4981.5941609135, -13402.3191538959, 7413.53805601633};

    Sigma.reshape(n, n);
    K.reshape(n, n);
    dH_inv.reshape(n, n);
    true_derivs.reshape(n, n);

    arma::mat H{K + arma::inv_sympd(Sigma)};

    auto res = dcond_vcov(H, dH_inv, Sigma);
    expect_true(res.n_rows == n);
    expect_true(res.n_cols == n);

    auto const eps = std::sqrt(std::numeric_limits<double>::epsilon());
    for(arma::uword j = 0; j < n; ++j)
      for(arma::uword i = 0; i < n; ++i)
        expect_true(std::abs(res(i, j) - true_derivs(i, j)) <
          eps * std::abs(true_derivs(i, j)));
  }
}
