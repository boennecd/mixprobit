#include "sobol.h"
#include "qmc.h"
#include <Rmath.h>
#include "arma-wrap.h"
#include "welfords.h"
#include <array>

namespace qmc {
qmc_approx_output approx(
    base_integrand const &func, bool const is_adaptive, size_t const n_max,
    arma::ivec const &seeds, double const releps){
  /* checks */
  if(seeds.size() < 1L)
    throw std::invalid_argument("qmc::approx(): invalid seeds");
  for(auto i : seeds)
    if(i < 0L)
      throw std::invalid_argument("qmc::approx(): invalid seeds");

  using namespace integrand;
  if(is_adaptive){
    mvn<base_integrand> func_w_mvn(func);
    adaptive<mvn<base_integrand> > new_func(func_w_mvn, true);

    return approx(new_func, false, n_max, seeds, releps);
  }

  constexpr size_t n_min = 100L;
  size_t const n_par = func.get_n_par(),
             n_seeds = seeds.n_elem,
                 inc = n_seeds * n_min;
  arma::mat par(n_par, n_min);

  /* get object to store quasi random number generators */
  std::vector<sobol_gen> generators;
  generators.reserve(n_seeds);
  for(auto s : seeds)
    generators.emplace_back(n_par, 1L, s);

  /* get vector to store estimates from each quasi-random number sequence */
  std::vector<welfords> estimates;
  estimates.reserve(n_seeds);
  for(size_t i = 0; i < n_seeds; ++i)
    estimates.emplace_back(false);

  auto get_mean_n_sd = [&](){
    std::array<double, 2L> out;
    welfords comb;
    for(auto &w : estimates)
      comb += w.mean();

    out[0L] = comb.mean();
    out[1L] = sqrt(comb.var() / (static_cast<double>(n_seeds) - 1L));
    return out;
  };

  size_t i;
  constexpr double const alpha_quant = 2.5758293035489; /* qnorm(.995) */

  for(i = 0; i < n_max; i += inc){
    /* refine the estimates */
    for(size_t k = 0; k < n_seeds; ++k){
      auto &my_gen = generators[k];
      auto &my_est = estimates[k];

      for(size_t j = 0; j < n_min; ++j)
        my_gen(par.colptr(j));
      par.for_each([](arma::mat::elem_type &val){
        val = R::qnorm5(val, 0, 1, 1L, 0L);
      });

      for(size_t j = 0; j < n_min; ++j){
        double const integrand = func(par.colptr(j), false);
        if(!std::isfinite(integrand))
          throw std::runtime_error("qmc::approx(): non-finite integrand");
        my_est += integrand;
      }
    }

    /* check for convergence */
    if(n_seeds > 5L){
      auto mean_n_sd = get_mean_n_sd();

      if(mean_n_sd[1L] / mean_n_sd[0L] * alpha_quant < releps)
        break;
    }
  }

  auto const mean_n_sd = get_mean_n_sd();
  return { mean_n_sd[0L], mean_n_sd[1L], static_cast<int>(i) };
}
} // namespace qmc
