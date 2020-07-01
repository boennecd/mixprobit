#ifndef QMC_H
#define QMC_H

#include "ranrth-wrapper.h"
namespace qmc {
using integrand::base_integrand;

struct qmc_approx_output {
  double value, err;
  int intvls;
};

qmc_approx_output approx(base_integrand const&, bool const, size_t const,
                         arma::ivec const&, double const);

} // namespace qmc
#endif
