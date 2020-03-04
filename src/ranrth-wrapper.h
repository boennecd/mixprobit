#ifndef RANRTH_H
#define RANRTH_H
#include <memory>
#include "integrand.h"

namespace ranrth_aprx {
/* set the current integrand. Notice that there can only be one integrand
 * at a time. */
void set_integrand(std::unique_ptr<integrand::base_integrand>);

struct integral_arpx_res {
  double value, err;
  int inivls, inform;
};

integral_arpx_res integral_arpx(int const, int const, double const,
                                double const);

struct jac_arpx_res {
  arma::vec value, err;
  int inivls, inform;
};

jac_arpx_res jac_arpx(int const, int const, double const, double const);
}

#endif
