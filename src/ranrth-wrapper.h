#ifndef RANRTH_H
#define RANRTH_H
#include <memory>

namespace ranrth_aprx {
/* base class to use for the integral approximations with ranrth */
class integrand {
public:
  /* returns the integrand value at a given point */
  virtual double operator()
  (double const*, bool const ret_log = false) const = 0;

  /* returns the dimension of the integral */
  virtual std::size_t get_n_par() const = 0;

  ~integrand() = default;
};

/* set the current integrand. Notice that there can only be one integrand
 * at a time. */
void set_integrand(std::unique_ptr<integrand>);

struct integral_arpx_res {
  double value, err;
  int inivls, inform;
};

integral_arpx_res integral_arpx(int const, int const, double const,
                                double const);
}

#endif
