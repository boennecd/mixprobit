#ifndef INTEGRAND_H
#define INTEGRAND_H

namespace integrand {
/* base class to use for the integral approximations with ranrth */
class base_integrand {
public:
  /* returns the integrand value at a given point */
  virtual double operator()
  (double const*, bool const ret_log = false) const = 0;

  /* returns the dimension of the integral */
  virtual std::size_t get_n_par() const = 0;

  ~base_integrand() = default;
};
} // namespace integrand
#endif
