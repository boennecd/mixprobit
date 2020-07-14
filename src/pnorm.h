#ifndef PNORM_H
#define PNORM_H
#include <Rmath.h>

inline double pnorm_std(double const x, int lower, int is_log){
  double p, cp;
  p = x;
  Rf_pnorm_both(x, &p, &cp, lower ? 0 : 1, is_log);
  return lower ? p : cp;
}

#endif
