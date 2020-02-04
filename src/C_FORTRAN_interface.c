/* $Id: C_FORTRAN_interface.c 313 2015-09-16 20:20:04Z mmaechler $
*
*  wrapper for calling R's random number generator from
*  the original FORTRAN code
*
*/

#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include "threat-safe-random.h"

double F77_SUB(unifrnd)(void) {
#ifdef USE_R_RNG
  return unif_rand();
#else
  return rngunif_wrapper();
#endif
}

double F77_SUB(sqrtqchisqint)(int const *n, double const *p) {
    return(sqrt(qchisq(p[0], (double) n[0], 0, 0)));
}

double F77_SUB(gamran)(double const *a){
#ifdef USE_R_RNG
  return rgamma(*a, 1);
#else
  return rnggamma_wrapper(*a);
#endif
}

double F77_SUB(norran)(void){
#ifdef USE_R_RNG
  return norm_rand();
#else
  return rngnorm_wrapper();
#endif
}

double F77_SUB(betran)(double const *a, double const *b){
#ifdef USE_R_RNG
  return rbeta(*a, *b);
#else
  return rngbeta_wrapper(*a, *b);
#endif
}

