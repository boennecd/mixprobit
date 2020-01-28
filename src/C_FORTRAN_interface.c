/* $Id: C_FORTRAN_interface.c 313 2015-09-16 20:20:04Z mmaechler $
*
*  wrapper for calling R's random number generator from
*  the original FORTRAN code
*
*/

#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>

double F77_SUB(unifrnd)(void) {
  return unif_rand();
}
double F77_SUB(sqrtqchisqint)(int *n, double *p) {
    return(sqrt(qchisq(p[0], (double) n[0], 0, 0)));
}
double F77_SUB(phid)(double *x){
  return pnorm(*x, 0.0, 1.0, 1, 0);
}
double F77_SUB(studnt)(int *nu, double *x){
  return pt(x[0], (double) nu[0], 1, 0);
}

double F77_SUB(gamran)(double *a){
  return rgamma(*a, 1);
}
double F77_SUB(norran)(void){
  return norm_rand();
}
double F77_SUB(rnrnor)(void){
  return norm_rand();
}
double F77_SUB(betran)(double *a, double *b){
  return rbeta(*a, *b);
}

