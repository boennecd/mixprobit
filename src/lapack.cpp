#include "lapack.h"
#include <Rconfig.h>
#include <R_ext/BLAS.h>
#ifndef FCLEN
#define FCLEN
#endif
#ifndef FCONE
#define FCONE
#endif
#include <R_ext/Lapack.h>

void dstev_call(const char* jobz, const int* n, double* d, double* e,
                double* z, const int* ldz, double* work, int* info){
  F77_NAME(dstev)(
      jobz, n, d, e,
      z, ldz, work, info FCONE);
}
