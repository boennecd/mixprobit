#include "sobol.h"
#include "R_ext/RS.h"

extern "C" {
  void F77_NAME(initsobol)(
      int const * /* dimen */, double * /* quasi */, int * /* ll */,
      int * /* count */, int * /* SV */, int const * /* iflag/scambling */,
      int * /* iseed/seed */);

  void F77_NAME(nextsobol)(
      int const * /* dimen */, double * /* quasi */, int const * /* ll */,
      int * /* count */,  int const * /* SV */);
}

sobol_gen::sobol_gen
  (int const dimen, int const scrambling, int const seed):
  dimen(dimen) {
  if(dimen < 1L or dimen > 1111L)
    throw std::invalid_argument("sobol_gen::sobol_gen(): invalid dimen");
  else if(scrambling < 0L or scrambling > 3L)
    throw std::invalid_argument("sobol_gen::sobol_gen(): invalid scrambling");
  else if(seed < 0L /* or seed > 2147483647L */) /*  2^31 - 1L */
    throw std::invalid_argument("sobol_gen::sobol_gen(): invalid seed");

  int scramb = scrambling,
       iseed = seed;
  F77_CALL(initsobol)(
      &dimen, quasi.get(), &ll, &count, sv.get(), &scramb, &iseed);
}

void sobol_gen::operator()(double * out){
  F77_CALL(nextsobol)(
      &dimen, quasi.get(), &ll, &count, sv.get());

  double * o = out, *q = quasi.get();
  for(int i = 0; i < dimen; ++i, ++o, ++q){
    double val = *q;
    if(val < std::numeric_limits<double>::epsilon())
      val = std::numeric_limits<double>::epsilon();
    if(val >= 1.)
      val = 1. - std::numeric_limits<double>::epsilon();
    *o = val;
  }
}
