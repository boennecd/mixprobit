#ifndef SOBOL_H
#define SOBOL_H
#include "arma-wrap.h"
#include <memory>

class sobol_gen {
public:
  /* need to call NEXTSOBOL */
  int const dimen;

private:
  std::unique_ptr<double[]> quasi =
    std::unique_ptr<double[]>(new double[dimen]);
  int count = 0L,
         ll = 0L;
  std::unique_ptr<int[]> sv =
    std::unique_ptr<int   []>(new int   [dimen * 30L]);;

public:
  sobol_gen(int const dimen, int const scrambling, int const seed);
  sobol_gen(int const dimen): sobol_gen(dimen, 0L, 4711L) { }

  /* returns the next vector by reference */
  void operator()(arma::vec&);

  /* allocates a new vector and returns the next vector */
  arma::vec operator()() {
    arma::vec out(dimen);
    operator()(out);
    return out;
  }
};

#endif
