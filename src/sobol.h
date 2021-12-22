#ifndef SOBOL_H
#define SOBOL_H
#include "arma-wrap.h"
#include <memory>

class sobol_gen {
public:
  /* need to call NEXTSOBOL */
  int const dimen;

private:
  std::unique_ptr<double[]> quasi{new double[dimen]};
  int count = 0L,
         ll = 0L;
  std::unique_ptr<int[]> sv{new int[dimen * 30L]};

public:
  sobol_gen(int const dimen, int const scrambling = 0, int const seed = 4711);

  /* sets the next vector to the array. No checks on the pointer */
  void operator()(double*);

  /* returns the next vector by reference */
  void operator()(arma::vec &out){
    if(out.n_elem != (size_t)dimen)
      throw std::invalid_argument("sobol_gen::operator(): invalid out");

    operator()(out.begin());
  }

  /* allocates a new vector and returns the next vector */
  arma::vec operator()() {
    arma::vec out(dimen);
    operator()(out);
    return out;
  }
};

#endif
