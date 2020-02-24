#ifndef WELFORDS_H
#define WELFORDS_H
#ifdef _OPENMP
#include <omp.h>
#endif

class welfords {
  double M = 0.,
      mea  = 0.;
  unsigned n = 0L;
public:
  double mean() const {
    return mea;
  }
  double var() const {
    return M / (double)n;
  }

  welfords& operator+=(double const x){
    double const old_diff = x - mea;
    mea += old_diff / (double)++n;
    M += (x - mea) * old_diff;

    return *this;
  }

  welfords& operator+=(welfords const &o) {
    unsigned const nm = n,
                   no = o.n;
    double const delta = o.mea - mea;
    n = nm + no;
    mea += delta * (double)no/(double)n;
    M += o.M + delta * delta * (double)(nm * no)/(double)n;

    return *this;
  }
};

#ifdef _OPENMP
#pragma omp declare reduction(welPlus: welfords: \
  omp_out += omp_in)
#endif

#endif
