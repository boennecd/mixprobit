/* call vmmin for BFGS */

#include "mix-optim.h"

/* from R_ext/Applic.h */
typedef double optimfn(int, double *, void *);
typedef void optimgr(int, double *, double *, void *);

extern "C" {
  void vmmin(int n, double *b, double *Fmin,
             optimfn fn, optimgr gr, int maxit, int trace,
             int *mask, double abstol, double reltol, int nREPORT,
             void *ex, int *fncount, int *grcount, int *fail);
}

namespace optimizers {
optim_res bfgs(arma::vec const &start_val, objective obj,
               objective_gradient gr, void *data, int const max_it,
               int const trace, double const abstol, double const reltol,
               int const n_report){
  int const npar = start_val.n_elem;

  optim_res out;
  double &val = out.val;
  arma::vec &par = out.par;
  par = start_val;
  int &fncount = out.fncount,
      &grcount = out.grcount,
      &fail    = out.fail;

  arma::ivec mask(npar);
  for(int i = 0; i < npar; ++i)
    mask[i] = 1;

  vmmin(npar, par.memptr(), &val, obj, gr, max_it, trace, mask.memptr(),
        abstol, reltol, n_report, data, &fncount, &grcount, &fail);

  return out;
}
} // namespace optimizers
