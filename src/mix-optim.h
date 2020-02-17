#include "arma-wrap.h"

namespace optimizers {
struct optim_res {
  double val;
  arma::vec par;
  int fncount, grcount, fail;
};

typedef double objective(int, double*, void*);
typedef void   objective_gradient(int, double *, double *, void *);

optim_res bfgs(arma::vec const &start_val, objective obj,
               objective_gradient gr, void *data, int const max_it,
               int const trace, double const abstol, double const reltol,
               int const n_report = 10L);
} // namespace optimizers
