#include "arma-wrap.h"

struct my_pmvnorm_output {
  double est, se;
  unsigned nsim;
};

my_pmvnorm_output my_pmvnorm
  (arma::vec const&, arma::mat const&, unsigned const, double const);
