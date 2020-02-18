#include "arma-wrap.h"

double dmvnrm(arma::vec x, arma::mat const &cov_chol_inv,
              bool const logd = true);
