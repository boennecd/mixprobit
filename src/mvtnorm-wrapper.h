#include "arma-wrap.h"

namespace pmvnorm {
arma::ivec get_infin(arma::vec const&, arma::vec const&);

struct cor_vec_res {
  arma::vec cor_vec, sds;
};

cor_vec_res get_cor_vec(const arma::mat&);

struct cdf_res {
  double error, value;
  int inform;
};

cdf_res cdf(arma::vec, arma::vec, arma::vec, arma::mat const&,
            int const maxpts = -1L, double const abseps = -1,
            double const releps = 1e-5);

cdf_res cdf(arma::vec const&, arma::vec const&, arma::ivec const&,
            arma::vec const&, arma::vec const&,
            int const maxpts = -1L, double const abseps = -1,
            double const releps = 1e-5);
}
