#include "utils.h"
using arma::uword;

arma::mat dchol(arma::mat const &X, bool const upper)
{
  uword const ndim = X.n_rows,
    nvech = (ndim * (ndim + 1L)) / 2L;
  arma::mat out(nvech, nvech, arma::fill::zeros);
  arma::mat const F  = arma::chol(X),
    Fi = arma::inv(arma::trimatu(F));

  auto func = [&]
  (uword const i, uword const j, uword const k, uword const l){
    double out(0);

    if(k != l){
      double x(0);
      out += F.at(j, i) * Fi.at(k, j) / 2.;
      x   += F.at(j, i) * Fi.at(l, j) / 2.;

      for(uword m = j + 1; m < ndim; m++){
        out += F.at(m, i) * Fi.at(k, m);
        x   += F.at(m, i) * Fi.at(l, m);
      }

      out *= Fi.at(l, j);
      x   *= Fi.at(k, j);
      out += x;

    } else {
      out += F.at(j, i) * Fi.at(k, j) / 2.;
      for(uword m = j + 1; m < ndim; m++)
        out += F.at(m, i) * Fi.at(k, m);

      out *= Fi.at(l, j);

    }

    return out;
  };

  if(upper){
    /* get index map from index in vech that maps to a lower triangular
    * matrix to one that maps to an upper triangular matrix.
    * TODO: very slow... */
    auto im = [&](uword const idx){
      uword co(0), ro(0);
      {
        uword dum(idx), remain(ndim);
        while(dum >= remain){
          ++co;
          dum -= remain--;
        }

        ro = co + dum;
      }

      uword ret(0);
      for(uword i = 0; i <= ro; i++)
        ret += i;

      return ret + co;
    };

    uword r(0);
    for(uword j = 0; j < ndim; j++)
      for(uword i = j; i < ndim; i++, r++){
        uword c(0);
        uword const rim = im(r);
        for(uword k = 0; c <= r and k < ndim; k++)
          for(uword l = k; l < ndim; l++, c++)
            out.at(rim, im(c)) = func(i, j, k, l);
      }

  } else {
    uword r(0);
    for(uword j = 0; j < ndim; j++)
      for(uword i = j; i < ndim; i++, r++){
        uword c(0);
        for(uword k = 0; c <= r and k < ndim; k++)
          for(uword l = k; l < ndim; l++, c++)
            out.at(r, c) = func(i, j, k, l);
      }
  }

  return out;
}

arma::mat dcond_vcov
  (arma::mat const &H, arma::mat const &dH_inv, arma::mat const &Sigma){
  arma::uword const n{H.n_cols};
  arma::mat res = dH_inv,
            dum(n, n);

  auto perform_solve =
    [](arma::mat &res, arma::mat const &lhs, arma::mat const &rhs){
      if(!arma::solve(res, lhs, rhs, arma::solve_opts::likely_sympd))
        throw std::runtime_error("dcond_vcov: solve failed");
    };

  // TODO: would be better to form decomposition of H and Sigma
  perform_solve(dum, H, res);
  perform_solve(res, Sigma, dum);
  arma::inplace_trans(res);
  perform_solve(dum, H, res);
  perform_solve(res, Sigma, dum);

  return res;
}
