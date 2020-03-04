#include "utils.h"
arma::mat dchol(arma::mat const &X, bool const upper)
{
  using arma::uword;
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
