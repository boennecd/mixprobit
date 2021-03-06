/*
  Copyright (c) 2014, Alexander W Blocker

  Permission is hereby granted, free of charge, to any person obtaining
  a copy of this software and associated documentation files (the
  "Software"), to deal in the Software without restriction, including
  without limitation the rights to use, copy, modify, merge, publish,
  distribute, sublicense, and/or sell copies of the Software, and to
  permit persons to whom the Software is furnished to do so, subject to
  the following conditions:

  The above copyright notice and this permission notice shall be
  included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/* The code is modified by Benjamin Christoffersen. */

#ifndef GAUSS_HERMITE_H
#define GAUSS_HERMITE_H
#include "ranrth-wrapper.h"
#include "arma-wrap.h"

namespace GaussHermite {
using integrand::base_integrand;

struct HermiteData {
  const arma::vec x, w,
                  w_log = arma::log(w);

  HermiteData(arma::vec const &x, arma::vec const &w): x(x), w(w) {
    if(x.n_elem != w.n_elem)
      throw std::invalid_argument("HermiteData::HermiteData(): invalid x");
  }
};

HermiteData gaussHermiteData(unsigned const);

/* throws an eror if cached is true when the first argument is large */
HermiteData const& gaussHermiteDataCached(unsigned const);

double approx(HermiteData const&, base_integrand const&,
              bool const is_adaptive = false);
} // namespace GaussHermite
#endif

