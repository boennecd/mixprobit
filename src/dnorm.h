/*
 *  Mathlib : A C Library of Special Functions
 *  Copyright (C) 1998 Ross Ihaka
 *  Copyright (C) 2000-2014 The R Core Team
 *  Copyright (C) 2003	    The R Foundation
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  https://www.R-project.org/Licenses/
 *
 *  SYNOPSIS
 *
 *	double dnorm4(double x, double mu, double sigma, int give_log)
 *	      {dnorm (..) is synonymous and preferred inside R}
 *
 *  DESCRIPTION
 *
 *	Compute the density of the normal distribution.
 */

#ifndef DNORM_H
#define DNORM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>

inline double dnorm_std(double x, int const give_log)
{
    if(isinf(x))
      return give_log ? -INFINITY : 0.;
    x = fabs(x);
    static double const too_large = 2 * sqrt(DBL_MAX);
    if (x >= too_large)
      return give_log ? -INFINITY : 0.;

    if (give_log)
    	return -(M_LN_SQRT_2PI + 0.5 * x * x);
    //  M_1_SQRT_2PI = 1 / sqrt(2 * pi)
    // more accurate, less fast :
    if (x < 5)
      return M_1_SQRT_2PI * exp(-0.5 * x * x);

    /*
     * x*x  may lose upto about two digits accuracy for "large" x
     * Morten Welinder's proposal for PR#15620
     * https://bugs.r-project.org/bugzilla/show_bug.cgi?id=15620

     * -- 1 --  No hoop jumping when we underflow to zero anyway:

     *  -x^2/2 <         log(2)*.Machine$double.min.exp  <==>
     *     x   > sqrt(-2*log(2)*.Machine$double.min.exp) =IEEE= 37.64031
     * but "thanks" to denormalized numbers, underflow happens a bit later,
     *  effective.D.MIN.EXP <- with(.Machine, double.min.exp + double.ulp.digits)
     * for IEEE, DBL_MIN_EXP is -1022 but "effective" is -1074
     * ==> boundary = sqrt(-2*log(2)*(.Machine$double.min.exp + .Machine$double.ulp.digits))
     *              =IEEE=  38.58601
     * [on one x86_64 platform, effective boundary a bit lower: 38.56804]
     */
    if (x > sqrt(-2*M_LN2*(DBL_MIN_EXP + 1-DBL_MANT_DIG))) return 0.;

    /* Now, to get full accurary, split x into two parts,
     *  x = x1+x2, such that |x2| <= 2^-16.
     * Assuming that we are using IEEE doubles, that means that
     * x1*x1 is error free for x<1024 (but we have x < 38.6 anyway).

     * If we do not have IEEE this is still an improvement over the naive formula.
     */
    double x1 = //  R_forceint(x * 65536) / 65536 =
	    ldexp( nearbyint(ldexp(x, 16)), -16);
    double x2 = x - x1;
    return M_1_SQRT_2PI *
      (exp(-0.5 * x1 * x1) * exp( (-0.5 * x2 - x1) * x2 ) );
}

inline double dnorm_w(double x, double const mu, double const sigma,
                    int const give_log)
{
    if(isnan(x) || isnan(mu) || isnan(sigma))
	    return x + mu + sigma;
    else if(sigma < 0)
      return NAN;
    else if(isinf(sigma))
      return give_log ? -INFINITY : 0.;
    else if(isinf(x) && mu == x)
      return NAN;
    else if(sigma == 0)
      return (x == mu) ? INFINITY : (give_log ? -INFINITY : 0.);

    double const out = dnorm_std((x - mu) / sigma, give_log);
    return give_log ? out - log(sigma) : out / sigma;
}

#ifdef __cplusplus
}
#endif
#endif
