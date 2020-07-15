#ifndef QNORM_H
#define QNORM_H

#ifdef __cplusplus
extern "C" {
#endif
double qnorm_w(double const p, double const mu, double const sigma,
               int const lower_tail, int const log_p);
#ifdef __cplusplus
}
#endif

#endif
