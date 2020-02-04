#ifdef __cplusplus
#include <vector>
namespace parallelrng {
/* set the seeds for up the number of element of the vector. Must be called
 * before calling any of the subsequent methods. */
void set_rng_seeds(std::vector<unsigned> const&);
/* set a given number of random seeds */
void set_rng_seeds(unsigned const);
}

extern "C"
{
#endif

double rngnorm_wrapper ();
double rngbeta_wrapper (double const, double const);
double rnggamma_wrapper(double const);
double rngunif_wrapper ();

#ifdef __cplusplus
}
#endif
