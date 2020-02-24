void dstev_call(const char* jobz, const int* n, double* d, double* e,
                double* z, const int* ldz, double* work, int* info);

void dsyr_call(const char *uplo, const int *n, const double *alpha,
               const double *x, const int *incx, double *a, const int *lda);
