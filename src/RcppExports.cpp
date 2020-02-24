// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// pmvnorm_cpp
Rcpp::List pmvnorm_cpp(arma::vec const& lower, arma::vec const& upper, arma::vec const& mean, arma::mat const& cov, int const maxpts, double const abseps, double const releps);
RcppExport SEXP _mixprobit_pmvnorm_cpp(SEXP lowerSEXP, SEXP upperSEXP, SEXP meanSEXP, SEXP covSEXP, SEXP maxptsSEXP, SEXP absepsSEXP, SEXP relepsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec const& >::type lower(lowerSEXP);
    Rcpp::traits::input_parameter< arma::vec const& >::type upper(upperSEXP);
    Rcpp::traits::input_parameter< arma::vec const& >::type mean(meanSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type cov(covSEXP);
    Rcpp::traits::input_parameter< int const >::type maxpts(maxptsSEXP);
    Rcpp::traits::input_parameter< double const >::type abseps(absepsSEXP);
    Rcpp::traits::input_parameter< double const >::type releps(relepsSEXP);
    rcpp_result_gen = Rcpp::wrap(pmvnorm_cpp(lower, upper, mean, cov, maxpts, abseps, releps));
    return rcpp_result_gen;
END_RCPP
}
// pmvnorm_cpp_restrict
Rcpp::List pmvnorm_cpp_restrict(arma::vec const& mean, arma::mat const& cov, int const maxpts, double const abseps, double const releps, bool const gradient);
RcppExport SEXP _mixprobit_pmvnorm_cpp_restrict(SEXP meanSEXP, SEXP covSEXP, SEXP maxptsSEXP, SEXP absepsSEXP, SEXP relepsSEXP, SEXP gradientSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec const& >::type mean(meanSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type cov(covSEXP);
    Rcpp::traits::input_parameter< int const >::type maxpts(maxptsSEXP);
    Rcpp::traits::input_parameter< double const >::type abseps(absepsSEXP);
    Rcpp::traits::input_parameter< double const >::type releps(relepsSEXP);
    Rcpp::traits::input_parameter< bool const >::type gradient(gradientSEXP);
    rcpp_result_gen = Rcpp::wrap(pmvnorm_cpp_restrict(mean, cov, maxpts, abseps, releps, gradient));
    return rcpp_result_gen;
END_RCPP
}
// aprx_binary_mix
Rcpp::NumericVector aprx_binary_mix(arma::ivec const& y, arma::vec const& eta, arma::mat const& Z, arma::mat const& Sigma, int const maxpts, double const abseps, double const releps, int const key, bool const is_adaptive);
RcppExport SEXP _mixprobit_aprx_binary_mix(SEXP ySEXP, SEXP etaSEXP, SEXP ZSEXP, SEXP SigmaSEXP, SEXP maxptsSEXP, SEXP absepsSEXP, SEXP relepsSEXP, SEXP keySEXP, SEXP is_adaptiveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::ivec const& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec const& >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type Sigma(SigmaSEXP);
    Rcpp::traits::input_parameter< int const >::type maxpts(maxptsSEXP);
    Rcpp::traits::input_parameter< double const >::type abseps(absepsSEXP);
    Rcpp::traits::input_parameter< double const >::type releps(relepsSEXP);
    Rcpp::traits::input_parameter< int const >::type key(keySEXP);
    Rcpp::traits::input_parameter< bool const >::type is_adaptive(is_adaptiveSEXP);
    rcpp_result_gen = Rcpp::wrap(aprx_binary_mix(y, eta, Z, Sigma, maxpts, abseps, releps, key, is_adaptive));
    return rcpp_result_gen;
END_RCPP
}
// aprx_binary_mix_cdf
Rcpp::NumericVector aprx_binary_mix_cdf(arma::ivec const& y, arma::vec eta, arma::mat Z, arma::mat const& Sigma, int const maxpts, double const abseps, double const releps);
RcppExport SEXP _mixprobit_aprx_binary_mix_cdf(SEXP ySEXP, SEXP etaSEXP, SEXP ZSEXP, SEXP SigmaSEXP, SEXP maxptsSEXP, SEXP absepsSEXP, SEXP relepsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::ivec const& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type Sigma(SigmaSEXP);
    Rcpp::traits::input_parameter< int const >::type maxpts(maxptsSEXP);
    Rcpp::traits::input_parameter< double const >::type abseps(absepsSEXP);
    Rcpp::traits::input_parameter< double const >::type releps(relepsSEXP);
    rcpp_result_gen = Rcpp::wrap(aprx_binary_mix_cdf(y, eta, Z, Sigma, maxpts, abseps, releps));
    return rcpp_result_gen;
END_RCPP
}
// aprx_binary_mix_cdf_get_ptr
SEXP aprx_binary_mix_cdf_get_ptr(Rcpp::List data, unsigned const n_threads, bool const gradient);
RcppExport SEXP _mixprobit_aprx_binary_mix_cdf_get_ptr(SEXP dataSEXP, SEXP n_threadsSEXP, SEXP gradientSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type data(dataSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type n_threads(n_threadsSEXP);
    Rcpp::traits::input_parameter< bool const >::type gradient(gradientSEXP);
    rcpp_result_gen = Rcpp::wrap(aprx_binary_mix_cdf_get_ptr(data, n_threads, gradient));
    return rcpp_result_gen;
END_RCPP
}
// aprx_binary_mix_cdf_eval
arma::vec aprx_binary_mix_cdf_eval(SEXP ptr, arma::vec const& beta, arma::vec const& log_sds, int const maxpts, double const abseps, double const releps);
RcppExport SEXP _mixprobit_aprx_binary_mix_cdf_eval(SEXP ptrSEXP, SEXP betaSEXP, SEXP log_sdsSEXP, SEXP maxptsSEXP, SEXP absepsSEXP, SEXP relepsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type ptr(ptrSEXP);
    Rcpp::traits::input_parameter< arma::vec const& >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::vec const& >::type log_sds(log_sdsSEXP);
    Rcpp::traits::input_parameter< int const >::type maxpts(maxptsSEXP);
    Rcpp::traits::input_parameter< double const >::type abseps(absepsSEXP);
    Rcpp::traits::input_parameter< double const >::type releps(relepsSEXP);
    rcpp_result_gen = Rcpp::wrap(aprx_binary_mix_cdf_eval(ptr, beta, log_sds, maxpts, abseps, releps));
    return rcpp_result_gen;
END_RCPP
}
// set_GH_rule_cached
Rcpp::List set_GH_rule_cached(unsigned const b);
RcppExport SEXP _mixprobit_set_GH_rule_cached(SEXP bSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned const >::type b(bSEXP);
    rcpp_result_gen = Rcpp::wrap(set_GH_rule_cached(b));
    return rcpp_result_gen;
END_RCPP
}
// aprx_binary_mix_ghq
double aprx_binary_mix_ghq(arma::ivec const& y, arma::vec eta, arma::mat Z, arma::mat const& Sigma, unsigned const b, bool const is_adaptive);
RcppExport SEXP _mixprobit_aprx_binary_mix_ghq(SEXP ySEXP, SEXP etaSEXP, SEXP ZSEXP, SEXP SigmaSEXP, SEXP bSEXP, SEXP is_adaptiveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::ivec const& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type Sigma(SigmaSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type b(bSEXP);
    Rcpp::traits::input_parameter< bool const >::type is_adaptive(is_adaptiveSEXP);
    rcpp_result_gen = Rcpp::wrap(aprx_binary_mix_ghq(y, eta, Z, Sigma, b, is_adaptive));
    return rcpp_result_gen;
END_RCPP
}
// aprx_binary_mix_brute
Rcpp::NumericVector aprx_binary_mix_brute(arma::ivec const& y, arma::vec eta, arma::mat Z, arma::mat const& Sigma, unsigned const n_sim, unsigned const n_threads, bool const is_is);
RcppExport SEXP _mixprobit_aprx_binary_mix_brute(SEXP ySEXP, SEXP etaSEXP, SEXP ZSEXP, SEXP SigmaSEXP, SEXP n_simSEXP, SEXP n_threadsSEXP, SEXP is_isSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::ivec const& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type Sigma(SigmaSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type n_sim(n_simSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type n_threads(n_threadsSEXP);
    Rcpp::traits::input_parameter< bool const >::type is_is(is_isSEXP);
    rcpp_result_gen = Rcpp::wrap(aprx_binary_mix_brute(y, eta, Z, Sigma, n_sim, n_threads, is_is));
    return rcpp_result_gen;
END_RCPP
}
// for_rngnorm_wrapper_test
Rcpp::NumericVector for_rngnorm_wrapper_test(unsigned const n, unsigned const n_threads);
RcppExport SEXP _mixprobit_for_rngnorm_wrapper_test(SEXP nSEXP, SEXP n_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned const >::type n(nSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type n_threads(n_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(for_rngnorm_wrapper_test(n, n_threads));
    return rcpp_result_gen;
END_RCPP
}

RcppExport SEXP run_testthat_tests();

static const R_CallMethodDef CallEntries[] = {
    {"_mixprobit_pmvnorm_cpp", (DL_FUNC) &_mixprobit_pmvnorm_cpp, 7},
    {"_mixprobit_pmvnorm_cpp_restrict", (DL_FUNC) &_mixprobit_pmvnorm_cpp_restrict, 6},
    {"_mixprobit_aprx_binary_mix", (DL_FUNC) &_mixprobit_aprx_binary_mix, 9},
    {"_mixprobit_aprx_binary_mix_cdf", (DL_FUNC) &_mixprobit_aprx_binary_mix_cdf, 7},
    {"_mixprobit_aprx_binary_mix_cdf_get_ptr", (DL_FUNC) &_mixprobit_aprx_binary_mix_cdf_get_ptr, 3},
    {"_mixprobit_aprx_binary_mix_cdf_eval", (DL_FUNC) &_mixprobit_aprx_binary_mix_cdf_eval, 6},
    {"_mixprobit_set_GH_rule_cached", (DL_FUNC) &_mixprobit_set_GH_rule_cached, 1},
    {"_mixprobit_aprx_binary_mix_ghq", (DL_FUNC) &_mixprobit_aprx_binary_mix_ghq, 6},
    {"_mixprobit_aprx_binary_mix_brute", (DL_FUNC) &_mixprobit_aprx_binary_mix_brute, 7},
    {"_mixprobit_for_rngnorm_wrapper_test", (DL_FUNC) &_mixprobit_for_rngnorm_wrapper_test, 2},
    {"run_testthat_tests", (DL_FUNC) &run_testthat_tests, 0},
    {NULL, NULL, 0}
};

RcppExport void R_init_mixprobit(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
