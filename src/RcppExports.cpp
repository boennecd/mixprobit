// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

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
Rcpp::List pmvnorm_cpp_restrict(arma::vec const& mean, arma::mat const& cov, int const maxpts, double const abseps, double const releps, bool const gradient, int const minvls);
RcppExport SEXP _mixprobit_pmvnorm_cpp_restrict(SEXP meanSEXP, SEXP covSEXP, SEXP maxptsSEXP, SEXP absepsSEXP, SEXP relepsSEXP, SEXP gradientSEXP, SEXP minvlsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec const& >::type mean(meanSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type cov(covSEXP);
    Rcpp::traits::input_parameter< int const >::type maxpts(maxptsSEXP);
    Rcpp::traits::input_parameter< double const >::type abseps(absepsSEXP);
    Rcpp::traits::input_parameter< double const >::type releps(relepsSEXP);
    Rcpp::traits::input_parameter< bool const >::type gradient(gradientSEXP);
    Rcpp::traits::input_parameter< int const >::type minvls(minvlsSEXP);
    rcpp_result_gen = Rcpp::wrap(pmvnorm_cpp_restrict(mean, cov, maxpts, abseps, releps, gradient, minvls));
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
// aprx_mult_mix
Rcpp::NumericVector aprx_mult_mix(unsigned const n_alt, arma::vec const& eta, arma::mat const& Z, arma::mat const& Sigma, int const maxpts, double const abseps, double const releps, int const key, bool const is_adaptive);
RcppExport SEXP _mixprobit_aprx_mult_mix(SEXP n_altSEXP, SEXP etaSEXP, SEXP ZSEXP, SEXP SigmaSEXP, SEXP maxptsSEXP, SEXP absepsSEXP, SEXP relepsSEXP, SEXP keySEXP, SEXP is_adaptiveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned const >::type n_alt(n_altSEXP);
    Rcpp::traits::input_parameter< arma::vec const& >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type Sigma(SigmaSEXP);
    Rcpp::traits::input_parameter< int const >::type maxpts(maxptsSEXP);
    Rcpp::traits::input_parameter< double const >::type abseps(absepsSEXP);
    Rcpp::traits::input_parameter< double const >::type releps(relepsSEXP);
    Rcpp::traits::input_parameter< int const >::type key(keySEXP);
    Rcpp::traits::input_parameter< bool const >::type is_adaptive(is_adaptiveSEXP);
    rcpp_result_gen = Rcpp::wrap(aprx_mult_mix(n_alt, eta, Z, Sigma, maxpts, abseps, releps, key, is_adaptive));
    return rcpp_result_gen;
END_RCPP
}
// aprx_jac_binary_mix
Rcpp::NumericVector aprx_jac_binary_mix(arma::ivec const& y, arma::vec const& eta, arma::mat const& X, arma::mat const& Z, arma::mat const& Sigma, int const maxpts, double const abseps, double const releps, int const key, bool const is_adaptive);
RcppExport SEXP _mixprobit_aprx_jac_binary_mix(SEXP ySEXP, SEXP etaSEXP, SEXP XSEXP, SEXP ZSEXP, SEXP SigmaSEXP, SEXP maxptsSEXP, SEXP absepsSEXP, SEXP relepsSEXP, SEXP keySEXP, SEXP is_adaptiveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::ivec const& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec const& >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type Sigma(SigmaSEXP);
    Rcpp::traits::input_parameter< int const >::type maxpts(maxptsSEXP);
    Rcpp::traits::input_parameter< double const >::type abseps(absepsSEXP);
    Rcpp::traits::input_parameter< double const >::type releps(relepsSEXP);
    Rcpp::traits::input_parameter< int const >::type key(keySEXP);
    Rcpp::traits::input_parameter< bool const >::type is_adaptive(is_adaptiveSEXP);
    rcpp_result_gen = Rcpp::wrap(aprx_jac_binary_mix(y, eta, X, Z, Sigma, maxpts, abseps, releps, key, is_adaptive));
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
// aprx_mult_mix_cdf
Rcpp::NumericVector aprx_mult_mix_cdf(unsigned const n_alt, arma::vec const& eta, arma::mat const& Z, arma::mat const& Sigma, int const maxpts, double const abseps, double const releps);
RcppExport SEXP _mixprobit_aprx_mult_mix_cdf(SEXP n_altSEXP, SEXP etaSEXP, SEXP ZSEXP, SEXP SigmaSEXP, SEXP maxptsSEXP, SEXP absepsSEXP, SEXP relepsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned const >::type n_alt(n_altSEXP);
    Rcpp::traits::input_parameter< arma::vec const& >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type Sigma(SigmaSEXP);
    Rcpp::traits::input_parameter< int const >::type maxpts(maxptsSEXP);
    Rcpp::traits::input_parameter< double const >::type abseps(absepsSEXP);
    Rcpp::traits::input_parameter< double const >::type releps(relepsSEXP);
    rcpp_result_gen = Rcpp::wrap(aprx_mult_mix_cdf(n_alt, eta, Z, Sigma, maxpts, abseps, releps));
    return rcpp_result_gen;
END_RCPP
}
// aprx_binary_mix_cdf_get_ptr
SEXP aprx_binary_mix_cdf_get_ptr(Rcpp::List data, unsigned const n_threads, bool const gradient, unsigned const minvls, bool const do_reorder);
RcppExport SEXP _mixprobit_aprx_binary_mix_cdf_get_ptr(SEXP dataSEXP, SEXP n_threadsSEXP, SEXP gradientSEXP, SEXP minvlsSEXP, SEXP do_reorderSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type data(dataSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type n_threads(n_threadsSEXP);
    Rcpp::traits::input_parameter< bool const >::type gradient(gradientSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type minvls(minvlsSEXP);
    Rcpp::traits::input_parameter< bool const >::type do_reorder(do_reorderSEXP);
    rcpp_result_gen = Rcpp::wrap(aprx_binary_mix_cdf_get_ptr(data, n_threads, gradient, minvls, do_reorder));
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
// aprx_mult_mix_ghq
double aprx_mult_mix_ghq(unsigned const n_alt, arma::vec const& eta, arma::mat const& Z, arma::mat const& Sigma, unsigned const b, bool const is_adaptive);
RcppExport SEXP _mixprobit_aprx_mult_mix_ghq(SEXP n_altSEXP, SEXP etaSEXP, SEXP ZSEXP, SEXP SigmaSEXP, SEXP bSEXP, SEXP is_adaptiveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned const >::type n_alt(n_altSEXP);
    Rcpp::traits::input_parameter< arma::vec const& >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type Sigma(SigmaSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type b(bSEXP);
    Rcpp::traits::input_parameter< bool const >::type is_adaptive(is_adaptiveSEXP);
    rcpp_result_gen = Rcpp::wrap(aprx_mult_mix_ghq(n_alt, eta, Z, Sigma, b, is_adaptive));
    return rcpp_result_gen;
END_RCPP
}
// aprx_binary_mix_qmc
Rcpp::NumericVector aprx_binary_mix_qmc(arma::ivec const& y, arma::vec eta, arma::mat Z, arma::mat const& Sigma, unsigned const n_max, arma::ivec const& seeds, double const releps, bool const is_adaptive);
RcppExport SEXP _mixprobit_aprx_binary_mix_qmc(SEXP ySEXP, SEXP etaSEXP, SEXP ZSEXP, SEXP SigmaSEXP, SEXP n_maxSEXP, SEXP seedsSEXP, SEXP relepsSEXP, SEXP is_adaptiveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< arma::ivec const& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type Sigma(SigmaSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type n_max(n_maxSEXP);
    Rcpp::traits::input_parameter< arma::ivec const& >::type seeds(seedsSEXP);
    Rcpp::traits::input_parameter< double const >::type releps(relepsSEXP);
    Rcpp::traits::input_parameter< bool const >::type is_adaptive(is_adaptiveSEXP);
    rcpp_result_gen = Rcpp::wrap(aprx_binary_mix_qmc(y, eta, Z, Sigma, n_max, seeds, releps, is_adaptive));
    return rcpp_result_gen;
END_RCPP
}
// aprx_mult_mix_qmc
Rcpp::NumericVector aprx_mult_mix_qmc(unsigned const n_alt, arma::vec const& eta, arma::mat const& Z, arma::mat const& Sigma, unsigned const n_max, arma::ivec const& seeds, double const releps, bool const is_adaptive);
RcppExport SEXP _mixprobit_aprx_mult_mix_qmc(SEXP n_altSEXP, SEXP etaSEXP, SEXP ZSEXP, SEXP SigmaSEXP, SEXP n_maxSEXP, SEXP seedsSEXP, SEXP relepsSEXP, SEXP is_adaptiveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< unsigned const >::type n_alt(n_altSEXP);
    Rcpp::traits::input_parameter< arma::vec const& >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type Sigma(SigmaSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type n_max(n_maxSEXP);
    Rcpp::traits::input_parameter< arma::ivec const& >::type seeds(seedsSEXP);
    Rcpp::traits::input_parameter< double const >::type releps(relepsSEXP);
    Rcpp::traits::input_parameter< bool const >::type is_adaptive(is_adaptiveSEXP);
    rcpp_result_gen = Rcpp::wrap(aprx_mult_mix_qmc(n_alt, eta, Z, Sigma, n_max, seeds, releps, is_adaptive));
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
// aprx_mult_mix_brute
Rcpp::NumericVector aprx_mult_mix_brute(unsigned const n_alt, arma::vec const& eta, arma::mat const& Z, arma::mat const& Sigma, unsigned const n_sim, unsigned const n_threads, bool const is_is);
RcppExport SEXP _mixprobit_aprx_mult_mix_brute(SEXP n_altSEXP, SEXP etaSEXP, SEXP ZSEXP, SEXP SigmaSEXP, SEXP n_simSEXP, SEXP n_threadsSEXP, SEXP is_isSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned const >::type n_alt(n_altSEXP);
    Rcpp::traits::input_parameter< arma::vec const& >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type Sigma(SigmaSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type n_sim(n_simSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type n_threads(n_threadsSEXP);
    Rcpp::traits::input_parameter< bool const >::type is_is(is_isSEXP);
    rcpp_result_gen = Rcpp::wrap(aprx_mult_mix_brute(n_alt, eta, Z, Sigma, n_sim, n_threads, is_is));
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
// my_pmvnorm_cpp
Rcpp::NumericVector my_pmvnorm_cpp(arma::vec const& mean_in, arma::mat const& sigma_in, unsigned const nsim, double const eps);
RcppExport SEXP _mixprobit_my_pmvnorm_cpp(SEXP mean_inSEXP, SEXP sigma_inSEXP, SEXP nsimSEXP, SEXP epsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec const& >::type mean_in(mean_inSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type sigma_in(sigma_inSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type nsim(nsimSEXP);
    Rcpp::traits::input_parameter< double const >::type eps(epsSEXP);
    rcpp_result_gen = Rcpp::wrap(my_pmvnorm_cpp(mean_in, sigma_in, nsim, eps));
    return rcpp_result_gen;
END_RCPP
}
// get_sobol_obj
SEXP get_sobol_obj(int const dimen, int const scrambling, int const seed);
RcppExport SEXP _mixprobit_get_sobol_obj(SEXP dimenSEXP, SEXP scramblingSEXP, SEXP seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< int const >::type dimen(dimenSEXP);
    Rcpp::traits::input_parameter< int const >::type scrambling(scramblingSEXP);
    Rcpp::traits::input_parameter< int const >::type seed(seedSEXP);
    rcpp_result_gen = Rcpp::wrap(get_sobol_obj(dimen, scrambling, seed));
    return rcpp_result_gen;
END_RCPP
}
// eval_sobol
arma::mat eval_sobol(unsigned const n, SEXP ptr);
RcppExport SEXP _mixprobit_eval_sobol(SEXP nSEXP, SEXP ptrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< unsigned const >::type n(nSEXP);
    Rcpp::traits::input_parameter< SEXP >::type ptr(ptrSEXP);
    rcpp_result_gen = Rcpp::wrap(eval_sobol(n, ptr));
    return rcpp_result_gen;
END_RCPP
}
// multinomial_inner_integral
arma::mat multinomial_inner_integral(arma::mat const& Z, arma::vec const& eta, arma::mat const& Sigma, unsigned const n_nodes, bool const is_adaptive, unsigned const n_times, arma::vec const& u, unsigned const order);
RcppExport SEXP _mixprobit_multinomial_inner_integral(SEXP ZSEXP, SEXP etaSEXP, SEXP SigmaSEXP, SEXP n_nodesSEXP, SEXP is_adaptiveSEXP, SEXP n_timesSEXP, SEXP uSEXP, SEXP orderSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< arma::mat const& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< arma::vec const& >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type Sigma(SigmaSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type n_nodes(n_nodesSEXP);
    Rcpp::traits::input_parameter< bool const >::type is_adaptive(is_adaptiveSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type n_times(n_timesSEXP);
    Rcpp::traits::input_parameter< arma::vec const& >::type u(uSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type order(orderSEXP);
    rcpp_result_gen = Rcpp::wrap(multinomial_inner_integral(Z, eta, Sigma, n_nodes, is_adaptive, n_times, u, order));
    return rcpp_result_gen;
END_RCPP
}

RcppExport SEXP run_testthat_tests();

static const R_CallMethodDef CallEntries[] = {
    {"_mixprobit_pmvnorm_cpp", (DL_FUNC) &_mixprobit_pmvnorm_cpp, 7},
    {"_mixprobit_pmvnorm_cpp_restrict", (DL_FUNC) &_mixprobit_pmvnorm_cpp_restrict, 7},
    {"_mixprobit_aprx_binary_mix", (DL_FUNC) &_mixprobit_aprx_binary_mix, 9},
    {"_mixprobit_aprx_mult_mix", (DL_FUNC) &_mixprobit_aprx_mult_mix, 9},
    {"_mixprobit_aprx_jac_binary_mix", (DL_FUNC) &_mixprobit_aprx_jac_binary_mix, 10},
    {"_mixprobit_aprx_binary_mix_cdf", (DL_FUNC) &_mixprobit_aprx_binary_mix_cdf, 7},
    {"_mixprobit_aprx_mult_mix_cdf", (DL_FUNC) &_mixprobit_aprx_mult_mix_cdf, 7},
    {"_mixprobit_aprx_binary_mix_cdf_get_ptr", (DL_FUNC) &_mixprobit_aprx_binary_mix_cdf_get_ptr, 5},
    {"_mixprobit_aprx_binary_mix_cdf_eval", (DL_FUNC) &_mixprobit_aprx_binary_mix_cdf_eval, 6},
    {"_mixprobit_set_GH_rule_cached", (DL_FUNC) &_mixprobit_set_GH_rule_cached, 1},
    {"_mixprobit_aprx_binary_mix_ghq", (DL_FUNC) &_mixprobit_aprx_binary_mix_ghq, 6},
    {"_mixprobit_aprx_mult_mix_ghq", (DL_FUNC) &_mixprobit_aprx_mult_mix_ghq, 6},
    {"_mixprobit_aprx_binary_mix_qmc", (DL_FUNC) &_mixprobit_aprx_binary_mix_qmc, 8},
    {"_mixprobit_aprx_mult_mix_qmc", (DL_FUNC) &_mixprobit_aprx_mult_mix_qmc, 8},
    {"_mixprobit_aprx_binary_mix_brute", (DL_FUNC) &_mixprobit_aprx_binary_mix_brute, 7},
    {"_mixprobit_aprx_mult_mix_brute", (DL_FUNC) &_mixprobit_aprx_mult_mix_brute, 7},
    {"_mixprobit_for_rngnorm_wrapper_test", (DL_FUNC) &_mixprobit_for_rngnorm_wrapper_test, 2},
    {"_mixprobit_my_pmvnorm_cpp", (DL_FUNC) &_mixprobit_my_pmvnorm_cpp, 4},
    {"_mixprobit_get_sobol_obj", (DL_FUNC) &_mixprobit_get_sobol_obj, 3},
    {"_mixprobit_eval_sobol", (DL_FUNC) &_mixprobit_eval_sobol, 2},
    {"_mixprobit_multinomial_inner_integral", (DL_FUNC) &_mixprobit_multinomial_inner_integral, 8},
    {"run_testthat_tests", (DL_FUNC) &run_testthat_tests, 0},
    {NULL, NULL, 0}
};

RcppExport void R_init_mixprobit(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
