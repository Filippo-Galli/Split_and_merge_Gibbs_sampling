#ifndef HYPERG_HPP
#define HYPERG_HPP

#include <RcppGSL.h>
#include <gsl/gsl_sf_hyperg.h>

double norm_const2(const double d, const double c, const double m);
double hyperg2(double a, double b, double c, double x);
double lF_conK2(const double u, const double d, const double c, const double m, const double K);
double bisec_hyper2(const double d, const double c, const double m, const double Omega);
Rcpp::NumericVector dhyper_raf2(const Rcpp::NumericVector u, const double d, const double c,
                              const double m, const bool log_scale=false);
double newton_hyper2(const double d, const double c, const double m, const double Omega, const double u0=0.5);
Rcpp::NumericVector rhyper_raf2(const int n, const double d, const double c, const double m);
Rcpp::NumericVector rhyper_sig(const int n, const double d, const double c, const double m);
Rcpp::NumericVector dhyper_sig_raf2(const Rcpp::NumericVector x, const double d, const double c,
                                  const double m, const bool log_scale=false);

Rcpp::NumericVector rhig(const int n, const double c, const double d, const double m);
#endif // HYPERG_HPP