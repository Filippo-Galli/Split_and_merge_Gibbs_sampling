// [[Rcpp::depends(RcppGSL)]]
#include <RcppGSL.h>
#include <iostream>
#include <gsl/gsl_sf_hyperg.h>
#include <cmath>

using namespace Rcpp;

// [[Rcpp::export]]
double norm_const2(const double d, const double c, const double m, const bool log_scale = false)
{

  double z = (m - 1) / m;
  double alpha = d + c;
  double beta = 1;
  double gamma = d + 2;
  gsl_sf_result out;

  gsl_set_error_handler_off();

  int stat = gsl_sf_hyperg_2F1_e(alpha, beta, gamma, z, &out);

  // Rcpp::Rcout<<"Valore della hyper "<<out.val<<"\n";

  if (stat != GSL_SUCCESS)
  {
    // Rcpp::Rcout<<"Sono qui\n";
    return R_NaN;
  }

  if (log_scale)
  {
    return std::log(d + 1) + (d + c) * std::log(m) - std::log(out.val);
    Rcpp::Rcout << "Sono in log_scale\n";
  }
  else
  {
    return std::exp(std::log(d + 1) + (d + c) * std::log(m) - std::log(out.val));
    Rcpp::Rcout << "Non sono in log_scale\n";
  }
}

// [[Rcpp::export]]
double hyperg2(double a, double b, double c, double x)
{
  gsl_sf_result result;
  gsl_set_error_handler_off();
  int stat = gsl_sf_hyperg_2F1_e(a, b, c, x, &result);
  if (stat != GSL_SUCCESS)
  {
    Rcpp::Rcout << "hypergeometric non converging\n";
    return R_NaN;
  }
  else
    return result.val;
}

// u is the argument of the density
// c and d the two real parameter of the hyper distr
// m also is a parameter but it is "fixed" by the data (m  greather than 2)
// [[Rcpp::export]]
Rcpp::NumericVector dhyper_raf2(const Rcpp::NumericVector u, const double d, const double c,
                                const double m, const bool log_scale = false)
{

  int n = u.length();
  Rcpp::NumericVector out(n);

  double lK = norm_const2(d, c, m, true);

  for (int i = 0; i < n; i++)
  {
    out[i] = lK + d * log(u[i]) - (d + c) * log(1 + u[i] * (m - 1));
  }
  if (log_scale)
  {
    return (out);
  }
  else
  {
    return (Rcpp::exp(out));
  }
}

/// Newtown method
// [[Rcpp::export]]
double newton_hyper2(const double d, const double c, const double m, const double Omega, const double u0 = 0.5)
{

  double hu = 1;
  double u_current = u0;
  double x;
  double dens;
  Rcpp::NumericVector app(1);
  int contatore = 0;
  while (std::abs(hu) > 0.00001)
  {
    x = u_current * (m - 1) / (1 + u_current * (m - 1));
    Rcpp::Rcout << "u_current=" << u_current << " x=" << x << "\n";
    app[0] = u_current;
    dens = dhyper_raf2(app, d, c, m, true)[0];
    hu = -u_current / (d + 1) * hyperg2(1, d + c, d + 2, x) + exp(log(Omega) - dens);
    Rcpp::Rcout << "hu=" << hu << " Omega/dens" << exp(log(Omega) - dens) << "\n";
    u_current += hu;
    if (u_current < 0)
    {
      u_current = 0.01;
    }
    if (u_current > 1)
    {
      u_current = 0.99;
    }
    contatore += 1;
    if (contatore > 100)
    {
      return R_NaN;
    }
  }

  return (u_current);
}

// [[Rcpp::export]]
double lF_conK2(const double u, const double d, const double c, const double m, const double lK)
{
  if (u == 0)
  {
    return -INFINITY;
  }
  if (u == 1)
  {
    return 0;
  }
  double x = u * (m - 1) / (1 + u * (m - 1));
  double app;
  app = hyperg2(1, d + c, d + 2, x);
  // Rcpp::Rcout<<" in lF_conK app="<<app<<"\n";
  double out = lK - log(d + 1) + (d + 1) * log(u) - (d + c) * log(1 + u * (m - 1)) + log(app);
  return out;
}

/// Newtown method
// [[Rcpp::export]]
double
bisec_hyper2(const double d, const double c, const double m, const double Omega)
{

  double centro = 0.5;
  double lK = norm_const2(d, c, m, true);
  double app = lF_conK2(centro, d, c, m, lK) - log(Omega);
  // Rcpp::Rcout<<"app="<<app<<"\n";

  double su;
  double giu;
  int counter = 1;
  int max_count = 150;

  if (app < 0)
  {
    giu = 0.5;
    su = 1;
  }
  else
  {
    giu = 0;
    su = 0.5;
  }

  while (((su - giu) > 0.000000001) & (counter < max_count))
  {
    centro = (su + giu) / 2;
    app = lF_conK2(centro, d, c, m, lK) - log(Omega);
    // Rcpp::Rcout<<"app="<<app<<"\n";

    if (app < 0)
    {
      giu = centro;
      su = su;
    }
    else
    {
      giu = giu;
      su = centro;
    }
    counter = counter + 1;
    //  Rcpp::Rcout<<"counter = "<< counter<<"\n";
  }

  if (counter == max_count)
  {
    Rcpp::Rcout << "Warning the bisection algorithm reached max number of iterations " << counter << "\n";
  }

  return (centro);
}

// n is an integer that represent the sample size
// c and d the two real parameter of the hyper distr
// m also is a parameter but it is "fixed" by the data (m>=2)
// u0 is the initial guess
// [[Rcpp::export]]
Rcpp::NumericVector rhyper_raf2(const int n, const double d, const double c, const double m)
{

  Rcpp::NumericVector out(n);
  double Omega;
  // the output of a gsl special function
  for (int i = 0; i < n; i++)
  {
    Omega = R::runif(0, 1);
    out[i] = bisec_hyper2(d, c, m, Omega);
  }

  return (out);
}

// n is an integer that represent the sample size
// c and d the two real parameter of the hyper distr
// m also is a parameter but it is "fixed" by the data (m>=2)
// [[Rcpp::export]]
Rcpp::NumericVector rhyper_sig2(const int n, const double d, const double c, const double m)
{

  Rcpp::NumericVector out(n);
  double Omega;
  // the output of a gsl special function
  for (int i = 0; i < n; i++)
  {
    Omega = R::runif(0, 1);
    out[i] = bisec_hyper2(d, c, m, Omega);
  }

  return (-1 / Rcpp::log(out));
}

// Prior over sigma
// x is the argument of the density
// c and d the two real parameter of the hyper distr
// m also is a parameter but it is "fixed" by the data (m  greather than 2)
// [[Rcpp::export]]
Rcpp::NumericVector dhyper_sig_raf2(const Rcpp::NumericVector x, const double d, const double c,
                                    const double m, const bool log_scale = false)
{

  int n = x.length();
  Rcpp::NumericVector out(n);

  double lK = norm_const2(d, c, m, true);

  for (int i = 0; i < n; i++)
  {
    out[i] = lK - (d + 1) / x[i] - (d + c) * log(1 + exp(-1 / x[i]) * (m - 1)) - 2 * log(x[i]);
  }
  if (log_scale)
  {
    return (out);
  }
  else
  {
    return (Rcpp::exp(out));
  }
}
