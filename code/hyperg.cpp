// [[Rcpp::depends(RcppGSL)]]
#include <RcppGSL.h>
#include <iostream>
#include <gsl/gsl_sf_hyperg.h>
#include <cmath>
#include "./hyperg.hpp"

using namespace Rcpp;

// [[Rcpp::export]]
double norm_const2(const double d, const double c, const double m){
  /**
   * @brief Compute the normalizing constant of the hypergeometric distribution
   * @param d First parameter of the hypergeometric distribution
   * @param c Second parameter of the hypergeometric distribution
   * @param m Third parameter of the hypergeometric distribution
   * @return Logarithm of the normalizing constant of the hypergeometric distribution
   */
  
  double z = (m - 1) / m;
  double alpha = d + c;
  double beta = 1;
  double gamma = d + 2;
  gsl_sf_result out;

  gsl_set_error_handler_off();

  int stat = gsl_sf_hyperg_2F1_e(alpha, beta, gamma, z, &out);

  if (stat != GSL_SUCCESS) {
    // Instead of throwing, return appropriate values for different cases
    if (stat == GSL_EMAXITER) {
        return -INFINITY;
    }
    if (stat == GSL_EOVRFLW) {
        return INFINITY;
    }
    // For other GSL errors, throw with more specific information
    std::string error_msg = "GSL hypergeometric error: " + std::string(gsl_strerror(stat));
    throw std::runtime_error(error_msg);
  }

  if(!std::isfinite(out.val) || out.val == 0){
    throw std::runtime_error("norm_const2 - hypergeometric diverging with infinity ");
  }

  return std::log(d + 1) + (d + c) * std::log(m) - std::log(out.val);
}

// [[Rcpp::export]]
double hyperg2(double a, double b, double c, double x){
  /**
  * @brief Computes the value of the Gauss hypergeometric function 2F1(a, b; c; x)
  * 
  * This function calculates the hypergeometric function 2F1(a, b; c; x) using the
  * GNU Scientific Library (GSL). It handles error cases by returning R_NaN
  * when the computation does not converge.
  * 
  * @param a First parameter of the hypergeometric function
  * @param b Second parameter of the hypergeometric function
  * @param c Third parameter of the hypergeometric function
  * @param x Argument of the hypergeometric function
  * @return The value of 2F1(a, b; c; x) if computation succeeds, R_NaN otherwise
  * 
  * @note Error handler is disabled during the computation to prevent program termination
  * @note Prints a message to Rcpp::Rcout when computation fails to converge
  */
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

// [[Rcpp::export]]
Rcpp::NumericVector dhyper_raf2(const Rcpp::NumericVector u, 
                                const double d, 
                                const double c,
                                const double m, 
                                const bool log_scale){
  /**
  * @brief Computes the density of a hypergeometric random variable using the RAF2 (Ratio Approximation Formula 2) method.
  * 
  * This function calculates the probability density function (PDF) of a hypergeometric distribution 
  * with parameters d, c, and m, evaluated at each value in vector u.
  * 
  * The hypergeometric RAF2 density is proportional to:
  *    u^d / (1 + u*(m-1))^(d+c)
  * where the normalizing constant is computed using norm_const2().
  * 
  * @param u A numeric vector of values at which to evaluate the density
  * @param d First shape parameter of the hypergeometric distribution
  * @param c Second shape parameter of the hypergeometric distribution
  * @param m Scale parameter of the hypergeometric distribution
  * @param log_scale Boolean indicating whether to return results on log scale (TRUE) or natural scale (FALSE)
  * 
  * @return A numeric vector of the same length as u containing density values,
  *         either on the log scale or natural scale depending on the log_scale parameter
  */

  int n = u.length();
  Rcpp::NumericVector out(n);

  double lK = norm_const2(d, c, m);

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
double newton_hyper2(const double d, const double c, const double m, const double Omega, const double u0){
  /**
  * @brief Solve for a specific value using Newton's method for a hypergeometric function equation.
  * 
  * This function uses Newton's method to numerically solve for a value u such that 
  * u/(d+1) * hyperg2(1, d+c, d+2, x) = Omega/density(u), where x is a function of u.
  * 
  * The method iterates until the absolute difference between successive approximations 
  * is less than 0.00001 or until 100 iterations are reached.
  * 
  * @param d Parameter for hypergeometric function
  * @param c Parameter for hypergeometric function
  * @param m Parameter affecting the transformation from u to x
  * @param Omega Target value for the equation
  * @param u0 Initial guess for u
  * 
  * @return The solution u if convergence is achieved, R_NaN if the method fails to converge after 100 iterations
  * 
  * @note The function bounds u between 0.01 and 0.99 during iterations to avoid boundary issues.
  * @note The function depends on dhyper_raf2() for density calculation and hyperg2() for hypergeometric function evaluation.
  */

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
double lF_conK2(const double u, const double d, const double c, const double m, const double lK){
  /**
  * @brief Computes the logarithm of the hypergeometric function with a normalizing constant.
  *
  * This function calculates the logarithm of the hypergeometric function
  * with parameters d, c, and m, evaluated at u. It also includes a normalizing constant lK.
  * The function is designed to handle edge cases where u is 0 or 1.
  *
  * @param u A numeric value at which to evaluate the function
  * @param d First shape parameter of the hypergeometric distribution
  * @param c Second shape parameter of the hypergeometric distribution
  * @param m Scale parameter of the hypergeometric distribution
  * @param lK Logarithm of the normalizing constant
  *
  * @return The logarithm of the hypergeometric function with the normalizing constant
  *         or -INFINITY if u is 0, or 0 if u is 1.
  *
  * @note The function uses the hyperg2() function to compute the hypergeometric function value.
  * @note The function is designed to be used in conjunction with the bisec_hyper2() function.
  */

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
  double out = lK - log(d + 1) + (d + 1) * log(u) - (d + c) * log(1 + u * (m - 1)) + log(app);
  return out;
}

/// Newtown method
// [[Rcpp::export]]
double bisec_hyper2(const double d, const double c, const double m, const double Omega){

  /**
  * @brief Solve for a specific value using the bisection method for a hypergeometric function equation.
  *
  * This function uses the bisection method to numerically solve for a value u such that
  * u/(d+1) * hyperg2(1, d+c, d+2, x) = Omega/density(u), where x is a function of u.
  *
  * The method iterates until the absolute difference between upper and lower bounds is less than 0.000000001
  * or until 150 iterations are reached.
  *
  * @param d Parameter for hypergeometric function
  * @param c Parameter for hypergeometric function
  * @param m Parameter affecting the transformation from u to x
  * @param Omega Target value for the equation
  *
  * @return The solution u if convergence is achieved, R_NaN if the method fails to converge after 150 iterations
  *
  * @note The function bounds u between 0 and 1 during iterations to avoid boundary issues.
  * @note The function depends on dhyper_raf2() for density calculation and hyperg2() for hypergeometric function evaluation.
  */
  
  double centro = 0.5;
  double lK = norm_const2(d, c, m);
  double app = lF_conK2(centro, d, c, m, lK) - log(Omega);

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
  }

  if (counter == max_count)
  {
    Rcpp::Rcout << "Warning the bisection algorithm reached max number of iterations " << counter << "\n";
  }

  return (centro);
}

// [[Rcpp::export]]
Rcpp::NumericVector rhyper_raf2(const int n, const double d, const double c, const double m){
  /**
   * @brief Generate random variables from the hypergeometric distribution using the RAF2 method.
   * 
   * This function generates random variables from the hypergeometric distribution
   * with parameters d, c, and m using the RAF2 method. It uses the bisection method
   * to find the appropriate value for each random variable.
   * 
   * @param n Number of random variables to generate
   * @param d First parameter of the hypergeometric distribution
   * @param c Second parameter of the hypergeometric distribution
   * @param m Third parameter of the hypergeometric distribution
   * @return A numeric vector of length n containing generated random variables
   */

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

// [[Rcpp::export]]
Rcpp::NumericVector rhyper_sig(const int n, const double d, const double c, const double m){
  /**
   * @brief Generate random variables from the hypergeometric distribution using the RAF2 method.
   * 
   * This function generates random variables from the hypergeometric distribution
   * with parameters d, c, and m using the RAF2 method. It uses the bisection method
   * to find the appropriate value for each random variable.
   * 
   * @param n Number of random variables to generate
   * @param d First parameter of the hypergeometric distribution
   * @param c Second parameter of the hypergeometric distribution
   * @param m Third parameter of the hypergeometric distribution
   * @return A numeric vector of length n containing generated random variables
   */

  Rcpp::NumericVector out(n);
  double Omega;
  // the output of a gsl special function
  for (int i = 0; i < n; i++)
  {
    Omega = R::runif(0, 1);
    out[i] = bisec_hyper2(d, c, m, Omega); // return u
  }

  return (-1 / Rcpp::log(out));
}

// [[Rcpp::export]]
Rcpp::NumericVector rhig(const int n, const double v, const double w, const double m){
/**
 * @brief Generate random variables from the hypergeometric inverse gamma distribution
 * @param n Number of random variables to generate
 * @param c First parameter of the hypergeometric distribution
 * @param w second parameter of the hypergeometric distribution
 * @param m Third parameter of the hypergeometric distribution
 * @return Random variables from the hypergeometric inverse gamma distribution
 */

  Rcpp::NumericVector out(n);
  
  // check if m-1/m is a quantile of the beta distribution greater than 0.1 to avoid long while loops
  if(R::qbeta(0.1, w + 1, v - 1, 1, 0) < (m - 1)/m && (m - 1)/m > 4/5){
    for(int i = 0; i < n; i++){
      double x = R::rbeta(w + 1, v - 1);
      // generate from a beta distribution until we find a value less than m-1/m
      while(x > (m - 1)/m){
        x = R::rbeta(w + 1, v - 1);
      }
      // transform the beta variable to the hypergeometric inverse gamma - (reference ipergeometric 2025 transformation from t to u)
      out[i] = x/((m - 1)*( 1 - x));
    }
  } 
  else {
    // generate from the correct hypergeometric inverse gamma distribution
    for(int i = 0; i < n; i++){
      double Omega = R::runif(0, 1);
      out[i] = bisec_hyper2(w, v, m, Omega);
    }
  }
  return -1 / Rcpp::log(out);
}

// [[Rcpp::export]]
Rcpp::NumericVector dhyper_sig_raf2(const Rcpp::NumericVector x, const double d, const double c,
                                    const double m, const bool log_scale){
  /**
    * @brief Computes the density of a hypergeometric random variable using the RAF2 (Ratio Approximation Formula 2) method.
    * 
    * This function calculates the probability density function (PDF) of a hypergeometric distribution 
    * with parameters d, c, and m, evaluated at each value in vector x.
    * 
    * The hypergeometric RAF2 density is proportional to:
    *    x^(-d-1) / (1 + exp(-1/x)*(m-1))^(d+c)
    * where the normalizing constant is computed using norm_const2().
    * 
    * @param x A numeric vector of values at which to evaluate the density
    * @param d First shape parameter of the hypergeometric distribution
    * @param c Second shape parameter of the hypergeometric distribution
    * @param m Scale parameter of the hypergeometric distribution
    * @param log_scale Boolean indicating whether to return results on log scale (TRUE) or natural scale (FALSE)
    * 
    * @return A numeric vector of the same length as x containing density values,
    *         either on the log scale or natural scale depending on the log_scale parameter
    */

  int n = x.length();
  Rcpp::NumericVector out(n);

  double lK = norm_const2(d, c, m);

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
