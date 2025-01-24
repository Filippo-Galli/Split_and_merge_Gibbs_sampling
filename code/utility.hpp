#ifndef UTILITY_HPP
#define UTILITY_HPP
// [[Rcpp::depends(RcppGSL)]]
#include <Rcpp.h>
#include <RcppGSL.h>
#include <iostream>
#include <gsl/gsl_sf_hyperg.h>
#include <cmath>

using namespace Rcpp;

extern bool debug_var;

// Debug utilities
namespace debug {
    template<typename... Args>
    void print(unsigned int tab_level, const char* func_name, int line, const std::string& message, const Args&... args) {
        /**
         * @brief Print a debug message to the console
         * @param tab_level Number of tabs to indent the message
         * @param func_name Name of the function where the message is printed
         * @param line Line number where the message is printed
         * @param message Message to print
         * @param args Additional arguments to format the message
         * @note This function is used to print debug messages to the console
         *      with additional information such as the function name and line number
         */
        std::stringstream ss;
        for(unsigned int i = 0; i < tab_level; ++i) {
            ss << '\t';
        }
        ss << "[DEBUG:" << func_name << ":" << line << "] ";
        
        size_t pos = 0;
        size_t count = 0;
        std::string temp = message;
        
        ((ss << temp.substr(pos, message.find("{}", pos) - pos) 
            << args
            , pos = message.find("{}", pos) + 2
            , count++), ...);
            
        ss << message.substr(pos);
        if(debug_var)
            Rcpp::Rcout << ss.str() << std::endl;
    }
}

#define DEBUG_PRINT(level, message, ...) \
    debug::print(level, __func__, __LINE__, message, ##__VA_ARGS__)


class Utility {
    public:
    List Center_prob(const NumericMatrix & data, const NumericVector & sigma, const NumericVector & attrisize) const;
    List Center_prob_2(NumericMatrix data, double sigma, NumericVector attrisize) const; 
    NumericVector Samp_Center(List attriList, List center_prob, int p) const;
    double dhamming(int x, int c, double s, int attrisize, bool log_scale) const;
    List Attributes_List(NumericMatrix data, int p) const;
    List Attributes_List_manual(NumericMatrix data, int p) const;
    int hamming_distance(NumericVector a, NumericVector b) const;
    NumericMatrix alloc_matrix(NumericMatrix cent, NumericMatrix sigma, NumericMatrix data, int M_curr, 
                            NumericVector attrisize, NumericVector Sm, bool log_scale) const;
    NumericMatrix alloc_matrix2(NumericMatrix cent, NumericVector sigma, NumericMatrix data, int M_curr,
                            NumericVector attrisize, NumericVector Sm, bool log_scale) const;
    double norm_const(const double d ,const double c, const double m) const;
    double hyperg(double a, double b, double c, double x) const;
    Rcpp::NumericVector rhyper_raf(const int n,const double d ,const double c, const double m) const;
    Rcpp::NumericVector dhyper_raf(const Rcpp::NumericVector u,const double d ,const double c,const double m, const bool log_scale) const;
    double newton_hyper(const double d,const double c,const double m,const double Omega,const  double u0) const;
    double lF_conK(const double u, const double d,const double c,const double m,const double K) const;
    double bisec_hyper(const double d,const double c,const double m,const double Omega) const;
    Rcpp::NumericVector rhyper_sig(const int n,const double d ,const double c, const double m) const;
    Rcpp::NumericVector dhyper_sig_raf(const Rcpp::NumericVector x,const double d ,const double c, const double m, const bool log_scale) const;
};

#endif

