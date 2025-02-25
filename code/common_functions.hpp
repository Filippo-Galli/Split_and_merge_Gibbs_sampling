#ifndef COMMON_FUNCTIONS_H
#define COMMON_FUNCTIONS_H

#include <Rcpp.h>
#include <random>
#include <algorithm>
#include <vector>
#include <set>
#include <chrono>
#include <string>

#include <Rcpp.h>
#include <RcppGSL.h>  // Add this explicit RcppGSL include
#include <Rinternals.h>
#include <gsl/gsl_sf_hyperg.h>

//#include <gibbs_utility.cpp>

using namespace Rcpp;

extern bool debug_var;

namespace debug {
    template<typename... Args>
    void print(unsigned int tab_level, const char* func_name, int line, const std::string& message, const Args&... args);

}


#define DEBUG_PRINT(level, message, ...) \
    debug::print(level, __func__, __LINE__, message, ##__VA_ARGS__)

// Data structures
struct alignas(64) internal_state {
    IntegerVector c_i;
    List center;
    List sigma;
    int total_cls = 0;

    internal_state& operator=(const internal_state& other) {
        if (this != &other) {  // Self-assignment protection
            c_i = clone(other.c_i);
            center = clone(other.center);
            sigma = clone(other.sigma);
            total_cls = other.total_cls;
        }
        return *this;  // Enable chain assignments
    }
};

struct alignas(64) aux_data {
    NumericMatrix data;
    int n;
    IntegerVector attrisize;
    double gamma;
    NumericVector v;
    NumericVector w;
};

void validate_state(const internal_state& state, const std::string& message);

void print_internal_state(const internal_state& state, int interest = -1);

void print_progress_bar(int progress, int total, const std::chrono::steady_clock::time_point& start_time);

IntegerVector sample_initial_assignment(double K = 4, int n = 10);

NumericVector sample_center_1_cluster(const IntegerVector & attrisize, const List & probs = List());

List sample_centers(const int number_cls, const IntegerVector & attrisize); 

NumericVector sample_sigma_1_cluster(const IntegerVector & attrisize, 
                                    const NumericVector & v, 
                                    const NumericVector & w);

List sample_sigmas(const int number_cls, const aux_data & const_data);

IntegerVector unique_classes(const IntegerVector & c_i);

IntegerVector unique_classes_without_index(const IntegerVector & c_i, const int index_to_del);

int count_cluster_members(const IntegerVector& c_i, int exclude_index, int cls);

void clean_var(internal_state & updated_state, 
              const internal_state current_state, 
              const IntegerVector& existing_cls, 
              const IntegerVector& attrisize);

double dhamming_pippo(int x, int c, double s, int attrisize);

double compute_loglikelihood(internal_state & state, aux_data & const_data);

NumericMatrix subset_data_for_cluster(const NumericMatrix& data, int cluster, const internal_state& state);

NumericVector compute_frequencies(const NumericVector& data_col, const int m_j);

List Center_prob_pippo(const NumericMatrix& data, const NumericVector& sigma, const IntegerVector & attrisize);

void update_centers(internal_state& state, const aux_data& const_data, 
                   std::vector<int> cluster_indexes = {});

void update_sigma(internal_state& state, const aux_data & const_data, std::vector<int> clusters_to_update = {});

#endif