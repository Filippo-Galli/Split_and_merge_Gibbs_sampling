/** 
 * @brief This file contains the class Data, which is used to store the data and parameters of the model.
 */

#ifndef DATA_HPP
#define DATA_HPP

#include <Rcpp.h>
#include <random>
#include <algorithm>
#include <vector>
#include <set>
#include <unordered_map>

#include <Rinternals.h>

using namespace Rcpp;

typedef IntegerVector IntVec;
typedef NumericVector DoubleVec;
typedef List DoubleMat;

struct internal_state {
    IntVec c_i;
    DoubleMat center;
    DoubleMat sigma;
    int total_cls = 0;

    internal_state() = default;
};

struct aux_data {
    NumericMatrix data;
    int n;
    IntVec attrisize;
    double gamma;
    DoubleVec v;
    DoubleVec w;

    aux_data() = default;
};

class Data {
    private: 
    // State variables
    IntVec c;
    DoubleMat center;
    DoubleMat sigma;
    int total_cls = 0;

    // Parameters
    const NumericMatrix data; // data matrix
    const int n; // number of rows
    const int p; // number of features
    const int m; // number of latent classes
    const IntVec attrisize; // vector of numbers of possibile attribute per feature
    const double gamma;
    const DoubleVec v;
    const DoubleVec w;

    // cache - if needed

    public:

    // Constructor
    Data(const NumericMatrix & data, const IntVec & attrisize, const double gamma, 
         const  DoubleVec & v, const  DoubleVec & w, const int m);

    // setters for State variables
    void set_c(const int & idx, const int & value);
    void set_c(const IntVec & c);
    void set_center(const int & idx, const DoubleVec & center);
    void set_center(const DoubleMat & center);
    void set_sigma(const int & idx, const DoubleVec & sigma);
    void set_sigma(const DoubleMat & sigma);
    void set_total_cls(const int total_cls);

    // getters for State variables
    IntVec get_c() const;
    double get_c(const int & idx) const;
     DoubleVec get_center(const int & idx) const;
    DoubleMat get_center() const;
     DoubleVec get_sigma(const int & idx) const;
    DoubleMat get_sigma() const;
    int get_total_cls() const;

    // getters for Parameters
     DoubleVec get_data_row(const int & idx) const;
     DoubleVec get_data_col(const int & feature) const;
    double get_data(const int & idx, const int & feature) const;
    NumericMatrix get_data() const;
    int get_n() const;
    int get_p() const;
    int get_m() const;
    double get_attrisize(const int & idx) const;
    IntVec get_attrisize() const;
    double get_gamma() const;
    double get_v(const int & idx) const;
     DoubleVec get_v() const;
    double get_w(const int & idx) const;
     DoubleVec get_w() const;

    // Utility functions
    IntVec get_cluster_indices(const int & cluster) const;
    NumericMatrix get_cluster_data(const int & cluster) const;
     DoubleVec get_cluster_center(const int & cluster) const;
     DoubleVec get_cluster_sigma(const int & cluster) const;
     DoubleVec get_unique_clusters() const;
     DoubleVec get_unique_clusters_without_idx(const int & idx) const;
    int get_cluster_size(const int & cluster) const;
     DoubleVec get_cluster_sizes() const;
    void clean_var(IntegerVector & c_to, DoubleMat & center_to, DoubleMat & sigma_to,
                    const IntVec & c_from, const DoubleMat & center_from, const DoubleMat & sigma_from,
                    const  DoubleVec& existing_cls) const;
    void clean_var(IntegerVector & c, DoubleMat & center, DoubleMat & sigma,
                    const  DoubleVec& existing_cls) const;
    void clean_var();
    void initialize_memory();
    void print(const int interest = -1) const;
};



#endif