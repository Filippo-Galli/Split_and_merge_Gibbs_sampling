/**
 * @brief this file contains the sampler class which is used to sample the state variables of the model
 */

#ifndef SAMPLER_HPP
#define SAMPLER_HPP

#include "data.hpp"
#include "utility.hpp"

class Sampler : public Data, public Utility {
    private:

    // Cache

    public:
    // Constructor
    Sampler(const NumericMatrix & data, const IntVec & attrisize, const double gamma, 
            const  DoubleVec & v, const  DoubleVec & w, const int & m);

    // Sampler functions
    void sample_c_1_obs(const int & index_i);
    void sample_c();
     DoubleVec sample_center_1_cluster() const;
     DoubleVec sample_center_1_cluster(const DoubleMat & probs) const;
    void sample_center();
     DoubleVec sample_sigma_1_cluster() const;
     DoubleVec sample_sigma_1_cluster(const  DoubleVec& v, const  DoubleVec& w) const;
    void sample_sigma();
};


#endif


