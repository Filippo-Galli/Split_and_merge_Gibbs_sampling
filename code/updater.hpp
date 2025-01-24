#ifndef UPDATER_HPP
#define UPDATER_HPP

#include "sampler.hpp"

class Updater : public Sampler {
    private:
    // Cache

    public:
    // Constructor
    Updater(const NumericMatrix & data, const IntVec & attrisize, const double gamma, 
            const  DoubleVec & v, const  DoubleVec & w, const int & m);

    // Update functions
    void update_center();
    void update_center(const std::vector<int> & cluster_indexes);
    void update_sigma();
    void update_sigma(const std::vector<int> & cluster_indexes);

    double compute_loglikelihood() const;
};

#endif