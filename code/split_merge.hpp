#ifndef SPLIT_MERGE_HPP
#define SPLIT_MERGE_HPP

#include "updater.hpp"

class SplitMerge : public Updater {

    private: 
        IntVec S;
        int i1, i2;

        Updater split_launch_state;
        Updater merge_launch_state;
        Updater state_star;

    public:
        // Constructor
        SplitMerge(const NumericMatrix & data, const IntVec & attrisize, const double gamma, 
                const  DoubleVec & v, const  DoubleVec & w, const int & m): Updater(data, attrisize, gamma, v, w, m) {};

        // Utility 
        void select_observation();
        void split_launch_state_creation(const int t);
        void merge_launch_state_creation(const int r);
        void split_restricted_gibbs_sampler();
        void split_restricted_gibbs_sampler(Updater & state_star);

        // Probability functions
        double split_acc_prob() const;
        double merge_acc_prob() const;
        double logprior() const;

        // Split-Merge functions
        void split_merge_step(const int & t, const int & r);

    
};

#endif