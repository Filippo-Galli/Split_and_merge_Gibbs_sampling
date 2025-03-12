#ifndef SPLIT_MERGE_HPP
#define SPLIT_MERGE_HPP

#include "common_functions.hpp"

double logdensity_hig(double sigmaj, double v, double w, double m);

double logprobgs_phi(const internal_state & gamma_star, const internal_state & gamma,
                    const aux_data & const_data, const int & choosen_idx);

double logprobgs_c_i(const internal_state & gamma_star, const internal_state & gamma, const aux_data & const_data, const std::vector<int> & S, const int i_1, const int i_2);

void split_restricted_gibbs_sampler(const std::vector<int> & S, internal_state & state, int i_1, int i_2, const aux_data & const_data, int t = 1);

void select_observations_random(const internal_state & state, int & i_1, int & i_2, std::vector<int> & S);

void select_observations_deterministic(const internal_state & state, int & i_1, int & i_2, std::vector<int> & S);

internal_state split_launch_state(const std::vector<int> & S, const internal_state & state, int i_1, int i_2, int t, const aux_data & const_data);

internal_state merge_launch_state(const std::vector<int> & S, const internal_state & state, int i_1, int i_2, int r, const aux_data & const_data);

double loglikelihood_hamming(const internal_state & state, int c, const aux_data & const_data);

double priors(const internal_state & state, int c, const aux_data & const_data);

double split_acc_prob(const internal_state & state_split,
                     const internal_state & state,
                     const internal_state & split_launch,
                     const internal_state & merge_launch,
                     const std::vector<int> & S,
                     int i_1, int i_2,
                     const aux_data & const_data);

double merge_acc_prob(const internal_state & state_merge,
                     const internal_state & state,
                     const internal_state & split_launch,
                     const internal_state & merge_launch,
                     const std::vector<int> & S,
                     int i_1, int i_2,
                     const aux_data & const_data);

int split_and_merge(internal_state & state,
                    const aux_data & const_data,
                    int t, int r, int & idx_1_sm, 
                    IntegerVector & gt);

#endif // SPLIT_MERGE_HPP