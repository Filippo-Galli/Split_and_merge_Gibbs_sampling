/**
 * @file neal8.cpp
 * @brief Implementation of Neal's Algorithm 8 for Dirichlet Process Mixture Models
 */

#include "./neal8.hpp"

using namespace Rcpp;

void sample_allocation(int idx, 
                        const aux_data & const_data, 
                        internal_state & state, 
                        const int m, 
                        const std::vector<NumericVector> & latent_center_reuse, 
                        const std::vector<NumericVector> & latent_sigma_reuse) {
    /**
     * @brief Sample new cluster assignment for observation i
     * @param idx Index of the observation
     * @param const_data Auxiliary data for the MCMC algorithm
     * @param state Internal state of the MCMC sampler
     * @param m Number of latent classes
     * @param latent_center_reuse Precomputed latent centers
     * @param latent_sigma_reuse Precomputed latent sigmas
     * @note This function is used to sample new cluster assignments for each observation
     *       based on the current state of the MCMC sampler following the Neal's Algorithm 8.
     */

    // Get data point
    const NumericVector & y_i = const_data.data(idx, _);
    const IntegerVector & unique_class = unique_classes(state.c_i);
    const IntegerVector & unique_classes_without_i = unique_classes_without_index(state.c_i, idx);
    const int k = unique_class.length(); // numeri di cluster attivi
    const int k_minus = unique_classes_without_i.length(); // numeri di cluster attivi escludendo l'osservazione corrente
    
    // Prepare vector of probabilities
    NumericVector probs(k + m);

    // Calculate allocation probabilities for existing clusters
    double log_likelihood = 0.0;
    for(int i = 0; i < k; ++i){
        log_likelihood = 0.0;
        
        const NumericVector & sigma_k = state.sigma[i];
        const NumericVector & center_k = state.center[i];

        // Calculate likelihood
        for (int j = 0; j < y_i.length(); j++) {
            log_likelihood += dhamming(y_i[j], center_k[j], sigma_k[j], const_data.attrisize[j]);
        }

        // Count instances of element in the cluster i excluding idx
        int n_i_z = sum(state.c_i == i) - (state.c_i[idx] == i);

        // Set probability
        probs[i] = n_i_z != 0 ? log(n_i_z) + log_likelihood : -INFINITY ; //if n_i_z == 0 then probs[i] = 0
    }

    // Prepare Latent Cluster vectors of parameters
    std::vector<NumericVector> latent_centers;
    latent_centers.reserve(m);
    std::vector<NumericVector> latent_sigmas;
    latent_sigmas.reserve(m);

    // initialize latent clusters
    for(int i = 0; i < m; ++i){
        int idx_latent = sample(latent_center_reuse.size(), 1, false)[0] - 1;
        latent_centers.push_back(latent_center_reuse[idx_latent]);
        latent_sigmas.push_back(latent_sigma_reuse[idx_latent]);
    }

    // If the observation is unique in its cluster, that cluster becomes a latent cluster
    if(k_minus < k){  
        latent_centers[0] = state.center[state.c_i[idx]];
        latent_sigmas[0] = state.sigma[state.c_i[idx]];
    }

    // Calculate allocation probabilities for latent clusters
    const double log_factor = std::log(const_data.gamma/m);
    for(int i = 0; i < m; ++i){
        log_likelihood = 0.0;
        
        const NumericVector & sigma_k = latent_sigmas[i];
        const NumericVector & center_k = latent_centers[i];

        // Calculate likelihood
        for (int j = 0; j < y_i.length(); j++) {
            log_likelihood += dhamming(y_i[j], center_k[j], sigma_k[j], const_data.attrisize[j]);
        }

        // probability calculation
        probs[k + i] = log_factor + log_likelihood;
    }

    // Normalize probabilities
    probs = exp((probs - max(probs)));
    probs = probs / sum(probs);

    // Sample new allocation
    NumericVector cls(k + m);
    std::iota(cls.begin(), cls.end(), 0); // fill cls with 0, 1, 2, ..., k + m - 1

    int new_cls = sample(cls, 1, true, probs)[0];
    int old_cls = state.c_i[idx];

    // Update center and sigma
    // Case 1: the sampled allocation is a known class and we do not remove any class (the observation is not unique)
    if( sum(state.c_i == state.c_i[idx]) != 1 && new_cls < k){
        // Update allocation
        state.c_i[idx] = new_cls;
        validate_state(state, "Neal8 case 1"); // check consistency of the states
        return;
    }

    // Case 2: we take a known class and remove a class (we are analyzing a unique observation)
    if( new_cls < k && sum(state.c_i == state.c_i[idx]) == 1 ){
        // Update allocation
        state.c_i[idx] = new_cls;
        // move the last class to the position of the class to be removed
        state.center[old_cls] = std::move(state.center[k - 1]);
        state.sigma[old_cls] = std::move(state.sigma[k - 1]);

        // remove the last class
        state.center.erase(k - 1);
        state.sigma.erase(k - 1);

        // update allocation 
        for(int i = 0; i < const_data.n; ++i){
            if(state.c_i[i] == (k - 1)){
                state.c_i[i] = old_cls;
            }
        }

        // update the number of active classes
        state.total_cls = k - 1;
        validate_state(state, "Neal8 case 2");
        return;
    }

    // Case 3: we take a latent class and do not remove any class
    if( new_cls >= k && sum(state.c_i == state.c_i[idx]) != 1){
        // Update allocation
        state.c_i[idx] = k;
        state.center.push_back(latent_centers[new_cls - k]);
        state.sigma.push_back(latent_sigmas[new_cls - k]);

        // Update the number of active classes
        state.total_cls += 1;
        validate_state(state, "Neal8 case 3");
        return;
    }

    // Case 4: we take a latent class and remove a class
    if (new_cls >= k && sum(state.c_i == state.c_i[idx]) == 1){
        // Move the last class to the position of the class to be removed
        state.center[old_cls] = std::move(latent_centers[new_cls - k]);
        state.sigma[old_cls] = std::move(latent_sigmas[new_cls - k]);
        validate_state(state, "Neal8 case 4");
        return;
    }
}
