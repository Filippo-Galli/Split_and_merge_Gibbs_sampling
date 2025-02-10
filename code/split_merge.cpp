#include "split_merge.hpp"
#include <cmath>
#include "hyperg.hpp"

double logdensity_hig(double sigmaj, double v, double w, double m){
    /**
     * @brief logdensity of hig(v,w,m)(sigmaj)
     * @param sigmaj sigma parameter
     * @param v v parameter
     * @param w w parameter
     * @param m number of attribute levels
     * @return logdensity of hig(v,w,m)(sigmaj)
     */
    double K = norm_const(w ,v, m); 
    return log(K) - (w + 1)/sigmaj - (v + w)*log(1+exp(-1/sigmaj)*(m-1)) - 2*log(sigmaj);

}

double logprobgs_phi(const internal_state & gamma_star, const internal_state & gamma,
                    const aux_data & const_data, const int & choosen_idx) {
    /**
     * @brief Compute the probability of the parameters - paper reference: P_{GS}(\phi*|phi^L, y)
     * @param gamma_star state containing the new cluster assignment, new cluster centers, and new cluster dispersions
     * @param gamma state containing the launch cluster assignment, launch cluster centers, and launch cluster dispersions - paper reference: (c^L, \phi^L)
     * @param const_data auxiliary data for the MCMC algorithm containing the input data matrix, attribute sizes, and hypergeometric parameters
     * @param choosen_idx index of the chosen observation
     */

    double log_center_prob = 0;
    
    // Get cluster for chosen observation
    int c = gamma_star.c_i[choosen_idx];

    // Calculate conditional probability of parameters given data
    const NumericMatrix& data_tmp = subset_data_for_cluster(const_data.data, c, gamma_star);

    // Prior probability of centers
    const List& prob_centers = Center_prob_pippo(data_tmp, 
                                gamma.sigma[gamma.c_i[choosen_idx]], 
                                const_data.attrisize);

    // Calculate probability of center parameters
    const NumericVector& centerstar = gamma_star.center[c];
    for(int j = 0; j < centerstar.length(); j++) {
        NumericVector z = prob_centers[j];
        
        // Convert probabilities to log space
        NumericVector log_z(z.length());        
        log_center_prob += log_z[centerstar[j] - 1];
    }

    double log_sigma_prob = 0;
    // Calculate probability of sigma parameters
    const NumericVector& sigstar = gamma_star.sigma[c];
    int nm = data_tmp.nrow();
    const NumericVector& centers = gamma_star.center[c];
    
    for(int j = 0; j < const_data.attrisize.length(); j++) {
        const NumericVector& col = data_tmp(_, j);
        double sumdelta = sum(col == centers[j]);
        double new_v = const_data.v[j] + sumdelta;
        double new_w = const_data.w[j] + nm - sumdelta;
        
        log_sigma_prob += logdensity_hig(sigstar[j], new_v, new_w, const_data.attrisize[j]);
    }

    return log_center_prob + log_sigma_prob;
}


double logprobgs_c_i(const internal_state & gamma_star, const internal_state & gamma, const aux_data & const_data, const std::vector<int> & S, const int i_1, const int i_2){
    /**
     * @brief Compute the probability of the cluster assignment - paper reference: P_{GS}(c*|c^L, phi*, y)
     * @param gamma_star state containing the new cluster assignment, new cluster centers, and new cluster dispersions
     * @param gamma state containing the launch cluster assignment, launch cluster centers, and launch cluster dispersions - paper reference: (c^L, \phi^L)
     * @param const_data auxiliary data for the MCMC algorithm containing the input data matrix, attribute sizes, and hypergeometric parameters
     * @param S vector of indices of observations in the same cluster of i_1 or i_2
     * @param i_1 index of the first chosen observation
     * @param i_2 index of the second chosen observation
     */

    // Variable to store the logprobability of the cluster assignment
    double logpgs = 0;

    internal_state aux_state = gamma;

    // Extract cluster of the first observation
    int c_i_1 = gamma.c_i[i_1];
    const NumericVector& center1 = as<NumericVector>(gamma_star.center[c_i_1]);
    const NumericVector& sigma1 = as<NumericVector>(gamma_star.sigma[c_i_1]);

    // Extract cluster of the second observation
    int c_i_2 = gamma.c_i[i_2];
    const NumericVector& center2 = as<NumericVector>(gamma_star.center[c_i_2]);
    const NumericVector& sigma2 = as<NumericVector>(gamma_star.sigma[c_i_2]);

    // support variable
    NumericVector probs(2);
    NumericVector center(const_data.attrisize.length());
    NumericVector sigma(const_data.attrisize.length());
    int cls;
    int n_s_cls;

    for (int s : S) {
        // extract datum at s
        const NumericVector& y_s = const_data.data(s, _);

        // evaluate probabilities
        for (int k = 0; k < 2; k++) {
            // select parameter values of the corresponding cluster
            if(k == 0){
                center = center1;
                sigma = sigma1;
                cls = c_i_1;
            }
            else{
                center = center2;
                sigma = sigma2;
                cls = c_i_2;
            }

            double Hamming = 0;

            for (int j = 0; j < y_s.length(); j++) {
                Hamming += dhamming_pippo(y_s[j], center[j], sigma[j], const_data.attrisize[j]);
            }

            // Count instances in the cluster excluding the current point s
            n_s_cls = count_cluster_members(aux_state.c_i, s, cls);
            
            probs[k] =  n_s_cls * std::exp(Hamming);
        }

        // Normalize probabilities
        probs = probs / sum(probs);

        // update for the current value of s
        aux_state.c_i[s] = gamma_star.c_i[s];

        int currrent_c = gamma_star.c_i[s] == c_i_1 ? 0 : 1;

        //  logprob of be assigned to the new state
        logpgs += log(probs[currrent_c]);

    }
    
    return logpgs;
}

void split_restricted_gibbs_sampler(const std::vector<int> & S, internal_state & state, int i_1, int i_2, const aux_data & const_data) {
    /**
     * @brief Restricted Gibbs Sampler between the cluster of two observations
     * @param state Internal state of the MCMC algorithm
     * @param i_1 Index of the first chosen observation
     * @param i_2 Index of the second chosen observation
     * @param S Vector of indices of observations in the same cluster of i_1 or i_2
     * @param const_data Auxiliary data for the MCMC algorithm
     */

    // Extract cluster of the first observation
    int c_i_1 = state.c_i[i_1];
    const NumericVector & center1 = as<NumericVector>(state.center[c_i_1]);
    const NumericVector & sigma1 = as<NumericVector>(state.sigma[c_i_1]);

    // Extract cluster of the second observation
    int c_i_2 = state.c_i[i_2];
    const NumericVector & center2 = as<NumericVector>(state.center[c_i_2]);
    const NumericVector & sigma2 = as<NumericVector>(state.sigma[c_i_2]);

    // support variables
    NumericVector probs(2);
    NumericVector center(const_data.attrisize.length());
    NumericVector sigma(const_data.attrisize.length());
    int cls;
    int n_s_cls;

    for (int s : S) {
        // extract datum at s
        const NumericVector & y_s = const_data.data(s, _);

        // evaluate probabilities
        for (int k = 0; k < 2; k++) {
            // select parameter values of the corresponding cluster
            if(k == 0){
                center = center1;
                sigma = sigma1;
                cls = c_i_1;
            }
            else{
                center = center2;
                sigma = sigma2;
                cls = c_i_2;
            }

            double Hamming = 0;

            for (int j = 0; j < y_s.length(); j++) {
                Hamming += dhamming_pippo(y_s[j], center[j], sigma[j], const_data.attrisize[j]);
            }

            // Count instances in the cluster excluding the current point s
            n_s_cls = count_cluster_members(state.c_i, s, cls);  

            probs[k] =  log(n_s_cls) + Hamming;
        }

        // Normalize probabilities
        probs = exp(probs - max(probs));
        probs = probs / sum(probs);
       
        // Sample new cluster assignment between the two clusters of i_1 and i_2
        state.c_i[s] = sample(IntegerVector::create(c_i_1, c_i_2), 1, true, probs)[0];
    }

    // check for empty clusters
    //clean_var(state, state, unique_classes(state.c_i), const_data.attrisize);

    update_centers(state, const_data, {c_i_1, c_i_2});
    update_sigma(state.sigma, state.center, state.c_i, const_data, {c_i_1, c_i_2});
}

void select_observations(const internal_state & state, int & i_1, int & i_2,std::vector<int> & S) {
    /**
     * @brief Select two observations and populate S with indexes in selected clusters
     * @param state Internal state of the MCMC algorithm
     * @param i_1 Index of the first chosen observation
     * @param i_2 Index of the second chosen observation
     * @param S Vector of indices of observations in the same cluster of i_1 or i_2
     */

    // Sample 2 observations
    IntegerVector indices = seq(0, state.c_i.size() - 1);
    IntegerVector sample_indexes = sample(indices, 2, false);

    // Extract indexes
    i_1 = sample_indexes[0];
    i_2 = sample_indexes[1];

    // Reserve space for S  to avoid multiple reallocations
    if(i_1 == i_2){
        // count of the number of observations in the cluster of i_1
        int n_i_1 = sum(state.c_i == state.c_i[i_1]) - 2;
        S.reserve(n_i_1);
    }
    else{
        int n_i_1 = sum(state.c_i == state.c_i[i_1]) - 1; // number of observations in the cluster of i_1
        int n_i_2 = sum(state.c_i == state.c_i[i_2]) - 1; // number of observations in the cluster of i_2
        S.reserve(n_i_1 + n_i_2);
    }

    // Populate S with indexes in selected clusters
    for(int i = 0; i < state.c_i.length(); ++i) {
        if(i == i_1 || i == i_2) 
            continue;
        if(state.c_i[i] == state.c_i[i_1] || state.c_i[i] == state.c_i[i_2]) {
            S.push_back(i);
        }
    }
}

internal_state split_launch_state(const std::vector<int> & S,const internal_state & state,
                                int i_1, int i_2, int t, const aux_data & const_data) {
    /**
     * @brief Generate the launch state for the split move
     * @param S Vector of indices of observations in the same cluster of i_1 or i_2
     * @param state Internal state of the MCMC algorithm
     * @param i_1 Index of the first chosen observation
     * @param i_2 Index of the second chosen observation
     * @param t Number of restricted Gibbs scans
     * @param const_data Auxiliary data for the MCMC algorithm
     * @return Internal state of the MCMC algorithm
     */
    
    // Initialize cluster assignments
    IntegerVector S_indexes = wrap(S);

    // Initialize split launch state
    internal_state state_launch_split = {clone(state.c_i), clone(state.center), clone(state.sigma), state.total_cls};
    
    // Initialize centers and sigmas based on cluster equality
    if (state.c_i[i_1] == state.c_i[i_2]) {
        // Assign new cluster to the first observation
        state_launch_split.c_i[i_1] = state.total_cls; 
        state_launch_split.center.push_back(sample_center_1_cluster(const_data.attrisize));
        state_launch_split.sigma.push_back(sample_sigma_1_cluster(const_data.attrisize, const_data.v, const_data.w));
        state_launch_split.total_cls++; // increase the number of clusters
    } 

    // Random allocation of S between clusters
    state_launch_split.c_i[S_indexes] = sample(IntegerVector::create(state_launch_split.c_i[i_1], state_launch_split.c_i[i_2]), S_indexes.length(), true);

    // Intermediate Gibbs sampling
    for(int iter = 0; iter < t; ++iter) {
        split_restricted_gibbs_sampler(S, state_launch_split, i_1, i_2, const_data);
    }

    validate_state(state_launch_split, "split_launch_state");

    // // Initialize split launch state
    // IntegerVector c_L_split = clone(state.c_i);
    // List center_L_split = clone(state.center);
    // List sigma_L_split = clone(state.sigma);

    // if(state.c_i[i_1] == state.c_i[i_2]) {
    //     c_L_split[i_1] = unique_classes(state.c_i).length();
    //     center_L_split.push_back(0);
    //     sigma_L_split.push_back(0);
    // }

    // // Random allocation of S between clusters
    // IntegerVector S_indexes = wrap(S);
    // c_L_split[S_indexes] = sample(IntegerVector::create(c_L_split[i_1], c_L_split[i_2]), S_indexes.length(), true);
    // // Sample new parameters
    // center_L_split[c_L_split[i_1]] = sample_center_1_cluster(const_data.attrisize);
    // sigma_L_split[c_L_split[i_1]] = sample_sigma_1_cluster(const_data.attrisize, const_data.v, const_data.w);
    // center_L_split[c_L_split[i_2]] = sample_center_1_cluster(const_data.attrisize);
    // sigma_L_split[c_L_split[i_2]] = sample_sigma_1_cluster(const_data.attrisize, const_data.v, const_data.w);

    // internal_state state_launch_split = {c_L_split, center_L_split, sigma_L_split, static_cast<int>(unique_classes(c_L_split).length())};

    // // Da controllare che funzioni siccome usa lo stesso stato come input e output - [WARNING] perchè la facciamo? Nessun cluster può essere vuoto
    // clean_var(state_launch_split, state_launch_split, unique_classes(state_launch_split.c_i), const_data.attrisize);

    // // Intermediate Gibbs sampling
    // for(int iter = 0; iter < t; ++iter) {
    //     split_restricted_gibbs_sampler(S, state_launch_split, i_1, i_2, const_data);
    // }

    return state_launch_split;
}

internal_state merge_launch_state(const std::vector<int> & S,
                                const internal_state & state,
                                int i_1, int i_2, int r,
                                const aux_data & const_data) {
    /**
     * @brief Generate the launch state for the merge move
     * @param S Vector of indices of observations in the same cluster of i_1 or i_2
     * @param state Internal state of the MCMC algorithm
     * @param i_1 Index of the first chosen observation
     * @param i_2 Index of the second chosen observation
     * @param r Number of restricted Gibbs scans
     * @param const_data Auxiliary data for the MCMC algorithm
     * @return Internal state of the MCMC algorithm
     */

    // internal_state state_launch_merge = {clone(state.c_i), clone(state.center), clone(state.sigma), state.total_cls};

    // if(state.c_i[i_1] != state.c_i[i_2]) {
    //     // change allocation from c_i[i_1] to c_i[i_2]
    //     IntegerVector S_indexes = wrap(S);
    //     state_launch_merge.c_i[S_indexes] = state.c_i[i_2];

    //     // To avoid empty clusters [clean_var()]
    //     // move the last cluster to the cluster of i_1
    //     state_launch_merge.center[state.c_i[i_1]] = state.center[state.total_cls - 1];
    //     state_launch_merge.sigma[state.c_i[i_1]] = state.sigma[state.total_cls - 1];

    //     // remove the last cluster
    //     state_launch_merge.center.erase(state.total_cls - 1);
    //     state_launch_merge.sigma.erase(state.total_cls - 1);

    //     // correct the indexes of the last cluster
    //     for(int i = 0; i < state.c_i.length(); ++i) {
    //         if(state.c_i[i] == state.total_cls - 1) {
    //             state_launch_merge.c_i[i] = state.c_i[i_1];
    //         }
    //     }

    //     // decrease the number of clusters
    //     state_launch_merge.total_cls--;

    //     // Move allocation of i_1 to the allocation of i_2
    //     state_launch_merge.c_i[i_1] = state.c_i[i_2];
    // }
    // validate_state(state_launch_merge, "merge_launch_state pre update");

    // // Update parameters r times
    // for(int iter = 0; iter < r; ++iter) {
    //     update_centers(state_launch_merge, const_data, {state_launch_merge.c_i[i_2]});
    //     update_sigma(state_launch_merge.sigma, state_launch_merge.center, state_launch_merge.c_i, const_data, {state_launch_merge.c_i[i_2]});
    // }


    // Initialize merge launch state
    IntegerVector c_L_merge = clone(state.c_i);
    List center_L_merge = clone(state.center);
    List sigma_L_merge = clone(state.sigma);

    if(state.c_i[i_1] != state.c_i[i_2]) {
        c_L_merge[i_1] = c_L_merge[i_2];
        IntegerVector S_indexes = wrap(S);
        c_L_merge[S_indexes] = c_L_merge[i_2];
    }

    // Draw new parameters for merged component
    center_L_merge[c_L_merge[i_2]] = sample_center_1_cluster(const_data.attrisize);
    sigma_L_merge[c_L_merge[i_2]] = sample_sigma_1_cluster(const_data.attrisize, const_data.v, const_data.w);

    internal_state state_launch_merge = {c_L_merge, center_L_merge, sigma_L_merge, static_cast<int>(unique_classes(c_L_merge).length())};

    clean_var(state_launch_merge, state_launch_merge, unique_classes(state_launch_merge.c_i), const_data.attrisize);                                 

    // Update parameters r times
    for(int iter = 0; iter < r; ++iter) {
        update_centers(state_launch_merge, const_data, {state_launch_merge.c_i[i_2]});
        update_sigma(state_launch_merge.sigma, state_launch_merge.center, state_launch_merge.c_i, const_data, {state_launch_merge.c_i[i_2]});
    }

    validate_state(state_launch_merge, "merge_launch_state");

    return state_launch_merge;
}

double loglikelihood_hamming(const internal_state & state, int c, const aux_data & const_data) {
    /**
     * @brief Compute the log-likelihood of the current state
     * @param state Internal state of the MCMC algorithm
     * @param c Cluster index
     * @param const_data Auxiliary data for the MCMC algorithm
     * @return Log-likelihood of the current state
     */

    double loglikelihood = 0.0;
    const NumericVector & center = as<NumericVector>(state.center[c]);
    const NumericVector & sigma = as<NumericVector>(state.sigma[c]);
    
    for (int i = 0; i < const_data.n; i++) {
        if(state.c_i[i] == c) {
            for (int j = 0; j < const_data.attrisize.length(); j++) {
                loglikelihood += dhamming_pippo(const_data.data(i, j),
                                      center[j],
                                      sigma[j],
                                      const_data.attrisize[j]);
            }
        }
    }
    return loglikelihood;
}

int cls_elem(const internal_state & state, int c) {
    /**
     * @brief Count the number of elements in cluster c
     * @param state Internal state of the MCMC algorithm
     * @param c Cluster index
     * @return Number of elements in cluster c
     */

    int f = 0;
    for (int i = 0; i < state.c_i.length(); i++) {
        if (state.c_i[i] == c) f++;
    }
    return f;
}

double priors(const internal_state & state, int c, const aux_data & const_data){
    /**
     * @brief Compute the prior of the current state
     * @param state Internal state of the MCMC algorithm
     * @param c Cluster index
     * @param const_data Auxiliary data for the MCMC algorithm
     * @return Prior of the current state
     */


    const NumericVector & sigma = as<List>(state.sigma)[c];

    double priorg=0;
    for (int j = 0; j < sigma.length(); j++){
        priorg -= log(const_data.attrisize[j]); // densità dell'uniforme è sempre 1/numero_modalità
        priorg += logdensity_hig(sigma[j], const_data.v[j], const_data.w[j], const_data.attrisize[j]);
    }
    return priorg;
}

double split_acc_prob(const internal_state & state_split,
                     const internal_state & state,
                     const internal_state & split_launch,
                     const internal_state & merge_launch,
                     const std::vector<int> & S,
                     int i_1, int i_2,
                     const aux_data & const_data) {
    /**
     * @brief Compute the acceptance probability for the split move
     * @param state_split State after the split move
     * @param state Current state
     * @param split_launch Launch state for the split move
     * @param merge_launch Launch state for the merge move
     * @param S Vector of indices of observations in the same cluster of i_1 or i_2
     * @param i_1 Index of the first chosen observation
     * @param i_2 Index of the second chosen observation
     * @param const_data Auxiliary data for the MCMC algorithm
     * @return log-acceptance probability for the split move
     */
    
    double alpha = const_data.gamma;
    double log_ratio = 0.0;
    double log_prior = 0.0;
    double log_likelihood = 0.0;
    double log_proposal = 0.0;

    // Prior ratio
    log_prior += std::log(alpha);
    log_prior += std::lgamma(cls_elem(state_split, state_split.c_i[i_1]));
    log_prior += std::lgamma(cls_elem(state_split, state_split.c_i[i_2]));
    log_prior += priors(state_split, state_split.c_i[i_1], const_data);
    log_prior += priors(state_split, state_split.c_i[i_2], const_data);
    log_prior -= std::lgamma(cls_elem(state, state.c_i[i_1]));
    log_prior -= priors(state, state.c_i[i_1], const_data);
    
    // Likelihood ratio
    log_likelihood += loglikelihood_hamming(state_split, state_split.c_i[i_1], const_data);
    log_likelihood += loglikelihood_hamming(state_split, state_split.c_i[i_2], const_data);
    log_likelihood -= loglikelihood_hamming(state, state.c_i[i_1], const_data);
    
    // Proposal ratio
    log_proposal += logprobgs_phi(state, merge_launch, const_data, i_1);
    log_proposal -= logprobgs_phi(state_split, split_launch, const_data, i_1);
    log_proposal -= logprobgs_phi(state_split, split_launch, const_data, i_2);
    log_proposal -= logprobgs_c_i(state_split, split_launch, const_data, S, i_1, i_2);
    
    log_ratio = std::min(0.0, log_prior + log_likelihood + log_proposal);
    
    return log_ratio;
}

double merge_acc_prob(const internal_state & state_merge,
                     const internal_state & state,
                     const internal_state & split_launch,
                     const internal_state & merge_launch,
                     const std::vector<int> & S,
                     int i_1, int i_2,
                     const aux_data & const_data) {
    /**
     * @brief Compute the acceptance probability for the merge move
     * @param state_merge State after the merge move
     * @param state Current state
     * @param split_launch Launch state for the split move
     * @param merge_launch Launch state for the merge move
     * @param S Vector of indices of observations in the same cluster of i_1 or i_2
     * @param i_1 Index of the first chosen observation
     * @param i_2 Index of the second chosen observation
     * @param const_data Auxiliary data for the MCMC algorithm
     * @return log-acceptance probability for the merge move
     */    
    
    double alpha = const_data.gamma;
    double log_ratio = 0.0;
    double log_prior = 0.0;
    double log_likelihood = 0.0;
    double log_proposal = 0.0;
    
    // Prior ratio
    log_prior += std::lgamma(cls_elem(state_merge, state_merge.c_i[i_1]));
    log_prior += priors(state_merge, state_merge.c_i[i_1], const_data);
    log_prior -= std::log(alpha);
    log_prior -= std::lgamma(cls_elem(state, state.c_i[i_1]));
    log_prior -= std::lgamma(cls_elem(state, state.c_i[i_2]));
    log_prior -= priors(state, state.c_i[i_1], const_data);
    log_prior -= priors(state, state.c_i[i_2], const_data);
    
    // Likelihood ratio
    log_likelihood += loglikelihood_hamming(state_merge, state_merge.c_i[i_2], const_data);
    log_likelihood -= loglikelihood_hamming(state, state.c_i[i_1], const_data);
    log_likelihood -= loglikelihood_hamming(state, state.c_i[i_2], const_data);
    
    // Proposal ratio
    log_proposal += logprobgs_phi(state, split_launch, const_data, i_1);
    log_proposal += logprobgs_phi(state, split_launch, const_data, i_2);
    log_proposal += logprobgs_c_i(state, split_launch, const_data, S, i_1, i_2);
    log_proposal -= logprobgs_phi(state_merge, merge_launch, const_data, i_2);
    
    log_ratio = std::min(0.0, log_prior + log_likelihood + log_proposal);
    
    return log_ratio;
}

void split_and_merge(internal_state & state,
                    const aux_data & const_data,
                    int t, int r) {
    /**
     * @brief Perform the split and merge move
     * @param state Internal state of the MCMC algorithm
     * @param const_data Auxiliary data for the MCMC algorithm
     * @param t Number of restricted Gibbs scans for the split move
     * @param r Number of restricted Gibbs scans for the merge move 
     */
    
    int i_1, i_2;
    std::vector<int> S;
    
    // Select observations and build S
    select_observations(state, i_1, i_2, S);
    
    // Create launch states
    internal_state split_launch = split_launch_state(S, state, i_1, i_2, t, const_data);
    internal_state merge_launch = merge_launch_state(S, state, i_1, i_2, r, const_data);
    
    // Initialize proposed state
    internal_state state_star = {IntegerVector(), List(), List(), 0};
    double acpt_ratio = .999;
    int type = 0;
    
    if(state.c_i[i_1] == state.c_i[i_2]) {
        // Split case
        state_star = split_launch;
        split_restricted_gibbs_sampler(S, state_star, i_1, i_2, const_data);
        acpt_ratio = split_acc_prob(state_star, state, split_launch, merge_launch, S, i_1, i_2, const_data);
    } else {
        // Merge case
        state_star = merge_launch;
        update_centers(state_star, const_data, {state_star.c_i[i_2]});
        update_sigma(state_star.sigma, state_star.center, state_star.c_i, 
                    const_data, {state_star.c_i[i_2]});
        acpt_ratio = merge_acc_prob(state_star, state, split_launch, merge_launch,
                                  S, i_1, i_2, const_data);
    }

    validate_state(state_star, "split_and_merge - state_star");
    
    // Accept/reject step
    if(log(R::runif(0,1)) < acpt_ratio) {
        clean_var(state, state_star, unique_classes(state_star.c_i), const_data.attrisize);
        validate_state(state, "split_and_merge - state");
    }
}
