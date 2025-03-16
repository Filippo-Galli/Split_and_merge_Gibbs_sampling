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
    double K = norm_const2(w ,v, m); 
    return K - (v + w)*log(1+exp(-1/sigmaj)*(m-1)) - (w + 1)/sigmaj - 2*log(sigmaj);

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

    // Get cluster for chosen observation
    int c = gamma_star.c_i[choosen_idx];
    
    // Subset data for the cluster
    const NumericMatrix& data_tmp = subset_data_for_cluster(const_data.data, c, gamma_star);
    
    // Get number of observations in the cluster
    int nm = data_tmp.nrow();
    
    /*
    ------------------------------ Center probability ------------------------------
    */
    double log_center_prob = 0;

    // Prior probability of centers
    const List& prob_centers = Center_prob_pippo(data_tmp, gamma.sigma[gamma.c_i[choosen_idx]], const_data.attrisize);

    // Calculate log-probability of center parameters
    const NumericVector& centerstar = gamma_star.center[c];
    for(int j = 0; j < centerstar.length(); j++) {
        const NumericVector & z = prob_centers[j];
        log_center_prob += log(z[centerstar[j] - 1]);
    }

    /*
    ------------------------------ Sigma probability ------------------------------
    */

    double log_sigma_prob = 0;
    
    // Log-probability of sigma parameters
    for(int j = 0; j < const_data.attrisize.length(); j++) {
        const NumericVector& col = data_tmp(_, j);
        double sumdelta = sum(col == as<NumericVector>(gamma_star.center[c])[j]);
        double new_v = const_data.v[j] + sumdelta;
        double new_w = const_data.w[j] + nm - sumdelta;
        
        log_sigma_prob += logdensity_hig(as<NumericVector>(gamma_star.sigma[c])[j], new_v, new_w, const_data.attrisize[j]);
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

    // Extract cluster of the first observation
    int c_i_1 = gamma.c_i[i_1];
    int c_i_2 = gamma.c_i[i_2];

    // support variable
    NumericVector probs(2);
    int cls;
    int n_s_cls;

    for (int s : S) {
        // extract datum at s
        const NumericVector& y_s = const_data.data(s, _);

        // evaluate probabilities
        for (int k = 0; k < 2; k++) {
            double Hamming = 0;
            
            if(k == 0)
                cls = c_i_1;
            else
                cls = c_i_2;

            for (int j = 0; j < y_s.length(); j++) {
                Hamming += dhamming_pippo(y_s[j], as<NumericVector>(gamma_star.center[cls])[j], 
                            as<NumericVector>(gamma_star.sigma[cls])[j], const_data.attrisize[j]);
            }

            // Count instances in the cluster excluding the current point s
            n_s_cls = sum(gamma.c_i == cls) - (gamma.c_i[s] == cls); //aux_state.c_i[s] == cls è false questo è uguale a 0
            
            probs[k] = log(n_s_cls) + Hamming;
        }

        // Normalize probabilities
        probs = exp(probs - max(probs));
        probs = probs / sum(probs);

        int currrent_c = gamma_star.c_i[s] == c_i_1 ? 0 : 1;

        //  logprob of be assigned to the new state
        logpgs += log(probs[currrent_c]);
    }
    
    return logpgs;
}

void split_restricted_gibbs_sampler(const std::vector<int> & S, internal_state & state, int i_1, int i_2, const aux_data & const_data, int t) {
    /**
     * @brief Restricted Gibbs Sampler between the cluster of two observations
     * @param state Internal state of the MCMC algorithm
     * @param i_1 Index of the first chosen observation
     * @param i_2 Index of the second chosen observation
     * @param S Vector of indices of observations in the same cluster of i_1 or i_2
     * @param const_data Auxiliary data for the MCMC algorithm
     */

    // Extract cluster parameters
    int c_i_1 = state.c_i[i_1];
    int c_i_2 = state.c_i[i_2];

    // support variables
    NumericVector probs(2);
    int cls;
    int n_s_cls;

    for(int iter = 0; iter < t; ++iter) {
        /*
        * ------------------------ Allocation Sampling ------------------------
        */
        for (int s : S) {
            // extract datum at s
            const NumericVector & y_s = const_data.data(s, _);

            // evaluate probabilities
            for (int k = 0; k < 2; k++) {
                double Hamming = 0;

                if(k == 0)
                    cls = c_i_1;
                else
                    cls = c_i_2;

                for (int j = 0; j < y_s.length(); j++) {
                    Hamming += dhamming_pippo(y_s[j], as<NumericVector>(state.center[cls])[j], 
                                as<NumericVector>(state.sigma[cls])[j], const_data.attrisize[j]);
                }

                // Count instances in the cluster excluding the current point s
                n_s_cls = sum(state.c_i == cls) - (state.c_i[s] == cls);  

                probs[k] =  log(n_s_cls) + Hamming;
            }

            // Normalize probabilities
            probs = exp(probs - max(probs));
            probs = probs / sum(probs);
        
            // Sample new cluster assignment between the two clusters of i_1 and i_2
            state.c_i[s] = sample(IntegerVector::create(c_i_1, c_i_2), 1, true, probs)[0];
        }

        /*
        * ------------------------ Update Centers ------------------------
        */

        update_phi(state, const_data, {c_i_1, c_i_2});
        validate_state(state, "split_restricted_gibbs_sampler - iteration " + std::to_string(iter));
    }
}

void select_observations_deterministic(const internal_state & state, int & i_1, int & i_2, std::vector<int> & S) {
    /**
     * @brief Select two observations and populate S with indexes in selected clusters
     * @param state Internal state of the MCMC algorithm
     * @param i_1 Index of the first chosen observation
     * @param i_2 Index of the second chosen observation
     * @param S Vector of indices of observations in the same cluster of i_1 or i_2
     */

        // indexes to choose from excluding i_1
    IntegerVector indexes = seq(0, state.c_i.size() - 1);
    // Sample i_2 observations
    i_2 = sample(indexes, 1, false)[0];

    // avoid to sample the same observation
    while(i_2 == i_1) {
        i_2 = sample(indexes, 1, false)[0];
    }

    // Reserve space for S  to avoid multiple reallocations
    int n_i_1 = sum(state.c_i == state.c_i[i_1]) - 1; // number of observations in the cluster of i_1
    int n_i_2 = sum(state.c_i == state.c_i[i_2]) - 1; // number of observations in the cluster of i_2
    S.reserve(n_i_1 + n_i_2);

    // Populate S with indexes in selected clusters
    for(int i = 0; i < state.c_i.length(); ++i) {
        if(i == i_1 || i == i_2) 
            continue;
        if(state.c_i[i] == state.c_i[i_1] || state.c_i[i] == state.c_i[i_2]) {
            S.emplace_back(i);
        }
    }
}

void select_observations_random(const internal_state & state, int & i_1, int & i_2, std::vector<int> & S) {
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
            S.emplace_back(i);
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
    internal_state state_launch_split(state);
    //Rcpp::Rcout << "state_launch_split before" << std::endl;
    //print_internal_state(state_launch_split, 2);

    // Initialize centers and sigmas based on cluster equality
    if (state.c_i[i_1] == state.c_i[i_2]) {
        // Assign new cluster to the first observation
        state_launch_split.c_i[i_1] = state.total_cls; 
        state_launch_split.center.push_back(sample_center_1_cluster(const_data.attrisize));
        state_launch_split.sigma.push_back(sample_sigma_1_cluster(const_data.attrisize, const_data.v, const_data.w));
        state_launch_split.total_cls++; // increase the number of clusters
    }
    else{
        // sample new parameters for the clusters of i_1
        state_launch_split.center[state.c_i[i_1]] = sample_center_1_cluster(const_data.attrisize);
        state_launch_split.sigma[state.c_i[i_1]] = sample_sigma_1_cluster(const_data.attrisize, const_data.v, const_data.w);
    }
    
    
    // Sample new parameters for the clusters of i_2
    state_launch_split.center[state.c_i[i_2]] = sample_center_1_cluster(const_data.attrisize);
    state_launch_split.sigma[state.c_i[i_2]] = sample_sigma_1_cluster(const_data.attrisize, const_data.v, const_data.w);

    //Rcpp::Rcout << "state_launch_split after" << std::endl;
    //print_internal_state(state_launch_split, 2);

    // Random allocation of S between clusters
    state_launch_split.c_i[S_indexes] = sample(IntegerVector::create(state_launch_split.c_i[i_1], state_launch_split.c_i[i_2]), S_indexes.length(), true);

    // print state_launch_split in S_indexes
    // Rcpp::Rcout << std::endl << "state_launch_split after random allocation before" << std::endl;
    // for(int i = 0; i < S_indexes.length(); ++i) {
    //     Rcpp::Rcout << state_launch_split.c_i[S_indexes[i]] << " ";
    // }

    // Intermediate Gibbs sampling
    split_restricted_gibbs_sampler(S, state_launch_split, i_1, i_2, const_data, t);

    // Rcpp::Rcout << std::endl << "state_launch_split after random allocation after" << std::endl;
    // for(int i = 0; i < S_indexes.length(); ++i) {
    //     Rcpp::Rcout << state_launch_split.c_i[S_indexes[i]] << " ";
    // }
    // Rcpp::Rcout << std::endl;

    validate_state(state_launch_split, "split_launch_state");

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
    for(int iter = 0; iter < r; ++iter) 
        update_phi(state_launch_merge, const_data, {state_launch_merge.c_i[i_2]});
    

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

double priors(const internal_state & state, int c, const aux_data & const_data){
    /**
     * @brief Compute the prior of the current state
     * @param state Internal state of the MCMC algorithm
     * @param c Cluster index
     * @param const_data Auxiliary data for the MCMC algorithm
     * @return log probability of from the prior of the current state
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
    log_prior += std::lgamma(static_cast<double>(sum(state_split.c_i == state_split.c_i[i_1])));
    log_prior += std::lgamma(static_cast<double>(sum(state_split.c_i == state_split.c_i[i_2])));
    log_prior += priors(state_split, state_split.c_i[i_1], const_data);
    log_prior += priors(state_split, state_split.c_i[i_2], const_data);
    log_prior -= std::lgamma(static_cast<double>(sum(state_split.c_i == state_split.c_i[i_1])));
    //Rcpp::Rcout << std::endl << "log_prior pre priors call: " << log_prior << std::endl;
    log_prior -= priors(state, state.c_i[i_1], const_data);
    //Rcpp::Rcout << std::endl << "log_prior pre post call: " << log_prior << std::endl;
    
    // Likelihood ratio
    log_likelihood += loglikelihood_hamming(state_split, state_split.c_i[i_1], const_data);
    log_likelihood += loglikelihood_hamming(state_split, state_split.c_i[i_2], const_data);
    log_likelihood -= loglikelihood_hamming(state, state.c_i[i_1], const_data);
    //Rcpp::Rcout << std::endl << "log_likelihood: " << log_likelihood << std::endl;
    
    // Proposal ratio
    log_proposal += logprobgs_phi(state, merge_launch, const_data, i_1);
    log_proposal -= logprobgs_phi(state_split, split_launch, const_data, i_1);
    log_proposal -= logprobgs_phi(state_split, split_launch, const_data, i_2);
    log_proposal -= logprobgs_c_i(state_split, split_launch, const_data, S, i_1, i_2);
    //Rcpp::Rcout << std::endl << "log_proposal: " << log_proposal << std::endl;
    
    //log_ratio = std::min(0.0, log_prior + log_likelihood + log_proposal);
    log_ratio = log_prior + log_likelihood + log_proposal;
    
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
    log_prior += std::lgamma(static_cast<double>(sum(state_merge.c_i == state_merge.c_i[i_1])));
    log_prior += priors(state_merge, state_merge.c_i[i_1], const_data);
    log_prior -= std::log(alpha);
    log_prior -= std::lgamma(static_cast<double>(sum(state.c_i == state.c_i[i_1])));
    log_prior -= std::lgamma(static_cast<double>(sum(state.c_i == state.c_i[i_2])));
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

    
    //log_ratio = std::min(0.0, log_prior + log_likelihood + log_proposal);
    log_ratio = log_prior + log_likelihood + log_proposal;

    return log_ratio;
}

int split_and_merge(internal_state & state,
                    const aux_data & const_data,
                    int t, int r, int & idx_1_sm, 
                    IntegerVector & gt) {
    /**
     * @brief Perform the split and merge move
     * @param state Internal state of the MCMC algorithm
     * @param const_data Auxiliary data for the MCMC algorithm
     * @param t Number of restricted Gibbs scans for the split move
     * @param r Number of restricted Gibbs scans for the merge move 
     * @param idx_1_sm Index of the first chosen observation
     */
    int i_1 = idx_1_sm;
    int i_2;
    std::vector<int> S;
    
    // Select observations and build S
    select_observations_deterministic(state, i_1, i_2, S);
    
    // Create launch states
    internal_state split_launch = split_launch_state(S, state, i_1, i_2, t, const_data);
    internal_state merge_launch = merge_launch_state(S, state, i_1, i_2, r, const_data);
    
    // Initialize proposed state
    internal_state state_star = {IntegerVector(), List(), List(), 0};
    double acpt_ratio = .999;
    
    bool split = false;
    if(state.c_i[i_1] == state.c_i[i_2]) {
        split = true;
        // Split case        
        state_star = split_launch;
        split_restricted_gibbs_sampler(S, state_star, i_1, i_2, const_data);
        acpt_ratio = split_acc_prob(state_star, state, split_launch, merge_launch, S, i_1, i_2, const_data);
    } else {
        // Merge case        
        state_star = merge_launch;
        update_phi(state_star, const_data, {state_star.c_i[i_2]});
        acpt_ratio = merge_acc_prob(state_star, state, split_launch, merge_launch, S, i_1, i_2, const_data);
    }

    validate_state(state_star, "split_and_merge - state_star");
    
    // Accept/reject step
    if(log(R::runif(0,1)) < acpt_ratio) {
        // if((sum(state.c_i == state.c_i[i_1]) == 1 || sum(state.c_i == state.c_i[i_2]) == 1)) {
        //     Rcpp::Rcout << "ACCETTATO - split: " << split << std::endl;
        //     Rcpp::Rcout << std::endl << "[DEBUG] acpt_ratio: " << acpt_ratio << std::endl;
        //     Rcpp::Rcout << std::endl << "[DEBUG] i_1: " << i_1  << " c_1: " << state.c_i[i_1] << " i_2: " << i_2 << " c_2: " << state.c_i[i_2] << std::endl;
        //     Rcpp::Rcout << std::endl << "[DEBUG] i_1: " << i_1  << " state_star.c_1: " << state_star.c_i[i_1] << " i_2: " << i_2 << " state_star.c_2: " << state_star.c_i[i_2] << std::endl;

        //     Rcpp::Rcout << "GT[c1]: " << std::endl;
        //     for(int i = 0; i < S.size(); i++) {
        //         if(state_star.c_i[S[i]] == state_star.c_i[i_1]) {
        //             Rcpp::Rcout << gt[i] << " ";
        //         }
        //     }
        //     Rcpp::Rcout << " " << gt[i_1] << " ";
        //     Rcpp::Rcout << std::endl;
        //     // Find indices manually
        //     Rcpp::Rcout << "GT[c2]: " << std::endl;
        //     for(int i = 0; i < S.size(); i++) {
        //         if(state_star.c_i[S[i]] == state_star.c_i[i_2]) {
        //             Rcpp::Rcout << gt[i] << " ";
        //         }
        //     }
        //     Rcpp::Rcout << " " << gt[i_2] << " ";
        //     Rcpp::Rcout << std::endl;

        //     // Rcpp::Rcout << std::endl << "[DEBUG] merge_launch: " << std::endl;
        //     // print_internal_state(merge_launch);
        //     // Rcpp::Rcout << std::endl << "[DEBUG] split_launch: " << std::endl;
        //     // print_internal_state(split_launch);

        //     Rcpp::Rcout << std::endl << "[DEBUG] state_star: " << std::endl;
        //     print_internal_state(state_star);
        //     Rcpp::Rcout << "[DEBUG] state: " << std::endl;
        //     print_internal_state(state);

        // }

        clean_var(state, state_star, unique_classes(state_star.c_i), const_data.attrisize);
        validate_state(state, "split_and_merge - state");

        return 1;
    }
    else{
        // if((sum(state.c_i == state.c_i[i_1]) == 1 || sum(state.c_i == state.c_i[i_2]) == 1)) {
        //     Rcpp::Rcout << "NONACC - split: " << split << std::endl;
        //     Rcpp::Rcout << std::endl << "[DEBUG] acpt_ratio: " << acpt_ratio << std::endl;
        //     Rcpp::Rcout << std::endl << "[DEBUG] i_1: " << i_1  << " state.c_1: " << state.c_i[i_1] << " i_2: " << i_2 << " state.c_2: " << state.c_i[i_2] << std::endl;
        //     Rcpp::Rcout << std::endl << "[DEBUG] i_1: " << i_1  << " state_star.c_1: " << state_star.c_i[i_1] << " i_2: " << i_2 << " state_star.c_2: " << state_star.c_i[i_2] << std::endl;
        //     // Find indices manually
        //     Rcpp::Rcout << "GT[c1]: " << std::endl;
        //     for(int i = 0; i < S.size(); i++) {
        //         if(state_star.c_i[S[i]] == state_star.c_i[i_1]) {
        //             Rcpp::Rcout << gt[i] << " ";
        //         }
        //     }
        //     Rcpp::Rcout << " " << gt[i_1] << " ";
        //     Rcpp::Rcout << std::endl;
        //     // Find indices manually
        //     Rcpp::Rcout << "GT[c2]: " << std::endl;
        //     for(int i = 0; i < S.size(); i++) {
        //         if(state_star.c_i[S[i]] == state_star.c_i[i_2]) {
        //             Rcpp::Rcout << gt[i] << " ";
        //         }
        //     }
        //     Rcpp::Rcout << " " << gt[i_2] << " ";
        //     Rcpp::Rcout << std::endl;
        //     // Rcpp::Rcout << std::endl << "[DEBUG] merge_launch: " << std::endl;
        //     // print_internal_state(merge_launch);
        //     // Rcpp::Rcout << std::endl << "[DEBUG] split_launch: " << std::endl;
        //     // print_internal_state(split_launch);

        //     Rcpp::Rcout << std::endl << "[DEBUG] state_star: " << std::endl;
        //     print_internal_state(state_star);
        //     Rcpp::Rcout << "[DEBUG] state: " << std::endl;
        //     print_internal_state(state);

        // }
    }
    return 0;
}
