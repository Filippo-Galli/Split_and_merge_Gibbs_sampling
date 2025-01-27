#include "split_merge.hpp"

void SplitMerge::select_observation() {
    /**
     * @brief Select two observation from the data and create vector S
     */

    auto temp = sample(get_data_rows(), 2, replace = FALSE);
    // Select an observation
    i1 = temp[0];
    i2 = temp[1];

    // retrive the cluster indices for one observations and add it to S
    IntVec cls_i1 = get_cluster_indices(get_c(i1));
    cls_i1.erase(std::remove(cls_i1.begin(), cls_i1.end(), i1), cls_i1.end());
    S(cls_i1);
    
    // If the two observations are in different clusters, add the second cluster to S
    if(get_c()[i1] != get_c()[i2]) {
        IntVec cls_i2 = get_cluster_indices(get_c(i2));
        cls_i2.erase(std::remove(cls_i2.begin(), cls_i2.end(), i2), cls_i2.end());
        S.push_back(cls_i2);
    }
    else{
        // If the two observations are in the same cluster, remove the second observation from S
        S.erase(std::remove(S.begin(), S.end(), i2), S.end());
    }
}

void SplitMerge::split_restricted_gibbs_sampler() {
    /**
     * @brief Perform a restricted Gibbs sampling for the split move
     */

    // Extract cluster of the first observation
    int c_i1 = get_c(i1);
    const DoubleVec& center_i1 = split_launch_state.get_center(c_i1);
    const DoubleVec& sigma_i1 = split_launch_state.get_sigma(c_i1);

    // Extract cluster of the second observation
    int c_i2 = get_c(i2);
    const DoubleVec& center_i2 = split_launch_state.get_center(c_i2);
    const DoubleVec& sigma_i2 = split_launch_state.get_sigma(c_i2);

    NumericVector probs(2);
    NumericVector center_k(split_launch_state.get_p());
    NumericVector sigma_k(split_launch_state.get_p());
    int cls_k;
    // for each observation in S
    for (auto && s : S){
        const NumericVector& x_s = get_data(s);

        for(int k = 0; k < 2; ++k){
            if(k == 0){
                center_k = center_i1;
                sigma_k = sigma_i1;
                cls_k = c_i1;
            }
            else{
                center_k = center_i2;
                sigma_k = sigma_i2;
                cls_k = c_i2;
            }

            double Hamming = 0;
            for (int j = 0; j < y_s.length(); j++) {
                Hamming += dhamming(x_s[j], center_k[j], sigma_k[j], get_attrisize(j), true);
            }

            int ns_k = split_launch_state.get_cluster_size(cls_k) - 1; // -1 since we are from both discard the observations i1 and i2
            probs[k] = log(ns_k) + Hamming;
        }

        probs = exp(probs - max(probs));
        probs = probs / sum(probs);

        split_launch_state.set_c(s, sample(IntegerVector::create(c_i1, c_i2), 1, true, probs)[0]);
    }

    split_launch_state.update_center({c_i1, c_i2});
    split_launch_state.update_sigma({c_i1, c_i2});
}

void SplitMerge::split_restricted_gibbs_sampler(Updater & state){
    /**
     * @brief Perform a restricted Gibbs sampling for the split move
     * @param state_star Internal state of the MCMC algorithm
     */

        // Extract cluster of the first observation
    int c_i1 = get_c(i1);
    const DoubleVec& center_i1 = state.get_center(c_i1);
    const DoubleVec& sigma_i1 = state.get_sigma(c_i1);

    // Extract cluster of the second observation
    int c_i2 = get_c(i2);
    const DoubleVec& center_i2 = state.get_center(c_i2);
    const DoubleVec& sigma_i2 = state.get_sigma(c_i2);

    NumericVector probs(2);
    NumericVector center_k(state.get_p());
    NumericVector sigma_k(state.get_p());
    int cls_k;
    // for each observation in S
    for (auto && s : S){
        const NumericVector& x_s = get_data(s);

        for(int k = 0; k < 2; ++k){
            if(k == 0){
                center_k = center_i1;
                sigma_k = sigma_i1;
                cls_k = c_i1;
            }
            else{
                center_k = center_i2;
                sigma_k = sigma_i2;
                cls_k = c_i2;
            }

            double Hamming = 0;
            for (int j = 0; j < y_s.length(); j++) {
                Hamming += dhamming(x_s[j], center_k[j], sigma_k[j], get_attrisize(j), true);
            }

            int ns_k = state.get_cluster_indices(cls_k).length() - 1;
            probs[k] = log(ns_k) + Hamming;
        }

        probs = exp(probs - max(probs));
        probs = probs / sum(probs);

        state.set_c(s, sample(IntegerVector::create(c_i1, c_i2), 1, true, probs)[0]);
    }

    // update center and sigma of the two clusters - [ERRORE] l'update deve essere fatto su split_launch_state!!!!!!!
    state.update_center({c_i1, c_i2});
    state.update_sigma({c_i1, c_i2});
}

void SplitMerge::split_launch_state_creation(const int t) {
    /**
     * @brief Generate the launch state for the split move
     * @param t Number of restricted Gibbs scans
     * @return Internal state of the MCMC algorithm
     */

    // Initialize split launch state
    split_launch_state(get_data(), get_attrisize(), get_gamma(), get_v(), get_w(), get_m());
    split_launch_state.set_c(get_c());
    split_launch_state.set_center(get_center());
    split_launch_state.set_sigma(get_sigma());
    split_launch_state.set_total_cls(get_total_cls());

    // If they are in the same cluster, split the cluster
    if(get_c(i1) == get_c(i2)) {
        split_launch_state.set_c(i2, get_total_cls());
        split_launch_state.set_center(i2, sample_center_1_cluster());
        split_launch_state.set_sigma(i2, sample_sigma_1_cluster());
        split_launch_state.set_total_cls(get_total_cls() + 1);
    }

    auto temp = sample(IntegerVector::create(split_launch_state.c_i[i1], split_launch_state.c_i[i2]), S.length(), true)

    // Randomly allocate S to the clusters of i1 and i2
    for(int i = 0; i < S.length(); ++i) {
        split_launch_state.set_c(S[i], temp[i]);
    }

    // Clean the variables
    split_launch_state.clean_var();

    // Intermediate Gibbs sampling
    for(int iter = 0; iter < t; ++iter) {
        split_restricted_gibbs_sampler();
    }
}

void SplitMerge::merge_launch_state_creation(const int r) {
    /**
     * @brief Generate the launch state for the merge move
     * @param r Number of restricted Gibbs scans
     * @return Internal state of the MCMC algorithm
     */

    // Initialize merge launch state
    merge_launch_state(get_data(), get_attrisize(), get_gamma(), get_v(), get_w(), get_m());
    merge_launch_state.set_c(get_c());
    merge_launch_state.set_center(get_center());
    merge_launch_state.set_sigma(get_sigma());
    merge_launch_state.set_total_cls(get_total_cls());

    // If they are in different clusters, merge the clusters
    if(get_c(i1) != get_c(i2)) {
        merge_launch_state.set_c(i2, get_c(i1));
        // remove the first cluster
        merge_launch_state.set_total_cls(get_total_cls() - 1);
    }

    // Intermediate Gibbs sampling
    for(int iter = 0; iter < r; ++iter) {
        merge_launch_state.update_center({get_c(i2)});
        merge_launch_state.update_sigma({get_c(i2)});
    }
}

double SplitMerge::priors(const & Updater state, const int & c) const {
    /**
     * @brief Compute the prior of the current state
     * @param state Internal state of the MCMC algorithm
     * @param c Cluster index
     * @param const_data Auxiliary data for the MCMC algorithm
     * @return Prior of the current state
     */

    const NumericVector& sigma = state.get_sigma(c);

    double priorg=0;
    for (int j = 0; j < get_p(); j++){
        priorg -= log(get_attrisize(j)); // densità dell'uniforme è sempre 1/numero_modalità
        priorg += logdensity_hig(sigma[j], get_v(j), get_w(j), get_attrisize(j));
    }
    return priorg;
}

double SplitMerge::split_acc_prob() const {
    /**
     * @brief Compute the acceptance probability for the split move
     * @return log-acceptance probability for the split move
     */
    
    double alpha = get_gamma();
    double log_ratio = 0.0;
    double log_prior = 0.0;
    double log_likelihood = 0.0;
    double log_proposal = 0.0;
    double logp = 0.0;

    // Prior ratio
    log_prior += std::log(alpha);
    log_prior += std::lgamma(state_star.get_cluster_size(get_c(i1)));
    log_prior += std::lgamma(state_star.get_cluster_size(get_c(i2)));
    log_prior += priors(state_star, state_star.get_c(i1));
    log_prior += priors(state_star, state_star.get_c(i2));

    log_prior -= std::lgamma(split_launch_state.get_cluster_size(split_launch_state.get_c(i1)));
    log_prior -= priors(split_launch_state, split_launch_state.get_c(i1));
    DEBUG_PRINT(1, "SPLIT - prior Logratio: {}", log_prior);
    
    // Likelihood ratio -DA SISTEMARE
    log_likelihood += loglikelihood_hamming(state_split, state_split.c_i[i_1], const_data);
    DEBUG_PRINT(2, "SPLIT - Numeratore loglikelihood: {}", loglikelihood_hamming(state_split, state_split.c_i[i_1], const_data));
    log_likelihood += loglikelihood_hamming(state_split, state_split.c_i[i_2], const_data);
    DEBUG_PRINT(2, "SPLIT - Numeratore loglikelihood: {}", loglikelihood_hamming(state_split, state_split.c_i[i_2], const_data));
    log_likelihood -= loglikelihood_hamming(state, state.c_i[i_1], const_data);
    DEBUG_PRINT(2, "SPLIT - Denominatore loglikelihood: {}", loglikelihood_hamming(state, state.c_i[i_1], const_data));
    DEBUG_PRINT(1, "SPLIT - likelihood Logratio: {}", log_likelihood);
    
    // Proposal ratio
    log_proposal += logprobgs_phi(state, merge_launch, const_data, i_1);
    log_proposal -= logprobgs_phi(state_split, split_launch, const_data, i_1);
    log_proposal -= logprobgs_phi(state_split, split_launch, const_data, i_2);
    log_proposal -= logprobgs_c_i(state_split, split_launch, const_data, S, i_1, i_2);
    DEBUG_PRINT(1, "SPLIT - proposal Logratio: {}", log_proposal);

    log_ratio = std::min(0.0, log_prior + log_likelihood + log_proposal);
    DEBUG_PRINT(0, "SPLIT - logAcceptance ratio: {}", log_ratio);
    
    return log_ratio;
}



void SplitMerge::split_merge_step(const int & t, const int & r) {
    /**
     * @brief Perform the split and merge move
     */

    // Select observation and create S
    select_observation();

    // Split launch state creation
    split_launch_state_creation(t);

    // Merge launch state creation
    merge_launch_state_creation(r);

    // Acceptance probability
    state_star(get_data(), get_attrisize(), get_gamma(), get_v(), get_w(), get_m());

    double acceptance_ratio = 0;
    if(get_c(i1) == get_c(i2)) {
        // Load data into state_star
        state_star.set_c(split_launch_state.get_c());
        state_star.set_center(split_launch_state.get_center());
        state_star.set_sigma(split_launch_state.get_sigma());
        state_star.set_total_cls(split_launch_state.get_total_cls());

        // Last restricted Gibbs sampling
        split_restricted_gibbs_sampler(state_star);

        // Calculate acceptance ratio
        acceptance_ratio = ;
    }
    else {
        // Load data into state_star
        state_star.set_c(merge_launch_state.get_c());
        state_star.set_center(merge_launch_state.get_center());
        state_star.set_sigma(merge_launch_state.get_sigma());
        state_star.set_total_cls(merge_launch_state.get_total_cls());

        // Calculate acceptance ratio
        acceptance_ratio = 1;
    }

}