#include "sampler.hpp"

Sampler::Sampler(const NumericMatrix & data, const IntVec & attrisize, const double gamma, const  DoubleVec & v, const  DoubleVec & w, const int & m):Data(data, attrisize, gamma, v, w, m){}

void Sampler::sample_c_1_obs(const int & index_i) {
    /**
     * @brief Sample a cluster assignment for observation i
     * @param index_i Index of observation i
     * @return Sampled cluster assignment
     * @note This function samples a cluster assignment for a single observation
     */

    //DEBUG_PRINT(3, "Starting sample_c_1_obs");
    // Get data point
    const  DoubleVec & y_i = get_data_row(index_i);

    //DEBUG_PRINT(3, "Got data point");

    // Get unique classes
    const  DoubleVec& unique_classes_without_i = get_unique_clusters_without_idx(index_i);
    const  DoubleVec& unique_classes = get_unique_clusters();

    //DEBUG_PRINT(3, "Got unique classes");

    // Parameters 
    const int k_minus = unique_classes_without_i.length();
    int m_temp = get_m(); // number to latent clusters to sample
    
    // allocation, center and sigma without considering the i-th observation
    IntVec c_temp(get_c());
    DoubleMat center_temp(get_center());
    DoubleMat sigma_temp(get_sigma());

    clean_var(c_temp, center_temp, sigma_temp, get_c(), get_center(), get_sigma(), unique_classes_without_i);

    // if index_i is the only element in its cluster use its own center and sigma as first latent cluster center and sigma
    if(k_minus != unique_classes.length()) {
        center_temp.push_back(get_center(get_c(index_i)));
        sigma_temp.push_back(get_sigma(get_c(index_i)));
        m_temp -= 1; // -1 since we already added the center and sigma of the i-th observation
    }

    // Sample the latent cluster parameters
    size_t temp_length = center_temp.length();
    for(int i = 0; i < m_temp; ++i){
        center_temp.push_back(sample_center_1_cluster());
        sigma_temp.push_back(sample_sigma_1_cluster());
    }
    temp_length += m_temp; // update the length of the center and sigma DoubleMats <-> number of unique clusters + latent clusters

    //DEBUG_PRINT(3, "Sampled latent cluster parameters");

    // Calculate the probabilities of allocation 
     DoubleVec probs(temp_length);

    //DEBUG_PRINT(3, "Calculated probabilities of allocation");

    // Calculate the probabilities of the existing clusters
    for (int k = 0; k < k_minus; k++) {
        double log_likelihood = 0.0; // is this correct to initialize with 0.0?

        const  DoubleVec & sigma_k(sigma_temp[k]);
        const  DoubleVec & center_k(center_temp[k]);

        // Calculate likelihood
        for (int j = 0; j < y_i.length(); j++) {
            log_likelihood += dhamming(y_i[j], center_k[j], sigma_k[j], get_attrisize(j), true);
        }

        // Count instances excluding current point
        int n_i_z = get_cluster_size(unique_classes_without_i[k]);
        // if the current point is in the cluster, decrease the count
        if(get_c(index_i) == unique_classes_without_i[k]){
            n_i_z -= 1;
        }

        // Set probability
        probs[k] = (n_i_z == 0) ? 0.0 : n_i_z * std::exp(log_likelihood); // should we use all in log scale? 
    }

    //DEBUG_PRINT(3, "Calculated probabilities of existing clusters");

    // Calculate the probabilities of the latent clusters
    for(size_t k = k_minus; k < temp_length; k++){
        double log_likelihood = 0.0;
        
        const  DoubleVec & sigma_k = sigma_temp[k];
        const  DoubleVec & center_k = center_temp[k];

        // Calculate likelihood
        for (int j = 0; j < y_i.length(); j++) {
            log_likelihood += dhamming(y_i[j], center_k[j], sigma_k[j], get_attrisize(j), true);
        }

        probs[k] = get_gamma()/get_m() * std::exp(log_likelihood); // should we use all in log scale? 
    }

    //DEBUG_PRINT(3, "Calculated probabilities of latent clusters");

    // Normalize the probabilities
    probs = probs / max(probs); // taken from gibbs_utility.cpp:34
    probs = probs / sum(probs);

    // Sample the cluster assignment
     DoubleVec cls(temp_length);
    for(size_t i = 0; i < temp_length; i++) {
        cls[i] = i;
    }
    c_temp[index_i] = sample(cls, 1, true, probs)[0];

    //DEBUG_PRINT(3, "Sampled cluster assignment");

    // Remove the latent clusters if they are not used
    clean_var(c_temp, center_temp, sigma_temp, as<NumericVector>(unique(c_temp).sort()));
    
    //DEBUG_PRINT(3, "Cleaned variables");

    // Update the state variables
    set_c(c_temp);
    set_center(center_temp);
    set_sigma(sigma_temp);

    //DEBUG_PRINT(3, "Updated state variables"); 
}

void Sampler::sample_c() {
    /**
     * @brief Sample the cluster assignments for all observations
     * @note This function samples the cluster assignments for all observations
     */
    //DEBUG_PRINT(2, "Starting sample_c");
    for(int i = 0; i < get_n(); ++i) {
        sample_c_1_obs(i);
    }
}

NumericVector Sampler::sample_center_1_cluster(const DoubleMat & probs) const {
    /**
     * @brief Sample center for a single cluster
     * @param probs DoubleMat of probabilities for each attribute
     * @return  DoubleVec containing the sampled center
     */
    DEBUG_PRINT(4, "Starting sample_center_1_cluster - p: {}", get_p());
     DoubleVec center(get_p());
    for (int j = 0; j < get_p(); j++) {
        const IntVec & attribute_vec(seq_len(get_attrisize(j)));
        center[j] = sample(attribute_vec, 1, true, as<NumericVector>(probs[j]))[0];
    } 
    return center;
}

NumericVector Sampler::sample_center_1_cluster() const {
    /**
     * @brief Sample center for a single cluster
     * @return  DoubleVec containing the sampled center
     */
     DoubleVec center(get_p());
    for (int j = 0; j < get_p(); j++) 
        center[j] = sample(get_attrisize(j), 1, true)[0];

    return center;
}

void Sampler::sample_center() {
    /**
     * @brief Sample the centers for all clusters
     * @note This function samples the centers for all clusters
     */

    for(int k = 0; k < get_total_cls(); ++k) {
         DoubleVec center = sample_center_1_cluster();
        set_center(k, center);
    }
}

NumericVector Sampler::sample_sigma_1_cluster() const {
    /**
     * @brief Sample sigma for a single cluster
     * @return  DoubleVec containing the sampled sigma
     */
     DoubleVec sigma(get_p());
    for (int j = 0; j < get_p(); ++j) {
        sigma[j] = rhyper_sig(1, get_w(j), get_v(j), get_attrisize(j))[0];
    } 

    return sigma;
}

NumericVector Sampler::sample_sigma_1_cluster(const  DoubleVec& v, const  DoubleVec& w) const {
    /**
     * @brief Sample sigma for a single cluster
     * @param v v parameter
     * @param w w parameter
     * @return  DoubleVec containing the sampled sigma
     */
     DoubleVec sigma(get_p());
    for (int j = 0; j < get_p(); ++j) {
        sigma[j] = rhyper_sig(1, get_w(j), get_v(j), get_attrisize(j))[0];
    } 

    return sigma;
}

void Sampler::sample_sigma() {
    /**
     * @brief Sample the sigmas for all clusters
     * @note This function samples the sigmas for all clusters
     */
    DoubleMat temp_sigma(get_total_cls());
    for(int k = 0; k < get_total_cls(); ++k) {
         DoubleVec sigma = sample_sigma_1_cluster();
        temp_sigma[k] = sigma;
    }
    set_sigma(temp_sigma);
}