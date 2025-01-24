#include "updater.hpp"

Updater::Updater(const NumericMatrix & data, const IntVec & attrisize, const double gamma, const  DoubleVec & v, const  DoubleVec & w, const int & m): Sampler(data, attrisize, gamma, v, w, m){}


void Updater::update_center() {
    /**
     * @brief Update the centers of the clusters
     * @note This function updates the centers of the clusters
     */
    DEBUG_PRINT(3, "Starting update_center");

    // Update all clusters if none specified
    for (int i = 0; i < get_total_cls(); ++i) {
        const NumericMatrix& data_tmp(get_cluster_data(i));
        DEBUG_PRINT(3, "Got cluster data - row: {} col: {}", data_tmp.nrow(), data_tmp.ncol());
        const DoubleMat& prob_centers(Center_prob(data_tmp, get_sigma(i), as<NumericVector>(get_attrisize())));
        DEBUG_PRINT(3, "Calculated prob_centers - {}", prob_centers.length());
        set_center(i, sample_center_1_cluster(prob_centers));
    }
}

void Updater::update_center(const std::vector<int> & cluster_indexes) {
    /**
     * @brief Update the centers of the specified clusters
     * @param cluster_indexes Vector of cluster indexes to update
     * @note This function updates the centers of the specified clusters
     */

    DoubleMat prob_centers(get_total_cls());

    // Update only specified clusters
    for (int idx : cluster_indexes) {
        if (idx >= 0 && idx < get_total_cls()) {
            NumericMatrix data_tmp = get_cluster_data(idx);
            prob_centers = Center_prob(data_tmp, get_sigma(idx), as<NumericVector>(get_attrisize()));
            set_center(idx, sample_center_1_cluster(prob_centers)); 
        }
    }
}

void Updater::update_sigma() {
    /**
     * @brief Update the sigmas of the clusters
     * @note This function updates the sigmas of the clusters
     */
     DoubleVec new_w(get_p());
     DoubleVec new_v(get_p());
    
    // Process each cluster
    for (int i = 0; i < get_total_cls(); ++i) {   

        // Get cluster data        
        NumericMatrix cluster_data = get_cluster_data(i);

        // Count the number of data points in the cluster
        int nm = cluster_data.nrow();
        const  DoubleVec & centers_cluster = get_center(i);
        
        // Update parameters
        for (int j = 0; j < get_p(); ++j) {
            const  DoubleVec & col = cluster_data(_, j);
            double sumdelta = sum(col == centers_cluster[j]);
            new_w[j] = get_w(j) + nm - sumdelta;
            new_v[j] = get_v(j) + sumdelta;   
        }

        // Sample new sigmas
        set_sigma(i, sample_sigma_1_cluster(new_v, new_w));
    }
}

void Updater::update_sigma(const std::vector<int> & cluster_indexes) {
    /**
     * @brief Update the sigmas of the specified clusters
     * @param cluster_indexes Vector of cluster indexes to update
     * @note This function updates the sigmas of the specified clusters
     */
    
     DoubleVec new_w(get_p());
     DoubleVec new_v(get_p());
    
    // Process each cluster
    for (int i : cluster_indexes) {   

        // Get cluster data        
        NumericMatrix cluster_data = get_cluster_data(i);

        // Count the number of data points in the cluster
        int nm = cluster_data.nrow();
        const  DoubleVec & centers_cluster = get_center(i);
        
        // Update parameters
        for (int j = 0; j < get_p(); ++j) {
            const  DoubleVec & col = cluster_data(_, j);
            double sumdelta = sum(col == centers_cluster[j]);
            new_w[j] = get_w(j) + nm - sumdelta;
            new_v[j] = get_v(j) + sumdelta;   
        }

        // Sample and assign new sigmas
        set_sigma(i, sample_sigma_1_cluster(new_v, new_w));
    }
}

double Updater::compute_loglikelihood() const {
    /**
     * @brief Compute log-likelihood of the current state
     * @return Log-likelihood of the current state
     */

    double loglikelihood = 0.0;
    for (int i = 0; i < get_n(); i++) {
        // temporary references
        const  DoubleVec & center_i = get_center(get_c(i));
        const  DoubleVec & sigma_i = get_sigma(get_c(i));
        
        // Compute log-likelihood
        for (int j = 0; j < get_p(); j++) 
            loglikelihood += dhamming(get_data(i, j), center_i[j], sigma_i[j], get_attrisize(j), true);
        
    }
    return loglikelihood;
}