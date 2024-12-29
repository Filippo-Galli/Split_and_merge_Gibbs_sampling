/**
 * @file neal_sampler.cpp
 * @brief Implementation of Markov Chain Monte Carlo clustering algorithm based on Neal algorithm 8 and Argiento paper 2022 using Rcpp
 * @details This file contains the implementation of a MCMC-based clustering algorithm
 *          specifically designed for categorical data using Hamming distance.
 */

#include <set>
#include <random>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <iomanip>

#include <Rcpp.h>
#include <RcppGSL.h>  // Add this explicit RcppGSL include
#include <Rinternals.h>
#include <gsl/gsl_sf_hyperg.h>


#include <gibbs_utility.cpp>
#include <hyperg.cpp>
using namespace Rcpp;

struct internal_state {
    /**
     * @brief Internal state of the MCMC algorithm
     * @details This struct contains the internal state of the MCMC algorithm
     *          including the current cluster assignments, cluster centers, and cluster dispersions
     */
    IntegerVector c_i;
    List center;
    List sigma;
    int total_cls = 0;
};

struct aux_data{
    /**
     * @brief Auxiliary data for the MCMC algorithm
     * @details This struct contains auxiliary data for the MCMC algorithm
     *          including the input data matrix, attribute sizes, and hypergeometric parameters
     */
    NumericMatrix data;
    int n;
    IntegerVector attrisize;
    // Below are the hypergeometric parameters
    double gamma;
    NumericVector v;
    NumericVector w;
};

void print_internal_state(const internal_state& state, int interest = -1) {
    /**
     * @brief Print internal state of the MCMC algorithm
     * @note Interest values <-> -1: print all, 1: print cluster assignments, 2: print cluster centers, 3: print cluster dispersions
     * @param state Internal state of the MCMC algorithm
     * @param interest Index of what print (default: -1).
     * @details This function prints the current cluster assignments, cluster centers, and cluster dispersions
     */
    if(interest == 1 || interest == -1){
        Rcpp::Rcout << "Cluster assignments: " << std::endl << "\t";
        Rcpp::Rcout << state.c_i << std::endl;
    }

    if(interest == 2 || interest == -1){
        Rcpp::Rcout << "Centers: " << std::endl<< "\t";
        for (int i = 0; i < state.center.length(); i++) {
            Rcpp::Rcout << "Cluster "<< i << " :"<< std::setprecision(5)<< as<NumericVector>(state.center[i]) << std::endl;
            if(i < state.center.length() - 1)
                Rcpp::Rcout << "\t";
        }
    }

    if(interest == 3 || interest == -1){
        Rcpp::Rcout << "Dispersions: " << std::endl << "\t";
        for (int i = 0; i < state.sigma.length(); i++) {
            Rcpp::Rcout << "Cluster "<< i << ": "<< as<NumericVector>(state.sigma[i]) << std::endl;
            if(i < state.center.length() - 1)
                Rcpp::Rcout << "\t";
        }
    }
}

void print_progress_bar(int progress, int total, int bar_width = 50) {
    /**
     * @brief Displays a progress bar in the console
     * @param progress Current progress value
     * @param total Total number of steps
     * @param bar_width Width of the progress bar in characters (default: 50)
     * @note This is a utility function for visual feedback during long computations
     */
    float ratio = static_cast<float>(progress) / total;
    int bar_progress = static_cast<int>(bar_width * ratio);
    
    Rcpp::Rcout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < bar_progress) Rcpp::Rcout << "=";
        else Rcpp::Rcout << " ";
    }
    Rcpp::Rcout << "] " << int(ratio * 100.0) << "%";
    Rcpp::Rcout.flush();
    
    if (progress == total) Rcpp::Rcout << std::endl;
}

IntegerVector unique_classes(const IntegerVector & c_i) {
    /**
     * @brief Gets unique class labels
     * @param c_i Vector of class assignments
     * @param index_to_del Index to exclude from unique class calculation
     * @return NumericVector containing unique class labels
     * @note Exported to R using Rcpp
     */
    IntegerVector unique_vec = unique(c_i);
    std::sort(unique_vec.begin(), unique_vec.end());
    return unique_vec;
}

IntegerVector unique_classes_without_index(const IntegerVector & c_i, const int index_to_del) {
    /**
     * @brief Gets unique class labels excluding a specific index
     * @param c_i Vector of class assignments
     * @param index_to_del Index to exclude from unique class calculation
     * @return NumericVector containing unique class labels
     * @note Exported to R using Rcpp
     */

    std::set<double> unique_classes;
    for (int i = 0; i < c_i.length(); ++i) {
        if (i != index_to_del) {
            unique_classes.insert(c_i[i]);
        }
    }
    return wrap(std::vector<double>(unique_classes.begin(), unique_classes.end()));
}

IntegerVector sample_initial_assignment(double K = 4, int n = 10){
    /**
     * @brief Samples initial cluster assignments
     * @param K Number of initial clusters
     * @return NumericVector containing initial cluster assignments
     * @note Exported to R using Rcpp
     */
    
    IntegerVector cluster_assignments = Rcpp::sample(K, n, true); 
    cluster_assignments = cluster_assignments - 1; 
    return cluster_assignments;
}

int count_cluster_members(const IntegerVector& c_i, int exclude_index, int cls) {
    // Add explicit bounds checking
    if (exclude_index < 0 || exclude_index >= c_i.length()) {
        Rcpp::warning("Exclude index %d is out of bounds for vector of length %d", 
                      exclude_index, c_i.length());
        return 0;
    }
    
    int n_i_z = 0;
    for (int i = 0; i < c_i.length(); i++) {
        // Explicit bounds check in the loop
        if (i < 0 || i >= c_i.length()) {
            std::cerr << "Unexpected index " << i <<" in count_cluster_members" << std::endl;
            break;
        }
        
        if (i != exclude_index && c_i[i] == cls) {
            n_i_z++;
        }
    }
    
    return n_i_z;
}

NumericVector sample_center_1_cluster(const IntegerVector & attrisize) {
    /**
     * @brief Sample initial cluster center
     * @param attrisize Vector of attribute sizes
     * @return NumericVector containing cluster centers
     */

    NumericVector center(attrisize.length());

    for (int a = 0; a < attrisize.length(); a++) {
        center[a] = sample(attrisize[a], 1, true)[0];
    } 

    return center;
}

void sample_centers(List & center, const int number_cls, const IntegerVector & attrisize) {
    /**
     * @brief Sample initial cluster centers
     * @param number_cls Number of clusters
     * @param attrisize Vector of attribute
     * @return List of cluster centers for each attribute
     * @note The returned list contains a NumericVector for each attribute, So has dimension equal to the number of attributes x number_cls
     * @note We have some perfomance issues with this function? Can we change this using NumericMatrix?
     */

    for (int c = 0; c < number_cls; c++) {
        center.push_back(sample_center_1_cluster(attrisize));
    }
}

NumericVector sample_sigma_1_cluster(const IntegerVector & attrisize, const NumericVector & v, const NumericVector & w){
    /**
     * @brief Sample initial cluster dispersion (sigma)
     * @param attrisize Vector of attribute sizes
     * @param number_cls Number of clusters
     * @param v Parameter for hypergeometric distribution
     * @param w Parameter for hypergeometric distribution
     * @return NumericVector containing cluster dispersions for each attribute
     */

    NumericVector sigma(attrisize.length());

    for (int i = 0; i < attrisize.length(); i++) {
        sigma[i] = rhyper_sig(1, v[i], w[i], attrisize[i])[0];
    } 

    return sigma;
}

void sample_sigmas(List & sigma, const int number_cls, const aux_data & const_data) {
    /**
     * @brief Sample initial cluster dispersions (sigma)
     * @param attrisize Vector of attribute sizes
     * @param number_cls Number of clusters
     * @return List of cluster dispersions for each attribute
     */

    for (int i = 0; i < number_cls; i++) {
        sigma.push_back(sample_sigma_1_cluster(const_data.attrisize, const_data.v, const_data.w));
    } 
}

void clean_var(internal_state & updated_state, const internal_state & current_state, const IntegerVector& existing_cls, const IntegerVector& attrisize) {
    /**
     * @brief Clean up internal state variables
     * @param updated_state Updated internal state
     * @param current_state Current internal state
     * @param existing_cls Existing cluster indices
     * @param attrisize Vector of attribute sizes
     * @details This function cleans up the internal state variables by removing unused clusters
     */

    // Efficiently find unique existing clusters
    int num_existing_cls = existing_cls.length();

    // Create a mapping of the existing clusters
    std::unordered_map<int, int> cls_to_new_index;
    for(int i = 0; i < num_existing_cls; ++i) {
        int idx_temp = 0;
        // If the cluster index is less than the number of existing clusters, keep the same index
        if(existing_cls[i] < num_existing_cls){
                cls_to_new_index[existing_cls[i]] = existing_cls[i];
        }
        else{
            // find the first available index for new clusters
            while(cls_to_new_index.find(idx_temp) != cls_to_new_index.end() && idx_temp < num_existing_cls){
                idx_temp++;
            }
            
            cls_to_new_index[existing_cls[i]] = idx_temp;
        }
    }
    
    // Preallocate new containers with the correct size
    List new_center(num_existing_cls);
    List new_sigma(num_existing_cls);

    // Check if i is correct or we need cls_to_new_index[existing_cls[i]]
    for(int i = 0; i < num_existing_cls; ++i) {
        new_center[i] = current_state.center[existing_cls[i]];
        new_sigma[i] = current_state.sigma[existing_cls[i]];
    }

    // Update state with new centers, sigmas, and cluster count
    updated_state.center = std::move(new_center);
    updated_state.sigma = std::move(new_sigma);
    updated_state.total_cls = num_existing_cls;

    // Vectorized cluster index update using the mapping
    for(int i = 0; i < current_state.c_i.length(); i++) {
        auto it = cls_to_new_index.find(current_state.c_i[i]);
        if(it != cls_to_new_index.end()) {
            updated_state.c_i[i] = it->second;
        }
    }
}

void sample_allocation(const int index_i, const aux_data & constant_data, 
                     internal_state & state, const int m, const IntegerVector & unique_classes_without_i) {
    /**
     * @brief Sample new cluster assignment for a single data point
     * @param index_i Index of the data point
     * @param constant_data Auxiliary data for the MCMC algorithm
     * @param state Internal state of the MCMC algorithm
     * @param m Number of latent classes
     * @return New cluster assignment for the data point
     */
    // read data point
    NumericVector x_i = constant_data.data(index_i, _);

    IntegerVector uni_clas = unique_classes(state.c_i);
    int k_minus = unique_classes_without_i.length();
    int h = k_minus + m; 
    int m_temp = uni_clas.length() == k_minus ? m : m - 1;
    NumericVector temp_center = as<List>(state.center)[state.c_i[index_i]];
    NumericVector temp_sigma = as<List>(state.sigma)[state.c_i[index_i]];

    
    // ------------------------------- Ri assegnamento labels --------------------------------
    internal_state state_temp = {IntegerVector(state.c_i.length()), List(state.center.length()), List(state.sigma.length()), 0};
    clean_var(state_temp, state, unique_classes_without_i, constant_data.attrisize);

    // ------------------------------- Add latent classes -------------------------------
    if(uni_clas.length() != k_minus){
        state_temp.center.push_back(temp_center);
        state_temp.sigma.push_back(temp_sigma);
    }

    for(int i = 0; i < m_temp; i++){
        state_temp.center.push_back(sample_center_1_cluster(constant_data.attrisize));
        state_temp.sigma.push_back(sample_sigma_1_cluster(constant_data.attrisize, constant_data.v, constant_data.w));
    } 
    // update total number of clusters
    state_temp.total_cls = state_temp.center.length();
    
    // Calculate probabilities
    NumericVector probs(state_temp.total_cls);
    NumericVector sigma_k(constant_data.attrisize.length());
    NumericVector center_k(constant_data.attrisize.length());
    
    // number of variable in cluster z without i
    int n_i_z = 0;

    // prob of existing clusters
    for (int k = 0; k < k_minus; k++) {
        // Density calculation, in Neal paper is the F function
        double Hamming = 0;

        sigma_k = state_temp.sigma[k]; // prendo le sigma del cluster k
        center_k = state_temp.center[k]; // prendo i centri del cluster k

        for (int j = 0; j < x_i.length(); j++) {
            Hamming += dhamming(x_i[j], center_k[j], sigma_k[j], constant_data.attrisize[j], true);
        }

        // Count instances in the z cluster excluding the current point i
        if(k < unique_classes_without_i.length())
            n_i_z = count_cluster_members(state_temp.c_i, index_i, unique_classes_without_i[k]);
        // Calculate probability
        if(k < probs.length())
            if (n_i_z == 0) {
                probs[k] = 0;
            } else {
                probs[k] = n_i_z * std::exp(Hamming);
            }
    }

    // prob of latent clusters
    for(int k = k_minus; k < state_temp.total_cls; k++){
        sigma_k = state_temp.sigma[k]; // prendo le sigma del cluster k
        center_k = state_temp.center[k]; // prendo i centri del cluster k

        double Hamming = 0;
        for (int j = 0; j < x_i.length(); j++) {
            Hamming += dhamming(x_i[j], center_k[j], sigma_k[j], constant_data.attrisize[j], true);
        }

        // probability calculation for latent clusters 
        probs[k] = constant_data.gamma/m * std::exp(Hamming);    
    }

    // Create the vector from 0 to num_cls 
    NumericVector cls(state_temp.total_cls);
    for (int i = 0; i < state_temp.total_cls; ++i) {
        cls[i] = i;
    }

    // Normalize probabilities
    probs = probs / sum(probs);    

    // Sample new cluster assignment using probabilities calculated before
    state_temp.c_i[index_i] = sample(cls, 1, true, probs)[0];

    // ------------------------------- Ri assegnamento labels --------------------------------

    clean_var(state, state_temp, unique_classes(state_temp.c_i), constant_data.attrisize);
}

NumericMatrix subset_data_for_cluster(const NumericMatrix & data, int cluster, const internal_state & state) {
    /**
     * @brief Extract data for a specific cluster
     * @param data Input data matrix
     * @param cluster Cluster index
     * @return NumericMatrix containing data for the specified cluster
     */
    IntegerVector cluster_indices;
    for (int i = 0; i < as<NumericVector>(state.c_i).length(); ++i) 
        if ( as<NumericVector>(state.c_i)[i] == cluster) 
            cluster_indices.push_back(i);
        
    
    NumericMatrix cluster_data(cluster_indices.length(), data.ncol());
    for (int i = 0; i < cluster_indices.length(); ++i) {
        cluster_data(i, _) = data(cluster_indices[i], _);
    }
    
    return cluster_data;
}

void update_centers(internal_state & state, const aux_data & const_data) {
    /**
     * @brief Update cluster centers
     * @param state Internal state of the MCMC algorithm
     * @param const_data Auxiliary data for the MCMC algorithm
     * @note This function updates the cluster centers using the current cluster assignments
     */

    IntegerVector clusters = unique_classes(state.c_i);
    int num_cls = state.total_cls;    // Initialize cluster assignments
    List prob_centers;

    NumericVector attr_centers(const_data.attrisize.length());
    List prob_centers_cluster;
    
    List attri_List = Attributes_List(const_data.data, const_data.data.ncol());    
    for (int i = 0; i < num_cls; i++) { 
        NumericMatrix data_tmp = subset_data_for_cluster(const_data.data, i, state);
        prob_centers = Center_prob(data_tmp, state.sigma[i], as<NumericVector>(const_data.attrisize));
        state.center[i] = Samp_Center(attri_List, prob_centers, const_data.attrisize.length());
    }
}

void update_sigma(List & sigma, const List & centers, const IntegerVector & c_i, const aux_data & const_data) {
    int num_cls = sigma.length();
    NumericVector new_w(const_data.attrisize.length());
    NumericVector new_v(const_data.attrisize.length());
    
    for (int c = 0; c < num_cls; c++) { // for each cluster
        // Create indices for rows in this cluster
        IntegerVector cluster_indices;
        for (int i = 0; i < c_i.length(); ++i) {
            if (c_i[i] == c) {
                cluster_indices.push_back(i);
            }
        }
        
        // Extract cluster-specific data
        NumericMatrix cluster_data(cluster_indices.length(), const_data.data.ncol());
        for (int i = 0; i < cluster_indices.length(); ++i) {
            cluster_data(i, _) = const_data.data(cluster_indices[i], _);
        }
        
        int nm = cluster_indices.length();
        NumericVector sigmas_cluster = as<NumericVector>(sigma[c]);
        NumericVector centers_cluster = as<NumericVector>(centers[c]);
        
        for (int i = 0; i < const_data.attrisize.length(); ++i ){ // for each attribute
            NumericVector col = cluster_data(_, i);
            double sumdelta = sum(col != centers_cluster[i]);
            new_w[i] = const_data.w[i] + nm - sumdelta;
            new_v[i] = const_data.v[i] + sumdelta;   
        }
        sigma[c] = clone(sample_sigma_1_cluster(const_data.attrisize, new_v, new_w));
    }
}

double compute_loglikelihood(internal_state & state, aux_data & const_data) {
    double loglikelihood = 0.0;
    
    // Compute likelihood for each observation
    for (int i = 0; i < const_data.n; i++) {
        int cluster = state.c_i[i];
        NumericVector center = as<NumericVector>(state.center[cluster]);
        NumericVector sigma = as<NumericVector>(state.sigma[cluster]);
        
        double Hamming = 0.0;
        for (int j = 0; j < const_data.attrisize.length(); j++) {
            Hamming += dhamming(const_data.data(i, j), center[j], sigma[j], const_data.attrisize[j], true);
        }
        loglikelihood += Hamming;
    }
    
    return loglikelihood;
}





void restricted_gibbs_sampler(internal_state & state, int idx1, int idx2, std::vector<int> & S, aux_data & const_data) {
    /**
     * @brief Restricted Gibbs Sampler between the cluster of two observations
     * @param state Internal state of the MCMC algorithm
     * @param idx1 Index of the first chosen observation
     * @param idx2 Index of the second chosen observation
     * @param S Vector of indices of observations in the same cluster of idx1 or idx2
     */

    // Create a vector of unique classes
    IntegerVector unique_cls = unique_classes(state.c_i);
    int num_cls = unique_cls.length();

    // Extract cluster of the first observation
    int cls1 = state.c_i[idx1];
    NumericVector center1 = as<NumericVector>(state.center[cls1]);
    NumericVector sigma1 = as<NumericVector>(state.sigma[cls1]);

    // Extract cluster of the second observation
    int cls2 = state.c_i[idx2];
    NumericVector center2 = as<NumericVector>(state.center[cls2]);
    NumericVector sigma2 = as<NumericVector>(state.sigma[cls2]);

    // Update cluster assignments for each observation in S
    for (unsigned i = 0; i < S.size(); i++) {
        int obs_idx = S[i];
        NumericVector x_i = const_data.data(obs_idx, _);

        // Calculate probabilities for the two clusters
        NumericVector probs(2);
        for (int k = 0; k < 2; k++) {
                NumericVector center_k; //correggere inizializzazione
                NumericVector sigma_k; //correggere inizializzazione
            if(k == 0){
                 center_k = center1;
                 sigma_k = sigma1;
            } else {
                 center_k = center2;
                 sigma_k = sigma2;
            }

            double Hamming = 0;
            for (int j = 0; j < x_i.length(); j++) {
                Hamming += dhamming(x_i[j], center_k[j], sigma_k[j], const_data.attrisize[j], true);
            }

            probs[k] = std::exp(Hamming);
        }

        // Normalize probabilities
        probs = probs / sum(probs);

        // Sample new cluster assignment between the two clusters of idx1 and idx2
        state.c_i[obs_idx] = sample(IntegerVector::create(cls1, cls2), 1, true, probs)[0];
    }
}


int fact(int n){
    if (n < 0) {
        std::cerr<<"numero negativo nel fattoriale"<<std::endl;
    }
    if (n == 0 || n == 1) {
        return 1;
    }
    return n * fact(n - 1);
}

int cls_elem(internal_state & gamma, int k){
    int f=0;
    for (int i=0; i<gamma.c_i.length(); i++){
        if (gamma.c_i[i]==k) 
            f++;
    }
    return f;
}

double priors(internal_state & gamma, int obs_index, aux_data & const_data){
    int ci=gamma.c_i[obs_index];
    NumericVector sigmap=gamma.sigma[ci];
    NumericVector centerp=gamma.center[ci];

    double priorg=1;
    // dhyper_raf in hyperg.cpp
    for (int h=0; h<centerp.length(); h++){
            priorg*=1/const_data.attrisize[h]; // densità dell'uniforme è sempre 1/numero_modalità
            priorg*=dhyper_raf(sigmap[h], const_data.v[h] ,const_data.w[h],const_data.attrisize[h], false)[0]; //ricontrollare la funzione restituisce vettore
    }
    return priorg;
}

double probgs_phi(const internal_state & gamma_star, const internal_state & gamma, const aux_data & const_data, const std::vector<int> & S, const int & choosen_idx){
    /**
     * @brief Compute the probability of the cluster dispersion and cluster center - paper reference: P_{GS}(\phi*|phi^L, c^L, y)
     * @param gamma_star state containing the new cluster assignment, new cluster centers, and new cluster dispersions
     * @param gamma state containing the launch cluster assignment, launch cluster centers, and launch cluster dispersions - paper reference: (c^L, \phi^L)
     * @param const_data auxiliary data for the MCMC algorithm containing the input data matrix, attribute sizes, and hypergeometric parameters
     * @param S vector of indices of observations in the same cluster of idx1 or idx2
     * @param star string to identify the type of operation (split or merge)
     * @param choosen_idx index of the chosen observation
     */

    // Variable to store the probability of the cluster dispersion and cluster center
    double center_prob=1;
    double sigma_prob=0;

    // --------------- Center probs ---------------
    // Compute the uniform probability of the cluster center
    int p = as<IntegerVector>(gamma_star.center[gamma_star.c_i[choosen_idx]]).length();
    for (int j=0; j<p; j++){
        center_prob*=1/const_data.attrisize[j];
    }  
    

    // --------------- Sigma probs ---------------
    // Compute the probability of the cluster dispersion
    // dhyper_raf in hyperg.cpp
    for (int j=0; j<p; j++){
        double temp = dhyper_raf(as<NumericVector>(gamma_star.sigma[gamma_star.c_i[choosen_idx]])[j], const_data.v[j], const_data.w[j], const_data.attrisize[j], true)[0];
        sigma_prob += temp;
    }
    
    // Since the function returns the log of the probability, we need to exponentiate it
    sigma_prob = std::exp(sigma_prob);
    


    return center_prob*sigma_prob;
}

double probgs_c_i(const internal_state & gamma_star, const internal_state & gamma, const aux_data & const_data, const std::vector<int> & S, const int idx1, const int idx2){
    /**
     * @brief Compute the probability of the cluster assignment - paper reference: P_{GS}(c*|c^L, phi*, y)
     * @param gamma_star state containing the new cluster assignment, new cluster centers, and new cluster dispersions
     * @param gamma state containing the launch cluster assignment, launch cluster centers, and launch cluster dispersions - paper reference: (c^L, \phi^L)
     * @param const_data auxiliary data for the MCMC algorithm containing the input data matrix, attribute sizes, and hypergeometric parameters
     * @param S vector of indices of observations in the same cluster of idx1 or idx2
     * @param idx1 index of the first chosen observation
     * @param idx2 index of the second chosen observation
     */

    // Variable to store the probability of the cluster assignment
    double pgs=1;

    // Save cluster centers and dispersions for the two observations
    NumericVector center_i=gamma_star.center[gamma.c_i[idx1]]; 
    NumericVector sigma_i=gamma_star.sigma[gamma.c_i[idx1]];

    NumericVector center_j=gamma_star.center[gamma.c_i[idx2]];
    NumericVector sigma_j=gamma_star.sigma[gamma.c_i[idx2]];

    // Compute the probability of the cluster assignment without the two observations idx1 and idx2
    for (unsigned k=0; k<S.size(); k++){
        // extract the k-th observation data point
        NumericVector y_k = const_data.data(k, _);

        double num=0, deni=0, denj=0;
        // Compute the number of elements in the cluster k without the k observation
        int nk=count_cluster_members(gamma.c_i, k, gamma.c_i[k]);
        // Compute the number of elements in the cluster i without the k observation
        int ni=count_cluster_members(gamma.c_i, k, gamma.c_i[idx1]);
        // Compute the number of elements in the cluster j without the k observation
        int nj=count_cluster_members(gamma.c_i, k, gamma.c_i[idx2]);
        
        // Extract the cluster centers and dispersions for the cluster of k observation
        NumericVector center_k=gamma_star.center[gamma.c_i[k]];
        NumericVector sigma_k=gamma_star.sigma[gamma.c_i[k]];
        
        // Compute the log-Hamming between the k-th observation and the cluster centers
        for (int j=0; j<y_k.length(); j++){
            num+=dhamming(y_k[j], center_k[j], sigma_k[j], const_data.attrisize[j], true);
            deni+=dhamming(y_k[j], center_i[j], sigma_i[j], const_data.attrisize[j], true);
            denj+=dhamming(y_k[j], center_j[j], sigma_j[j], const_data.attrisize[j], true);
        }
        // Compute the probability of the cluster assignment
        pgs*=(nk*std::exp(num))/(ni*std::exp(deni)+nj*std::exp(denj));
    }

    return pgs;
}



double acceptance_ratio(internal_state & gamma, internal_state & gamma_star, aux_data & const_data, double & q, int obs_1_idx, int obs_2_idx, std::vector<int> & S, const std::string & star){
    double qpl=1;
    double alpha=const_data.gamma;
    // compute L(gamma|y) and L(gammastar|y)    va bene la *LOG*likelihood?
    double Lgamma=0, Lgammastar=0, ratioL=1;
    Lgamma=compute_loglikelihood(gamma,const_data);
    Lgammastar=compute_loglikelihood(gamma_star,const_data);
    ratioL=exp(Lgammastar)/exp(Lgamma);

    // compute P(gamma) and P(gammastar)
    double Pgamma=0, Pgammastar=0; 
    double ratioP=1;

    if (star=="split"){
        Pgammastar=(fact(cls_elem(gamma_star,gamma_star.c_i[obs_1_idx])-1))*(fact(cls_elem(gamma_star,gamma_star.c_i[obs_2_idx])-1))*priors(gamma_star, obs_1_idx, const_data)*priors(gamma_star, obs_2_idx, const_data);
        Pgamma=(fact(cls_elem(gamma,gamma.c_i[obs_1_idx])-1))*priors(gamma, obs_1_idx, const_data);
        ratioP=alpha*Pgammastar/Pgamma;
    }
    if (star=="merge"){
        Pgamma=(fact(cls_elem(gamma,gamma.c_i[obs_1_idx])-1))*(fact(cls_elem(gamma,gamma.c_i[obs_2_idx])-1))*priors(gamma, obs_1_idx, const_data)*priors(gamma, obs_2_idx, const_data);
        Pgammastar=(fact(cls_elem(gamma_star,gamma_star.c_i[obs_1_idx])-1))*priors(gamma_star, obs_1_idx, const_data);
        ratioP=(1/alpha)*(Pgammastar/Pgamma);
    }

    // q arriva da fuori
    qpl= q*ratioP*ratioL;
    // return minimum, see equation (4)
    return std::min(qpl, 1.0);
}




void split_and_merge(internal_state & state, aux_data & const_data, int t = 100, int r = 100) {
    /**
     * @brief Split and merge step
     * @details This function implements the split and merge step of the MCMC algorithm
     */

    // --------------- Step 1 ---------------
	// choose 2 observation random from the data
    std::cout<<"Inizio Split and Merge: " << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, const_data.n - 1);
	int obs_1_idx = dis(gen);
	int obs_2_idx;
	do {
    		obs_2_idx = dis(gen);
	} while (obs_2_idx == obs_1_idx);
    std::cout<<"osservazioni scelte: " << obs_1_idx<<" e  " << obs_2_idx<< std::endl;
	
	// --------------- Step 2 ---------------
	// Create S the set of idx of obs in the same cluster of obs_1 or obs_2
	// REMIND: obs_1 and obs_2 aren't in S
	std::vector<int> S;
    for(int i = 0; i < const_data.n; ++i){
        // skip obs_1 and obs_2
        if(i == obs_1_idx || i == obs_2_idx)
            continue;

        if(state.c_i[i] == state.c_i[obs_1_idx] || state.c_i[i] == state.c_i[obs_2_idx]){
            S.push_back(i);
        }
    }
    
	std::cout<<"\ncluster a cui appartengono: " << state.c_i[obs_1_idx]<<" e  " << state.c_i[obs_2_idx]<< std::endl;
    std::cout<<"lunghezza di S: "<< S.size()<<std::endl;
	
	// --------------- Step 3 ---------------
	// gamma split
	IntegerVector c_L_split(state.c_i);
	List center_L_split = clone(state.center);
	List sigma_L_split = clone(state.sigma);
	
	// gamma merge	
	IntegerVector c_L_merge(state.c_i);
	List center_L_merge = clone(state.center);
	List sigma_L_merge = clone(state.sigma);
	
    // ----- gamma split popolation -----
    if(state.c_i[obs_1_idx] == state.c_i[obs_2_idx]){
        std::cout<<"\n siamo nello split 1.1" <<std::endl;
        int lat_cls = unique_classes(state.c_i).length(); 
        // set the allocation of obs_1_idx to a latent cluster
        c_L_split[obs_1_idx] = lat_cls;
    }
    
    // randomly allocate with equal probs data in S between cls-1 and cls-2
    for(unsigned datum = 0; datum < S.size(); ++datum){
        // assignment with equal probs
        c_L_split[S[datum]] = sample(2, 1, true)[0] == 1 ? c_L_split[obs_1_idx] : c_L_split[obs_2_idx];
        std::cout<<std::endl;
    }
    
    // Draw a new value for the centers
    center_L_split[c_L_split[obs_1_idx]] = sample_center_1_cluster(const_data.attrisize);
    center_L_split[c_L_split[obs_2_idx]] = sample_center_1_cluster(const_data.attrisize);
    
    // draw a new value for the sigma
    sigma_L_split[c_L_split[obs_1_idx]] = sample_sigma_1_cluster(const_data.attrisize, const_data.v, const_data.w);
    sigma_L_split[c_L_split[obs_2_idx]] = sample_sigma_1_cluster(const_data.attrisize, const_data.v, const_data.w);
    
    // aux state for the split
    internal_state state_split = {c_L_split, center_L_split, sigma_L_split, static_cast<int>(unique_classes(c_L_split).length())};

    // Intermediate restricted Gibbs Sampler on c_L_split
    for(int iter = 0; iter < t; ++iter ){
        restricted_gibbs_sampler(state_split, obs_1_idx, obs_2_idx, S, const_data);
        // update both cls new center and sigma 
        update_centers(state_split, const_data);
        update_sigma(state_split.sigma, state_split.center, state_split.c_i, const_data);
    }
            
	
	// ----- gamma merge popolation -----
    if(state.c_i[obs_1_idx] != state.c_i[obs_2_idx]){
        std::cout<<"\n siamo nel merge 1" <<std::endl;
        // set the allocation of obs_1_idx equal to the cls of obs_2 (c_j)
        c_L_merge[obs_1_idx] = state.c_i[obs_2_idx];
    }

    // Allocate all the data in S to the cls of obs_2_idx
    for(unsigned datum = 0; datum < S.size(); ++datum){
        // assignment with equal probs
        c_L_merge[S[datum]] = state.c_i[obs_2_idx];
    }
    
    // Draw a new value for the centers
    center_L_merge[c_L_merge[obs_2_idx]] = sample_center_1_cluster(const_data.attrisize);
    
    // draw a new value for the sigma
    sigma_L_merge[c_L_merge[obs_2_idx]] = sample_sigma_1_cluster(const_data.attrisize, const_data.v, const_data.w);
    
    // aux state for the merge
    internal_state state_merge = {c_L_merge, center_L_merge, sigma_L_merge, static_cast<int>(unique_classes(c_L_split).length())};
    
    // Intermediate restricted Gibbs Sampler on c_L_split
    for(int iter = 0; iter < r; ++iter ){
        // update only merge cls center and sigma 
        update_centers(state_merge, const_data);
        update_sigma(state_merge.sigma, state_merge.center, state_merge.c_i, const_data);
    }
    
	// --------------- Step 4&5 ---------------
	// variable to store prob
	double q = 1;

    // Aux var to store *-state
    internal_state state_star = {IntegerVector(), List(), List(), 0};
    double acpt_ratio=1;

	if(state.c_i[obs_1_idx] == state.c_i[obs_2_idx]){
		state_star.c_i = clone(c_L_split);
		
		// ----- (a) - last Restricted Gibbs Sampler -----
		restricted_gibbs_sampler(state_star, obs_1_idx, obs_2_idx, S, const_data);
		update_centers(state_star, const_data);
		update_sigma(state_star.sigma, state_star.center, state_star.c_i, const_data);
		
		// ----- (b) - Transition probabilities ----- 
        // Equation (15)
        // Numerator
		q *= probgs_phi(state, state_merge, const_data, S, obs_1_idx);
        // Denominator;
        q /= probgs_c_i(state_star, state_split, const_data, S, obs_1_idx, obs_2_idx);
        q /= probgs_phi(state_star, state_split, const_data, S, obs_1_idx);
        q /= probgs_phi(state_star, state_split, const_data, S, obs_2_idx);
					
		// Calculate the acceptance ratio
		acpt_ratio = acceptance_ratio(state, state_star, const_data, q, obs_1_idx, obs_2_idx, S, "split");
    
	}	
	else{
		// ----- (a) - merge -----
		state_star.c_i = clone(c_L_merge);
		
		// Last restricted Gibbs Sampling to update merge cls parameters
		update_centers(state_star, const_data);
		update_sigma(state_star.sigma, state_star.center, state_star.c_i, const_data);
		
		// ----- (b) - transition probabilities -----
        // Equation (16)
        // Numerator
        q *= probgs_phi(state, state_split, const_data, S, obs_1_idx);
        q *= probgs_phi(state, state_split, const_data, S, obs_2_idx);
        q *= probgs_c_i(state, state_split, const_data, S, obs_1_idx, obs_2_idx);
        // Denominator
        q /= probgs_phi(state_star, state_merge, const_data, S, obs_1_idx);
		
		// Calculate the acceptance ratio
		acpt_ratio = acceptance_ratio(state, state_star, const_data, q, obs_1_idx, obs_2_idx, S, "merge");
		
	}
	
	// ----- (c) - Metropolis-Hastings step -----
	// sample if accept or not the MC state stored in c_star
    /*if(sample(acpt_ratio)){
        state = state_star;
    }*/

}


// [[Rcpp::export]]
List run_markov_chain(NumericMatrix data, IntegerVector attrisize, double gamma, NumericVector v, NumericVector w, 
                    int verbose = 0, int m = 5, int iterations = 1000, int L = 1, 
                    Rcpp::Nullable<Rcpp::IntegerVector> c_i = R_NilValue, int burnin = 5000) {
    /**
     * @brief Main Markov Chain Monte Carlo sampling function
     * @param data Input data matrix
     * @param attrisize Vector of attribute sizes
     * @param gamma Concentration parameter
     * @param v Parameter for sigma update
     * @param w Parameter for sigma update
     * @param m Number of latent classes (default: 5)
     * @param iterations Number of MCMC iterations (default: 1000)
     * @return List containing final clustering results
    */
    aux_data const_data = {data, data.nrow(), attrisize, gamma, v, w};
    internal_state state = {IntegerVector(), List(), List(), L};

    // Initialize cluster assignments
    IntegerVector initial_c_i;
    if(c_i.isNotNull()) {
        Rcpp::Rcout << "Initial cluster assignments provided" << std::endl;
        initial_c_i = as<IntegerVector>(c_i) - min(as<IntegerVector>(c_i)); // rescale input cluster assignments to start from 0
        state.total_cls = unique_classes(initial_c_i).length();

    } else {
        initial_c_i = sample_initial_assignment(L, const_data.n);
    }
    state.c_i = std::move(initial_c_i);
    
    // Initialize centers
    sample_centers(state.center, state.total_cls, const_data.attrisize);
    // Initialize sigma
    sample_sigmas(state.sigma, state.total_cls, const_data);

    if(verbose == 2 or verbose == 1){
        print_internal_state(state);
    }

    List results = List::create(Named("total_cls") = List(iterations),
                                Named("c_i") = List(iterations),
                                Named("centers") = List(iterations),
                                Named("sigmas") = List(iterations), 
                                Named("loglikelihood") = NumericVector(iterations));

    auto start_time = std::chrono::high_resolution_clock::now();
    Rcpp::Rcout << "Starting Markov Chain sampling..." << std::endl;
    
    int n_update_latent = 0; 
    for (int iter = 0; iter < iterations + burnin; iter++) {
        if(verbose != 0)
            std::cout << std::endl <<"[DEBUG] - Iteration " << iter << " of " << iterations << std::endl;

        // Sample new cluster assignments for each observation
        for (int index_i = 0; index_i < const_data.n; index_i++) {
            L = unique_classes(state.c_i).length();
            IntegerVector unique_classes_without_i = unique_classes_without_index(state.c_i, index_i);
            
            // Sample new cluster assignment for observation i
            sample_allocation(index_i, const_data, state, m, unique_classes_without_i);       
        } 
        
        // Update centers and sigmas
        update_centers(state, const_data);
        update_sigma(state.sigma, state.center, state.c_i, const_data);

        if(verbose == 2){
            print_internal_state(state);
        }

        
        // Split and merge step
        std::cout<<"Split and Merge step"<<std::endl;
        split_and_merge(state, const_data);

        // Calculate likelihood
        double loglikelihood = compute_loglikelihood(state, const_data);

        // Update progress bar
        if(verbose == 0)
            print_progress_bar(iter + 1, iterations + burnin);

        // Save results
        if(iter >= burnin){
            as<List>(results["total_cls"])[iter - burnin] = state.total_cls;
            as<List>(results["c_i"])[iter - burnin] = clone(state.c_i);
            as<List>(results["centers"])[iter - burnin] = clone(state.center);
            as<List>(results["sigmas"])[iter - burnin] = clone(state.sigma);
            as<NumericVector>(results["loglikelihood"])[iter - burnin] = loglikelihood;
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    Rcpp::Rcout << std::endl << "Markov Chain sampling completed in: "<< duration.count() << " s"<< std::endl;
    
    return results;
}
