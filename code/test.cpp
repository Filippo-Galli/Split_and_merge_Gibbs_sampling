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

    center.resize(number_cls);
    for (int c = 0; c < number_cls; c++) {
        center[c] = sample_center_1_cluster(attrisize);
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
    sigma.resize(number_cls);

    for (int i = 0; i < number_cls; i++) {
        sigma[i] = sample_sigma_1_cluster(const_data.attrisize, const_data.v, const_data.w);
    } 
}

void clean_var_1(internal_state & state, const IntegerVector& attrisize, const IntegerVector& existing_cls, 
                List & centers, List & sigmas, int & total_cls, IntegerVector & c_i) {
    // Efficiently find unique existing clusters
    int num_existing_cls = existing_cls.length();

    // Create a mapping for efficient cluster index lookup
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
        new_center[i] = centers[existing_cls[i]];
        new_sigma[i] = sigmas[existing_cls[i]];
    }

    // Update state with new centers, sigmas, and cluster count
    centers = std::move(new_center);
    sigmas = std::move(new_sigma);
    total_cls = num_existing_cls;

    // Vectorized cluster index update using the mapping
    for(int i = 0; i < c_i.length(); i++) {
        auto it = cls_to_new_index.find(c_i[i]);
        if(it != cls_to_new_index.end()) {
            c_i[i] = it->second;
        }
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
    /**
     * @brief Compute log-likelihood of the current state
     * @param state Internal state of the MCMC algorithm
     * @param const_data Auxiliary data for the MCMC algorithm
     * @return Log-likelihood of the current state
     */

    double loglikelihood = 0.0;

    // Pre-compute Hamming distance matrices
    std::vector<NumericMatrix> hamming_distances(state.total_cls);
    for (int k = 0; k < state.total_cls; k++) {
        NumericVector center = as<NumericVector>(state.center[k]);
        NumericVector sigma = as<NumericVector>(state.sigma[k]);
        hamming_distances[k] = dhamming_matrix(const_data.data, center, sigma, const_data.attrisize);
    }

    // Compute loglikelihood using pre-computed Hamming distances
    for (int i = 0; i < const_data.n; i++) {
        int cluster = state.c_i[i];
        loglikelihood += hamming_distances[cluster](i, _).sum();
    }

    return loglikelihood;
}

NumericMatrix dhamming_matrix(const NumericMatrix & data, const NumericVector & center, const NumericVector & sigma, const IntegerVector & attrisize) {
    /**
     * @brief Compute Hamming distance matrix
     * @param data Input data matrix
     * @param center Cluster center
     * @param sigma Cluster dispersion
     * @param attrisize Vector of attribute sizes
     * @return NumericMatrix containing Hamming distances
     */
    int n = data.nrow();
    int p = data.ncol();
    NumericMatrix hamming_dist(n, p);

    for (int j = 0; j < p; j++) {
        NumericVector col = data(_, j);
        hamming_dist(_, j) = dhamming(col, center[j], sigma[j], attrisize[j], false);
    }

    return hamming_dist;
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
