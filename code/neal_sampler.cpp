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

bool debugging = true;

namespace debug {

template<typename... Args>
void print(unsigned int tab_level, const char* func_name, int line, const std::string& message, const Args&... args) {
    std::stringstream ss;
    
    // Add tabs based on level
    for(unsigned int i = 0; i < tab_level; ++i) {
        ss << '\t';
    }
    
    // Add debug prefix with function name and line number
    ss << "[DEBUG:" << func_name << ":" << line << "] ";
    
    // Handle the message with potential format specifiers
    size_t pos = 0;
    size_t count = 0;
    std::string temp = message;
    
    // Fold expression to process all arguments
    ((ss << temp.substr(pos, message.find("{}", pos) - pos) 
        << args
        , pos = message.find("{}", pos) + 2
        , count++), ...);
        
    // Add any remaining message text
    ss << message.substr(pos);
    
    // Output the final formatted string
    Rcpp::Rcout << ss.str() << std::endl;
}

// Overload for when there are no variables to format
inline void print(unsigned int tab_level, const char* func_name, int line, const std::string& message) {
    std::stringstream ss;
    for(unsigned int i = 0; i < tab_level; ++i) {
        ss << '\t';
    }
    Rcpp::Rcout << ss.str() << "[DEBUG:" << func_name << ":" << line << "] " << message << std::endl;
}

// Convenience macro to automatically include function name and line number
#define DEBUG_PRINT(level, message, ...) \
    debug::print(level, __func__, __LINE__, message, ##__VA_ARGS__)

}

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
     * @param state Internal state of the MCMC algorithm
     * @param interest Index of what print (default: -1 <-> print all). 1: print cluster assignments, 2: print cluster centers, 3: print cluster dispersions
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
     * @brief Sample cluster centers parameter from its prior
     * @param attrisize Vector of attribute sizes
     * @return NumericVector containing cluster centers
     */

    NumericVector center(attrisize.length());

    // sample from uniform {1 - attrisize[j]}
    for (int j = 0; j < attrisize.length(); j++) {
        center[j] = sample(attrisize[j], 1, true)[0];
    } 

    return center;
}

List sample_centers(const int number_cls, const IntegerVector & attrisize) {
    /**
     * @brief Sample initial cluster centers
     * @param number_cls Number of clusters
     * @param attrisize Vector of attribute
     * @return List of cluster centers for each attribute
     * @note The returned list contains a NumericVector for each attribute, So has dimension equal to the number of attributes x number_cls
     * @note We have some perfomance issues with this function? Can we change this using NumericMatrix?
     */

    List centers(number_cls);

    for (int c = 0; c < number_cls; c++) {
        centers[c] = sample_center_1_cluster(attrisize);
    }   
    return centers;
}

NumericVector sample_sigma_1_cluster(const IntegerVector & attrisize, const NumericVector & v, const NumericVector & w){
    /**
     * @brief Sample cluster dispersion (sigma) parameter from its prior
     * @param attrisize Vector of attribute sizes
     * @param number_cls Number of clusters
     * @param v Parameter for hypergeometric distribution
     * @param w Parameter for hypergeometric distribution
     * @return NumericVector containing cluster dispersions for each attribute
     */

    NumericVector sigma(attrisize.length());

    // sample from HIG(v, w)
    for (int j = 0; j < attrisize.length(); j++) {
        sigma[j] = rhyper_sig(1, w[j], v[j], attrisize[j])[0]; //inversion of parameter coming from previous written function
    } 

    return sigma;
}

List sample_sigmas(const int number_cls, const aux_data & const_data) {
    /**
     * @brief Sample initial cluster dispersions (sigma)
     * @param attrisize Vector of attribute sizes
     * @param number_cls Number of clusters
     * @return List of cluster dispersions for each attribute
     */
    List sigma(number_cls);

    for (int i = 0; i < number_cls; i++) {
        sigma[i] = sample_sigma_1_cluster(const_data.attrisize, const_data.v, const_data.w);
    } 

    return sigma;
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
    NumericVector y_i = constant_data.data(index_i, _);

    IntegerVector uni_clas = unique_classes(state.c_i);
    int k_minus = unique_classes_without_i.length();
    //int h = k_minus + m; 
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

        for (int j = 0; j < y_i.length(); j++) {
            Hamming += dhamming(y_i[j], center_k[j], sigma_k[j], constant_data.attrisize[j], true);
        }

        // Count instances in the z cluster excluding the current point i
        if(k < unique_classes_without_i.length())
            n_i_z = count_cluster_members(state_temp.c_i, index_i, unique_classes_without_i[k]);
        // Calculate probability
        if(k < probs.length()){
            if (n_i_z == 0) {
                probs[k] = 0;
            } 
            else {
                probs[k] = n_i_z * std::exp(Hamming);
            }
        }
    }

    // prob of latent clusters
    for(int k = k_minus; k < state_temp.total_cls; k++){
        sigma_k = state_temp.sigma[k]; // prendo le sigma del cluster k
        center_k = state_temp.center[k]; // prendo i centri del cluster k

        double Hamming = 0;
        for (int j = 0; j < y_i.length(); j++) {
            Hamming += dhamming(y_i[j], center_k[j], sigma_k[j], constant_data.attrisize[j], true);
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

void update_centers(internal_state & state, const aux_data & const_data, std::vector<int> cluster_indexes = {}) {
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

    // If no specific clusters specified, update all clusters
    if (cluster_indexes.size() == 0) {
        for (int i = 0; i < num_cls; i++) {
            NumericMatrix data_tmp = subset_data_for_cluster(const_data.data, i, state);
            prob_centers = Center_prob(data_tmp, state.sigma[i], as<NumericVector>(const_data.attrisize));
            state.center[i] = Samp_Center(attri_List, prob_centers, const_data.attrisize.length());
        }
    } else {
        // Update only the specified clusters
        for (int idx : cluster_indexes) {
            if (idx >= 0 && idx < num_cls) {  // Bounds check
                NumericMatrix data_tmp = subset_data_for_cluster(const_data.data, idx, state);
                prob_centers = Center_prob(data_tmp, state.sigma[idx], as<NumericVector>(const_data.attrisize));
                state.center[idx] = Samp_Center(attri_List, prob_centers, const_data.attrisize.length());
            }
        }
    }
}

void update_sigma(List & sigma, const List & centers, const IntegerVector & c_i, 
                 const aux_data & const_data, std::vector<int> clusters_to_update = {}) {
    int num_cls = sigma.length();
    NumericVector new_w(const_data.attrisize.length());
    NumericVector new_v(const_data.attrisize.length());
    
    // Determine which clusters to process
    std::vector<int> clusters;
    if (clusters_to_update.size() == 0) {
        // If no specific clusters provided, process all clusters
        clusters.resize(num_cls);
        for (int i = 0; i < num_cls; i++) {
            clusters[i] = i;
        }
    } else {
        clusters = clusters_to_update;
    }
    
    // Process each specified cluster
    for (int c : clusters) {
        if (c >= num_cls) {
            warning("Skipping invalid cluster index: %d", c);
            continue;
        }
        
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
        
        for (int j = 0; j < const_data.attrisize.length(); ++j) {
            NumericVector col = cluster_data(_, j);
            double sumdelta = sum(col == centers_cluster[j]);
            new_w[j] = const_data.w[j] + nm - sumdelta;
            new_v[j] = const_data.v[j] + sumdelta;   
        }

        sigma[c] = clone(sample_sigma_1_cluster(const_data.attrisize, new_v, new_w));
    }
}

double loglikelihood_Hamming(const internal_state & state, int c, const aux_data & const_data) {
    /**
     * @brief loglikelihood for observation in cluster c 
     * @note could be made that eval it for two cluster at time
     */
    double loglikelihood = 0.0;

    // cluster parameter
    NumericVector center = as<NumericVector>(state.center[c]);
    NumericVector sigma = as<NumericVector>(state.sigma[c]);

    // Compute likelihood 
    for (int i = 0; i < const_data.n; i++) {
        // for observation in cluster c
        if(state.c_i[i] == c){
            for (int j = 0; j < const_data.attrisize.length(); j++) {
                loglikelihood += dhamming(const_data.data(i, j), center[j], sigma[j], const_data.attrisize[j], true);
            }
        }
    }
    
    return loglikelihood;
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
    NumericVector center1 = as<NumericVector>(state.center[c_i_1]);
    NumericVector sigma1 = as<NumericVector>(state.sigma[c_i_1]);

    // Extract cluster of the second observation
    int c_i_2 = state.c_i[i_2];
    NumericVector center2 = as<NumericVector>(state.center[c_i_2]);
    NumericVector sigma2 = as<NumericVector>(state.sigma[c_i_2]);

    // support variables
    NumericVector y_s;
    NumericVector probs(2);
    NumericVector center(const_data.attrisize.length());
    NumericVector sigma(const_data.attrisize.length());
    int cls;
    int n_s_cls;
    
    for (int s : S) {
        // extract datum at s
        y_s = const_data.data(s, _);

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
                Hamming += dhamming(y_s[j], center[j], sigma[j], const_data.attrisize[j], true);
            }

            // Count instances in the cluster excluding the current point s
            n_s_cls = count_cluster_members(state.c_i, s, cls);

            probs[k] =  n_s_cls * std::exp(Hamming);
        }

        // Normalize probabilities
        probs = probs / sum(probs);

        // Sample new cluster assignment between the two clusters of i_1 and i_2
        state.c_i[s] = sample(IntegerVector::create(c_i_1, c_i_2), 1, true, probs)[0];
        update_centers(state, const_data, {c_i_1, c_i_2});
        update_sigma(state.sigma, state.center, state.c_i, const_data, {c_i_1, c_i_2});

    }

    if(debugging){
        DEBUG_PRINT(1, "SPLIT - Parameter update for cluster: {} - {}", c_i_1, c_i_2);
    }
}

int lfact(int n){
    /**
     * @brief logfactorial n
     * @note maybe already exist but faster?
     */
    if (n < 0) {
        std::cerr<<"numero negativo nel fattoriale"<<std::endl;
    }

    double result = 0;
    if (n==0){
        return result;
    }

    do{
        result += std::log(n);
        n--;
    }while(n > 1);
    return result;
}

int cls_elem(const internal_state & state, int c){
    /**
     * @brief number of element in cluster c
     * @note possible upgrade - work only on S indexes
     */
    int f = 0;
    for (int i = 0; i < state.c_i.length(); i++){ 
        if (state.c_i[i] == c) 
            f++;
    }
    return f;
}

double logdensity_hig(double sigmaj, double v, double w, double m){
    /**
     * @brief logdensity of hig(v,w,m)(sigmaj)
     */
    double K = norm_const(w ,v, m); 
    return log(K) - (w + 1)/sigmaj - (v+w)*log(1+exp(-1/sigmaj)*(m-1)) - 2*log(sigmaj);

}

double priors(const internal_state & state, int c, const aux_data & const_data){
    /**
     * @brief prior of the parameters associated to cluster c 
     */

    NumericVector sigma = as<List>(state.sigma)[c];

    double priorg=0;
    for (int j = 0; j < sigma.length(); j++){
        priorg -= log(const_data.attrisize[j]); // densità dell'uniforme è sempre 1/numero_modalità
        priorg += logdensity_hig(sigma[j], const_data.v[j], const_data.w[j], const_data.attrisize[j]);
    }
    return priorg;
}

double logprobgs_phi(const internal_state & gamma_star, const internal_state & gamma, const aux_data & const_data, const int & choosen_idx){
    /**
     * @brief Compute the probability of the cluster dispersion and cluster center - paper reference: P_{GS}(\phi*|phi^L, c^L, y)
     * @param gamma_star state containing the new cluster assignment, new cluster centers, and new cluster dispersions
     * @param gamma state containing the launch cluster assignment, launch cluster centers, and launch cluster dispersions - paper reference: (c^L, \phi^L)
     * @param const_data auxiliary data for the MCMC algorithm containing the input data matrix, attribute sizes, and hypergeometric parameters
     * @param star string to identify the type of operation (split or merge)
     * @param choosen_idx index of the chosen observation
     */
    
    double log_center_prob = 0;
    double log_sigma_prob=0;

    // --------------- Center probs ---------------
    // Compute the uniform probability of the cluster center

    // number of attributes
    int p = const_data.data.ncol();
    
    //NumericMatrix data_tmp = subset_data_for_cluster(const_data.data, gamma_star.c_i[choosen_idx], gamma_star);
    //List prob_centers(Center_prob(data_tmp, gamma_star.sigma[gamma_star.c_i[choosen_idx]], as<NumericVector>(const_data.attrisize)));

    // list of p probalities, each is a vector of probability for center of j-th attribute
    List prob_centers = Center_prob(const_data.data, gamma.sigma[gamma.c_i[choosen_idx]], as<NumericVector>(const_data.attrisize));

    if(debugging){
        DEBUG_PRINT(0, "Proposal");
        Rcpp::Rcout << prob_centers << std::endl;
    }

    // center parameter for which calculate probability
    NumericVector centerstar = gamma_star.center[gamma_star.c_i[choosen_idx]];

    for (int j = 0; j < centerstar.length(); j++){
        NumericVector z = prob_centers[j]; // center attribute probabilities for component j
        log_center_prob += log(z[centerstar[j] - 1]); // probability of center attribute to be the actual
    }
    
    // --------------- Sigma probs ---------------
    // Compute the probability of the cluster dispersion
    
    NumericVector new_w(const_data.attrisize.length());
    NumericVector new_v(const_data.attrisize.length());

    // Create indices for rows in this cluster
    IntegerVector cluster_indices;
    int c = gamma_star.c_i[choosen_idx];
    for (int i = 0; i < gamma_star.c_i.length(); ++i) {
        if (gamma_star.c_i[i] == c){
            cluster_indices.push_back(i);
        }
    }

    // Extract cluster-specific data
    NumericMatrix cluster_data(cluster_indices.length(), const_data.data.ncol());
    for (int i = 0; i < cluster_indices.length(); ++i) {
        cluster_data(i, _) = const_data.data(cluster_indices[i], _);
    } 

    int nm = cluster_indices.length();
    //NumericVector sigmas_cluster = as<NumericVector>(gamma_star.sigma[c]);
    NumericVector centers_cluster = as<NumericVector>(gamma_star.center[c]);
    
    for (int i = 0; i < const_data.attrisize.length(); ++i ){ // for each attribute
        NumericVector col = cluster_data(_, i);
        double sumdelta = sum(col == centers_cluster[i]);
        new_w[i] = const_data.w[i] + nm - sumdelta;
        new_v[i] = const_data.v[i] + sumdelta;   
    }
    
    // eval probability from full conditional
    NumericVector sig = gamma_star.sigma[gamma_star.c_i[choosen_idx]];

    for (int j=0; j<p; j++){
        double temp = logdensity_hig(sig[j], new_v[j], new_w[j], const_data.attrisize[j]);
        log_sigma_prob += temp;
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
    int cls1 = gamma.c_i[i_1];
    NumericVector center1 = as<NumericVector>(gamma.center[cls1]);
    NumericVector sigma1 = as<NumericVector>(gamma.sigma[cls1]);

    // Extract cluster of the second observation
    int cls2 = gamma.c_i[i_2];
    NumericVector center2 = as<NumericVector>(gamma.center[cls2]);
    NumericVector sigma2 = as<NumericVector>(gamma.sigma[cls2]);

    // support variable
    NumericVector y_s;
    NumericVector probs(2);
    NumericVector center(const_data.attrisize.length());
    NumericVector sigma(const_data.attrisize.length());
    int cls;
    int n_s_cls;

    for (int s : S) {
        // extract datum at s
        y_s = const_data.data(s, _);

        // evaluate probabilities
        for (int k = 0; k < 2; k++) {
            // select parameter values of the corresponding cluster
            if(k == 0){
                center = center1;
                sigma = sigma1;
                cls = cls1;
            }
            else{
                center = center2;
                sigma = sigma2;
                cls = cls2;
            }

            double Hamming = 0;

            for (int j = 0; j < y_s.length(); j++) {
                Hamming += dhamming(y_s[j], center[j], sigma[j], const_data.attrisize[j], true);
            }

            // Count instances in the cluster excluding the current point s
            n_s_cls = count_cluster_members(gamma.c_i, s, cls);
            
            probs[k] =  n_s_cls * std::exp(Hamming);
        }

        // Normalize probabilities
        probs = probs / sum(probs);

        int currrent_c = gamma_star.c_i[s] == cls1 ? 0 : 1;

        //  logprob of be assigned to the new state
        logpgs += log(probs[currrent_c]);

    }
    /*
    // Compute the probability of the cluster assignment without the two observations i_1 and i_2
    for (unsigned s = 0; s < S.size(); s++){
        // extract the s-th observation data point
        NumericVector y_s = const_data.data(s, _);

        double num=0, deni=0, denj=0;
        // Compute the number of elements in the cluster s without the s observation
        int ns=count_cluster_members(gamma.c_i, S[s], gamma.c_i[s]);
        
        // Compute the number of elements in the cluster i without the s observation
        int ni=count_cluster_members(gamma.c_i, S[s], gamma.c_i[i_1]);
        
        // Compute the number of elements in the cluster j without the s observation
        int nj=count_cluster_members(gamma.c_i, S[s], gamma.c_i[i_2]);
        
        
        //std::cout<<"ns, ni, nj: " << ns<<" - "<<ni<<" - "<<nj<< std::endl;
        
        // Extract the cluster centers and dispersions for the cluster of s observation
        NumericVector center_s=gamma_star.center[gamma.c_i[s]];
        NumericVector sigma_s=gamma_star.sigma[gamma.c_i[s]];
        
        // Compute the log-Hamming between the s-th observation and the cluster centers
        for (int j=0; j<y_s.length(); j++){
            num+=dhamming(y_s[j], center_s[j], sigma_s[j], const_data.attrisize[j], true);
            deni+=dhamming(y_s[j], center_i[j], sigma_i[j], const_data.attrisize[j], true);
            denj+=dhamming(y_s[j], center_j[j], sigma_j[j], const_data.attrisize[j], true);
        }
        // Compute the probability of the cluster assignment
        pgs*=(ns*std::exp(num))/(ni*std::exp(deni)+nj*std::exp(denj));
        
    }*/
    return logpgs;
}

void select_observations(const internal_state & state, int & i_1, int & i_2, std::vector<int> & S) {
    /**
     * @brief Building S
     * @details select distinct index i_1 and i_2 and populate S with indexes of other observation belonging to the same clusters of the selected
     */

    // number of observations
    int n = state.c_i.size();

    // random uniform sample for (distinct) indexes
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> unif(0, n-1);

    i_1 = unif(gen);
    do {
        i_2 = unif(gen);
    } while (i_2 == i_1);

    // populate S with indexes in selected clusters
    for(int i = 0; i < n; ++i){
        // skip obs_1 and obs_2
        if(i == i_1 || i == i_2)
            continue;

        if(state.c_i[i] == state.c_i[i_1] || state.c_i[i] == state.c_i[i_2]){
            S.push_back(i);
        }
    }

    if(debugging){
        DEBUG_PRINT(0, "Selected observation : {} - {}", i_1, i_2);
        DEBUG_PRINT(0, "Belonging cluster    : {} - {}", state.c_i[i_1], state.c_i[i_2]);
        Rcpp::Rcout << "S: { ";
        for(int s : S){
            Rcpp::Rcout << s << ", ";
        }
        Rcpp::Rcout << "}" << std::endl;
    }

}

internal_state split_launch_state(const std::vector<int> & S, const internal_state & state, int i_1, int i_2, int t, const aux_data & const_data){
    /**
     * @brief build split launch state
     */

    // gamma split L
	IntegerVector c_L_split = clone(state.c_i);
	List center_L_split = clone(state.center);
	List sigma_L_split = clone(state.sigma);

    IntegerVector S_indexes = wrap(S); // R version for indexing

    // ----- gamma split launch popolation -----
    if(state.c_i[i_1] == state.c_i[i_2]){
        // set the allocation of i_1 to latent cluster -> (last cluster + 1)
        c_L_split[i_1] = unique_classes(state.c_i).length();
        // add space for the parameters of the new cluster
        center_L_split.push_back(0);
        sigma_L_split.push_back(0);
    }

    // randomly allocate with equal probs data in S between cls-1 and cls-2
    c_L_split[S_indexes] = sample(IntegerVector::create(c_L_split[i_1], c_L_split[i_2]), S_indexes.length(), true);

    // draw a new value for the center and sigma of clusters from their prior
    center_L_split[c_L_split[i_1]] = sample_center_1_cluster(const_data.attrisize);
    sigma_L_split[c_L_split[i_1]] = sample_sigma_1_cluster(const_data.attrisize, const_data.v, const_data.w);
    center_L_split[c_L_split[i_2]] = sample_center_1_cluster(const_data.attrisize);
    sigma_L_split[c_L_split[i_2]] = sample_sigma_1_cluster(const_data.attrisize, const_data.v, const_data.w);
    
    // aux state for the split launch
    internal_state state_launch_split = {c_L_split, center_L_split, sigma_L_split, static_cast<int>(unique_classes(c_L_split).length())};

    if(debugging){
        DEBUG_PRINT(0, "SPLIT - Split launch state");
        print_internal_state(state_launch_split, 1);
    }

    // clean split state, reset indexing
    clean_var(state_launch_split, state_launch_split, unique_classes(state_launch_split.c_i), const_data.attrisize);

    if(debugging){
        DEBUG_PRINT(0, "SPLIT - Split launch state, after clean_var");
        print_internal_state(state_launch_split);
    }
    
    // Intermediate restricted Gibbs Sampler on gamma split launch 
    for(int iter = 0; iter < t; ++iter ){
        split_restricted_gibbs_sampler(S, state_launch_split, i_1, i_2, const_data);
    }

    if(debugging){
        DEBUG_PRINT(0, "SPLIT - Split launch state, after t restricted Gibbs scan");
        print_internal_state(state_launch_split);
    }

    return state_launch_split;
}

internal_state merge_launch_state(const std::vector<int> & S, const internal_state & state, int i_1, int i_2, int r, const aux_data & const_data){
    /**
     * @brief build merge launch state
     */

    // gamma merge L
	IntegerVector c_L_merge = clone(state.c_i);
	List center_L_merge = clone(state.center);
	List sigma_L_merge = clone(state.sigma);

    IntegerVector S_indexes = wrap(S); // R version for indexing

    // ----- gamma merge popolation -----
    if(state.c_i[i_1] != state.c_i[i_2]){
        // set the allocation of i_1 equal to the cls of i_2 (c_j)
        c_L_merge[i_1] = c_L_merge[i_2];
        // allocate all the data in S to the cls of i_2
        c_L_merge[S_indexes] = c_L_merge[i_2]; 
    }

    // draw a new value for the center and sigma of cluster from their prior
    center_L_merge[c_L_merge[i_2]] = sample_center_1_cluster(const_data.attrisize);
    sigma_L_merge[c_L_merge[i_2]] = sample_sigma_1_cluster(const_data.attrisize, const_data.v, const_data.w);

    // aux state for the merge launch
    internal_state state_launch_merge = {c_L_merge, center_L_merge, sigma_L_merge, static_cast<int>(unique_classes(c_L_merge).length())};

    if(debugging){
        DEBUG_PRINT(0, "MERGE - Merge launch state");
        print_internal_state(state_launch_merge, 1);
    }

    // clean merge state 
    clean_var(state_launch_merge, state_launch_merge, unique_classes(state_launch_merge.c_i), const_data.attrisize);

    if(debugging){
        DEBUG_PRINT(0, "MERGE - Merge launch state, after clean");
        print_internal_state(state_launch_merge);
    }

    if(debugging){
        DEBUG_PRINT(1, "MERGE - Parameter update for cluster: {}", state_launch_merge.c_i[i_2]);
    }

    for(int iter = 0; iter < r; ++iter ){
        // update only merge cls center and sigma 
        update_centers(state_launch_merge, const_data, {state_launch_merge.c_i[i_2]});
        update_sigma(state_launch_merge.sigma, state_launch_merge.center, state_launch_merge.c_i, const_data, {state_launch_merge.c_i[i_2]});
    }

    if(debugging){
        DEBUG_PRINT(0, "MERGE - Merge launch state, after r restricted Gibbs scan");
        print_internal_state(state_launch_merge);
    }
    
    return state_launch_merge;
}

double split_acc_prob(const internal_state & state_split, const internal_state & state, const internal_state & split_launch, const internal_state & merge_launch, const std::vector<int> & S, int i_1, int i_2, const aux_data & const_data){
    /**
     * @brief evaluate the acceptance probability for the proposed split state
     */

    double alpha = const_data.gamma;

    // evaluate prior ratio         
    double logp = 0;
    logp += log(alpha);
    // numerator
    logp += lfact(cls_elem(state_split, state_split.c_i[i_1]) - 1);
    logp += lfact(cls_elem(state_split, state_split.c_i[i_2]) - 1);
    logp += priors(state_split, state_split.c_i[i_1], const_data);
    logp += priors(state_split, state_split.c_i[i_2], const_data);
    // denominator
    logp -= lfact(cls_elem(state, state.c_i[i_1]) - 1);
    logp -= priors(state, state.c_i[i_1], const_data);

    if(debugging){
        DEBUG_PRINT(1, "SPLIT - prior Logratio: {}", logp);
        DEBUG_PRINT(1, "SPLIT - prior ratio: {}", exp(logp));
    }
    

    // evaluate likelihood ratio
    double logl = 0;
    // numerator
    logl += loglikelihood_Hamming(state_split, state_split.c_i[i_1], const_data);
    logl += loglikelihood_Hamming(state_split, state_split.c_i[i_2], const_data);
    // denominator
    logl -= loglikelihood_Hamming(state, state.c_i[i_1], const_data);

    if(debugging){
        DEBUG_PRINT(1, "SPLIT - likelihood Logratio: {}", logl);
        DEBUG_PRINT(1, "SPLIT - likelihood ratio: {}", exp(logl)); 
    } 

    // evaluate proposal ratio
    double logq = 0;
    // numerator
    logq += logprobgs_phi(state, merge_launch, const_data, i_1);
    // denominator
    logq -= logprobgs_phi(state_split, split_launch, const_data, i_1);
    logq -= logprobgs_phi(state_split, split_launch, const_data, i_2);
    logq -= logprobgs_c_i(state_split, split_launch, const_data, S, i_1, i_2);

    if(debugging){
        DEBUG_PRINT(1, "SPLIT - proposal Logratio: {}", logq);
        DEBUG_PRINT(1, "SPLIT - proposal ratio: {}", exp(logq)); 
    } 
    
    double tot_ratio = logp + logl + logq;

    if(debugging){
        DEBUG_PRINT(0, "SPLIT - final ratio: {}",exp(tot_ratio));
    }

    return std::min(exp(tot_ratio), 1.0);
}

double merge_acc_prob(const internal_state & state_merge, const internal_state & state, const internal_state & split_launch, const internal_state & merge_launch, const std::vector<int> & S, int i_1, int i_2, const aux_data & const_data){
    /**
     * @brief evaluate the acceptance probability for the proposed merge state
     */

    double alpha = const_data.gamma;

    // evaluate prior ratio         
    double logp = 0;
    // numerator
    logp += lfact(cls_elem(state_merge, state_merge.c_i[i_2]) - 1);
    logp += priors(state_merge, state_merge.c_i[i_2], const_data);
    // denominator
    logp -= log(alpha);
    logp -= lfact(cls_elem(state, state.c_i[i_1]) - 1);
    logp -= lfact(cls_elem(state, state.c_i[i_2]) - 1);
    logp -= priors(state, state.c_i[i_1], const_data);
    logp -= priors(state, state.c_i[i_2], const_data);

    if(debugging){
        DEBUG_PRINT(1, "MERGE - prior Logratio: {}", logp);
        DEBUG_PRINT(1, "MERGE - prior ratio: {}", exp(logp));
    }
    // evaluate likelihood ratio
    double logl = 0;
    // numerator
    logl += loglikelihood_Hamming(state_merge, state_merge.c_i[i_2], const_data);
    // denominator
    logl -= loglikelihood_Hamming(state, state.c_i[i_1], const_data);
    logl -= loglikelihood_Hamming(state, state.c_i[i_2], const_data);

    if(debugging){
        DEBUG_PRINT(1, "MERGE - likelihood Logratio: {}", logl);
        DEBUG_PRINT(1, "MERGE - likelihood ratio: {}", exp(logl));
    }

    // evaluate proposal ratio
    double logq = 0;
    // numerator
    logq += logprobgs_phi(state, split_launch, const_data, i_1);
    logq += logprobgs_phi(state, split_launch, const_data, i_2);
    logq += logprobgs_c_i(state, split_launch, const_data, S, i_1, i_2);
    // denominator
    logq -= logprobgs_phi(state_merge, merge_launch, const_data, i_2);

    if(debugging){
        DEBUG_PRINT(1, "MERGE - proposal Logratio: {}", logq);
        DEBUG_PRINT(1, "MERGE - proposal ratio: {}", exp(logq));
    }
    double tot_ratio = logp + logl + logq;

    if(debugging){
        DEBUG_PRINT(0, "MERGE - final ratio: {}",exp(tot_ratio));
    }

    return std::min(exp(tot_ratio), 1.0);
}

void split_and_merge(internal_state & state, const aux_data & const_data, int t, int r, double & acpt_ratio, int & accepted, int & split_n, int & merge_n) {
    /**
     * @brief Split and merge step
     * @details This function implements the split and merge step of the MCMC algorithm
     */

    // --------------- Step 1 ---------------
    // choose 2 observation random from the data
    int i_1;
    int i_2;

    // --------------- Step 2 ---------------
    // Create S the set of idx of obs in the same cluster of obs_1 or obs_2
    // REMIND: obs_1 and obs_2 aren't in S
    std::vector<int> S;

    // perform Step 1 , Step 2
    select_observations(state, i_1, i_2, S);

	// --------------- Step 3 ---------------
	// Launch states creation
    internal_state split_launch = split_launch_state(S, state, i_1, i_2, t, const_data);
    internal_state merge_launch = merge_launch_state(S, state, i_1, i_2, r, const_data);

    // --------------- Step 4&5 ---------------

    // Aux var to store *-state
    internal_state state_star = {IntegerVector(), List(), List(), 0};
    acpt_ratio = .999;

    if(state.c_i[i_1] == state.c_i[i_2]){
        
        if(debugging){
            DEBUG_PRINT(0, "SPLIT - propose");
        }
        
        state_star = split_launch;

        if(debugging){
            DEBUG_PRINT(1, "SPLIT - state before");
            print_internal_state(state_star);
        }

        // ----- (a) - last Restricted Gibbs Sampler -----
        split_restricted_gibbs_sampler(S, state_star, i_1, i_2, const_data);

        if(debugging){
            DEBUG_PRINT(1, "SPLIT - state after");
            print_internal_state(state_star);
        }

        // ----- (b) - Transition probabilities ----- 
        acpt_ratio = split_acc_prob(state_star, state, split_launch, merge_launch, S, i_1, i_2, const_data);
        split_n++;
    }    
    else{

        if(debugging){
            DEBUG_PRINT(0, "MERGE - propose");
        }
        
        state_star = merge_launch;
        // ----- (a) - last restricted Gibbs Sampling to update merge cls parameters -----
        update_centers(state_star, const_data, {state_star.c_i[i_2]});
        update_sigma(state_star.sigma, state_star.center, state_star.c_i, const_data, {state_star.c_i[i_2]});

        // ----- (b) - transition probabilities -----
        acpt_ratio = merge_acc_prob(state_star, state, split_launch, merge_launch, S, i_1, i_2, const_data);

        merge_n++;        
    }
    
    // ----- (c) - Metropolis-Hastings step -----
    // sample if accept or not the MC state stored in c_star
    if(R::runif(0,1) < acpt_ratio){
        clean_var(state, state_star, unique_classes(state_star.c_i), const_data.attrisize); // per sicurezza
        state = state_star;
        if(debugging){
            DEBUG_PRINT(0, "ACCEPTED");
        }
        accepted++;
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

// [[Rcpp::export]]
List run_markov_chain(NumericMatrix data, IntegerVector attrisize, double gamma, NumericVector v, NumericVector w, 
                    int verbose = 0, int m = 5, int iterations = 1000, int L = 1, 
                    Rcpp::Nullable<Rcpp::IntegerVector> c_i = R_NilValue, int burnin = 5000, int t = 10, int r = 10, bool neal8=false, bool split_merge = true) {
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
    state.c_i = clone(initial_c_i);
    
    // Initialize centers
    state.center = sample_centers(state.total_cls, const_data.attrisize);
    // Initialize sigma
    state.sigma = sample_sigmas(state.total_cls, const_data);

    if(verbose == 2 or verbose == 1){
        print_internal_state(state);
    }

    List results = List::create(Named("total_cls") = List(iterations),
                                Named("c_i") = List(iterations),
                                Named("centers") = List(iterations),
                                Named("sigmas") = List(iterations), 
                                Named("loglikelihood") = NumericVector(iterations), 
                                Named("acceptance_ratio") = NumericVector(iterations),
                                Named("accepted") = IntegerVector(iterations),
                                Named("split_n") = IntegerVector(iterations),
                                Named("merge_n") = IntegerVector(iterations),
                                Named("final_ass") = IntegerVector(data.nrow())                           
                                );

    auto start_time = std::chrono::high_resolution_clock::now();
    Rcpp::Rcout << "Starting Markov Chain sampling..." << std::endl;

    double acpt_ratio = 0.0;
    int accepted = 0; 
    int split_n = 0;
    int merge_n = 0;

    for (int iter = 0; iter < iterations + burnin; iter++) {
        if(verbose != 0)
            std::cout << std::endl <<"[DEBUG] - Iteration " << iter << " of " << iterations << std::endl;

        // Sample new cluster assignments for each observation
        if(neal8){
            for (int index_i = 0; index_i < const_data.n; index_i++) {
                L = unique_classes(state.c_i).length();
                IntegerVector unique_classes_without_i = unique_classes_without_index(state.c_i, index_i);
                
                // Sample new cluster assignment for observation i
                sample_allocation(index_i, const_data, state, m, unique_classes_without_i);       
            } 
            
            // Update centers and sigmas
            update_centers(state, const_data);
            update_sigma(state.sigma, state.center, state.c_i, const_data);
        }

        if(verbose == 2){
            std::cout << "State after Neal8" << std::endl;
            print_internal_state(state);
        }

        // Split and merge step
        if(split_merge){
            split_and_merge(state, const_data, t, r, acpt_ratio, accepted, split_n, merge_n);
        }

        if(verbose == 2){
            std::cout << "State after Split and Merge" << std::endl;
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
            as<NumericVector>(results["acceptance_ratio"])[iter - burnin] = acpt_ratio;
            as<IntegerVector>(results["accepted"])[iter - burnin] = accepted;
            as<IntegerVector>(results["split_n"])[iter - burnin] = split_n;
            as<IntegerVector>(results["merge_n"])[iter - burnin] = merge_n;
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    Rcpp::Rcout << std::endl << "Markov Chain sampling completed in: "<< duration.count() << " s " << std::endl;

    results["final_ass"] = state.c_i;
    
    return results;
}