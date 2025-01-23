#ifndef COMMON_FUNCTIONS_H
#define COMMON_FUNCTIONS_H

#include <Rcpp.h>
#include <random>
#include <algorithm>
#include <vector>
#include <set>

#include <Rcpp.h>
#include <RcppGSL.h>  // Add this explicit RcppGSL include
#include <Rinternals.h>
#include <gsl/gsl_sf_hyperg.h>

#include <gibbs_utility.cpp>
#include <hyperg.cpp>

using namespace Rcpp;

bool debug_var = false;

// Debug utilities
namespace debug {
    template<typename... Args>
    void print(unsigned int tab_level, const char* func_name, int line, const std::string& message, const Args&... args) {
        /**
         * @brief Print a debug message to the console
         * @param tab_level Number of tabs to indent the message
         * @param func_name Name of the function where the message is printed
         * @param line Line number where the message is printed
         * @param message Message to print
         * @param args Additional arguments to format the message
         * @note This function is used to print debug messages to the console
         *      with additional information such as the function name and line number
         */
        std::stringstream ss;
        for(unsigned int i = 0; i < tab_level; ++i) {
            ss << '\t';
        }
        ss << "[DEBUG:" << func_name << ":" << line << "] ";
        
        size_t pos = 0;
        size_t count = 0;
        std::string temp = message;
        
        ((ss << temp.substr(pos, message.find("{}", pos) - pos) 
            << args
            , pos = message.find("{}", pos) + 2
            , count++), ...);
            
        ss << message.substr(pos);
        if(debug_var)
            Rcpp::Rcout << ss.str() << std::endl;
    }
}

#define DEBUG_PRINT(level, message, ...) \
    debug::print(level, __func__, __LINE__, message, ##__VA_ARGS__)

// Data structures
struct internal_state {
    IntegerVector c_i;
    List center;
    List sigma;
    int total_cls = 0;
};

struct aux_data {
    NumericMatrix data;
    int n;
    IntegerVector attrisize;
    double gamma;
    NumericVector v;
    NumericVector w;
};

// Utility functions
inline double log_sum_exp(const std::vector<double>& log_values) {
    /**
     * @brief Compute the log sum of exponentials
     * @param log_values Vector of log values
     * @return Log sum of exponentials
     * @note This function is used to compute the log sum of exponentials
     *      for numerical stability in log-space calculations
     */
    if (log_values.empty()) return -std::numeric_limits<double>::infinity();
    double max_val = *std::max_element(log_values.begin(), log_values.end());
    double sum = 0.0;
    for (double log_val : log_values) {
        sum += std::exp(log_val - max_val);
    }
    return max_val + std::log(sum);
}

// Print functions
void print_internal_state(const internal_state& state, int interest = -1) {
    /**
     * @brief Print internal state of the MCMC sampler
     * @param state Internal state of the MCMC sampler
     * @param interest Level of detail to print (default: -1)
     * @note This function is used to print the internal state of the MCMC sampler
     *      for debugging and visualization purposes
     */
    if(debug_var){
        if(interest == 1 || interest == -1){
            Rcpp::Rcout << "Cluster assignments: " << std::endl << "\t";
            Rcpp::Rcout << state.c_i << std::endl;
        }

        if(interest == 2 || interest == -1){
            Rcpp::Rcout << "Centers: " << std::endl;
            for (int i = 0; i < state.center.length(); i++) {
                Rcpp::Rcout << "\tCluster "<< i << " :" << std::setprecision(5) 
                        << as<NumericVector>(state.center[i]) << std::endl;
            }
        }

        if(interest == 3 || interest == -1){
            Rcpp::Rcout << "Dispersions: " << std::endl;
            for (int i = 0; i < state.sigma.length(); i++) {
                Rcpp::Rcout << "\tCluster "<< i << ": " 
                        << as<NumericVector>(state.sigma[i]) << std::endl;
            }
        }
    }
}

void print_progress_bar(int progress, int total, const std::chrono::steady_clock::time_point& start_time) {
    const int bar_width = 50;
    const double ratio = static_cast<double>(progress) / total;
    const int bar_progress = static_cast<int>(bar_width * ratio);
    
    // Calculate elapsed time
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
    
    // Improved time estimation using exponential moving average
    static double avg_time_per_step = 0.0;
    const double alpha = 0.1; // smoothing factor
    
    if (progress > 1) {
        double current_time_per_step = elapsed.count() / static_cast<double>(progress);
        avg_time_per_step = (alpha * current_time_per_step) + ((1 - alpha) * avg_time_per_step);
    } else {
        avg_time_per_step = elapsed.count();
    }
    
    // Calculate remaining time using the smoothed average
    double remaining_seconds = avg_time_per_step * (total - progress);
    double elapsed_seconds = elapsed.count();
    
    // Format time with improved precision
    auto format_time = [](double seconds) -> std::string {
        int hours = static_cast<int>(seconds) / 3600;
        int minutes = (static_cast<int>(seconds) % 3600) / 60;
        int secs = static_cast<int>(seconds) % 60;
        
        std::ostringstream time_str;
        if (hours > 0) {
            time_str << hours << "h ";
        }
        if (minutes > 0 || hours > 0) {
            time_str << minutes << "m ";
        }
        time_str << secs << "s";
        return time_str.str();
    };
    
    // Construct progress bar with speed indicator
    std::ostringstream bar;
    bar << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < bar_progress) bar << "=";
        else if (i == bar_progress) bar << ">";
        else bar << " ";
    }
    
    // Add percentage and improved time information
    bar << "] " << std::fixed << std::setprecision(1) << (ratio * 100.0) << "% | ";
    
    // Only show remaining time if we have made some progress
    if (progress > 0 && progress < total) {
        bar << "ETA: " << format_time(remaining_seconds);
        bar << " | " << std::fixed << std::setprecision(1) 
            << (progress / elapsed_seconds) << " it/s"; // iterations per second
        bar << " | Elapsed: " << format_time(elapsed_seconds);
    } else if (progress == total) {
        bar << "Completed in: " << format_time(elapsed_seconds);
    }
    
    // Print the progress bar
    REprintf("%s", bar.str().c_str());
    
    // Add newline when complete
    if (progress == total) REprintf("\n");
    
    // Ensure output is displayed immediately
    R_FlushConsole();
}

// Initialization functions
IntegerVector sample_initial_assignment(double K = 4, int n = 10) {
    /**
     * @brief Sample initial cluster assignments
     * @param K Number of clusters
     * @param n Number of observations
     */
    IntegerVector cluster_assignments = Rcpp::sample(K, n, true); 
    cluster_assignments = cluster_assignments - 1; 
    return cluster_assignments;
}

NumericVector sample_center_1_cluster(const IntegerVector & attrisize, const List & probs = List()) {
    /**
     * @brief Sample center for a single cluster
     * @param attrisize Vector of attribute sizes
     * @return NumericVector containing the sampled center
     */
    NumericVector center(attrisize.length());
    for (int j = 0; j < attrisize.length(); j++) {
        if(probs.length() > 0){
            IntegerVector attribute_vec = seq_len(attrisize[j]);
            center[j] = sample(attribute_vec, 1, true, as<NumericVector>(probs[j]))[0];
        }
        
        else 
            center[j] = sample(attrisize[j], 1, true)[0];
    } 
    return center;
}

List sample_centers(const int number_cls, const IntegerVector & attrisize) {
    /**
     * @brief Sample centers for all clusters
     * @param number_cls Number of clusters
     * @param attrisize Vector of attribute sizes
     * @return List containing the sampled centers
     */
    List centers(number_cls);
    for (int c = 0; c < number_cls; c++) {
        centers[c] = sample_center_1_cluster(attrisize);
    }   
    return centers;
}

NumericVector sample_sigma_1_cluster(const IntegerVector & attrisize, 
                                   const NumericVector & v, 
                                   const NumericVector & w) {
    /**
     * @brief Sample sigma for a single cluster
     * @param attrisize Vector of attribute sizes
     * @param v Vector of v parameters
     * @param w Vector of w parameters
     * @return NumericVector containing the sampled sigma
     */

    NumericVector sigma(attrisize.length());
    for (int j = 0; j < attrisize.length(); j++) {
        sigma[j] = rhyper_sig(1, w[j], v[j], attrisize[j])[0];
    } 
    return sigma;
}

List sample_sigmas(const int number_cls, const aux_data & const_data) {
    /**
     * @brief Sample sigmas for all clusters
     * @param number_cls Number of clusters
     * @param const_data Auxiliary data for the MCMC algorithm
     * @return List containing the sampled sigmas
     */
    List sigma(number_cls);
    for (int i = 0; i < number_cls; i++) {
        sigma[i] = sample_sigma_1_cluster(const_data.attrisize, const_data.v, const_data.w);
    } 
    return sigma;
}

IntegerVector unique_classes(const IntegerVector & c_i) {
    /**
     * @brief Get unique classes from cluster assignments
     * @param c_i Cluster assignments
     * @return IntegerVector containing unique classes
     */
    IntegerVector unique_vec = unique(c_i);
    std::sort(unique_vec.begin(), unique_vec.end());
    return unique_vec;
}

IntegerVector unique_classes_without_index(const IntegerVector & c_i, const int index_to_del) {
    /**
     * @brief Get unique classes excluding a specific index
     * @param c_i Cluster assignments
     * @param index_to_del Index to exclude
     * @return IntegerVector containing unique classes
     */
    std::set<double> unique_classes;
    for (int i = 0; i < c_i.length(); ++i) {
        if (i != index_to_del) {
            unique_classes.insert(c_i[i]);
        }
    }
    return wrap(std::vector<double>(unique_classes.begin(), unique_classes.end()));
}

int count_cluster_members(const IntegerVector& c_i, int exclude_index, int cls) {
    /**
     * @brief Count members of a cluster excluding a specific index
     * @param c_i Cluster assignments
     * @param exclude_index Index to exclude
     * @param cls Cluster index
     * @return Number of members in the cluster
     */

    if (exclude_index < 0 || exclude_index >= c_i.length()) {
        Rcpp::warning("Exclude index %d is out of bounds for vector of length %d", 
                     exclude_index, c_i.length());
        return 0;
    }
    
    int n_i_z = 0;
    for (int i = 0; i < c_i.length(); i++) {
        if (i != exclude_index && c_i[i] == cls) {
            n_i_z++;
        }
    }
    return n_i_z;
}

void clean_var(internal_state & updated_state, 
              const internal_state & current_state, 
              const IntegerVector& existing_cls, 
              const IntegerVector& attrisize) {
    /**
     * @brief Clean variables and update state
     * @param updated_state Updated internal state
     * @param current_state Current internal state
     * @param existing_cls Existing cluster indices
     * @param attrisize Vector of attribute sizes
     * @note This function is used to clean variables and update the internal state
     *     after sampling new cluster assignments
     */
    int num_existing_cls = existing_cls.length();
    std::unordered_map<int, int> cls_to_new_index;
    
    for(int i = 0; i < num_existing_cls; ++i) {
        int idx_temp = 0;
        if(existing_cls[i] < num_existing_cls){
            cls_to_new_index[existing_cls[i]] = existing_cls[i];
        } else {
            while(cls_to_new_index.find(idx_temp) != cls_to_new_index.end() 
                  && idx_temp < num_existing_cls){
                idx_temp++;
            }
            cls_to_new_index[existing_cls[i]] = idx_temp;
        }
    }
    
    List new_center(num_existing_cls);
    List new_sigma(num_existing_cls);

    for(int i = 0; i < num_existing_cls; ++i) {
        new_center[cls_to_new_index[existing_cls[i]]] = current_state.center[existing_cls[i]];
        new_sigma[cls_to_new_index[existing_cls[i]]] = current_state.sigma[existing_cls[i]];
    }

    updated_state.center = std::move(new_center);
    updated_state.sigma = std::move(new_sigma);
    updated_state.total_cls = num_existing_cls;

    for(int i = 0; i < current_state.c_i.length(); i++) {
        auto it = cls_to_new_index.find(current_state.c_i[i]);
        if(it != cls_to_new_index.end()) {
            updated_state.c_i[i] = it->second;
        }
    }
}

double compute_loglikelihood(internal_state & state, aux_data & const_data) {
    /**
     * @brief Compute log-likelihood of the current state
     * @param state Internal state of the MCMC sampler
     * @param const_data Auxiliary data for the MCMC algorithm
     * @return Log-likelihood of the current state
     */

    double loglikelihood = 0.0;
    for (int i = 0; i < const_data.n; i++) {
        int cluster = state.c_i[i];
        NumericVector center = as<NumericVector>(state.center[cluster]);
        NumericVector sigma = as<NumericVector>(state.sigma[cluster]);
        
        for (int j = 0; j < const_data.attrisize.length(); j++) {
            loglikelihood += dhamming(const_data.data(i, j), 
                                    center[j], 
                                    sigma[j], 
                                    const_data.attrisize[j], 
                                    true);
        }
    }
    return loglikelihood;
}

NumericMatrix subset_data_for_cluster(const NumericMatrix & data, int cluster, const internal_state & state) {
    /**
     * @brief Subset data for a specific cluster
     * @param data Input data matrix
     * @param cluster Cluster index
     * @param state Internal state of the MCMC sampler
     * @return NumericMatrix containing the subset of data
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
     * @param state Internal state of the MCMC sampler
     * @param const_data Auxiliary data for the MCMC algorithm
     * @param cluster_indexes Vector of cluster indexes to update (default: empty)
     * @note This function is used to update cluster centers based on the current state
     *    and the input data
     */
    IntegerVector clusters = unique_classes(state.c_i);
    int num_cls = state.total_cls;
    List prob_centers;

    NumericVector attr_centers(const_data.attrisize.length());
    List prob_centers_cluster;


    // Update all clusters if none specified
    if (cluster_indexes.size() == 0) {
        for (int i = 0; i < num_cls; i++) {
            NumericMatrix data_tmp = subset_data_for_cluster(const_data.data, i, state);
            prob_centers = Center_prob(data_tmp, state.sigma[i], as<NumericVector>(const_data.attrisize));
            state.center[i] = sample_center_1_cluster(const_data.attrisize, prob_centers);
        }
    } else {
        // Update only specified clusters
        for (int idx : cluster_indexes) {
            if (idx >= 0 && idx < num_cls) {
                NumericMatrix data_tmp = subset_data_for_cluster(const_data.data, idx, state);
                prob_centers = Center_prob(data_tmp, state.sigma[idx], as<NumericVector>(const_data.attrisize));
                state.center[idx] = sample_center_1_cluster(const_data.attrisize, prob_centers);
            }
        }
    }
}

void update_sigma(List & sigma, const List & centers, const IntegerVector & c_i, 
                 const aux_data & const_data, std::vector<int> clusters_to_update = {}) {
    /**
     * @brief Update cluster sigmas
     * @param sigma List of sigmas
     * @param centers List of centers
     * @param c_i Cluster assignments
     * @param const_data Auxiliary data for the MCMC algorithm
     * @param clusters_to_update Vector of cluster indexes to update (default: empty)
     * @note This function is used to update cluster sigmas based on the current state
     *   and the input data
     */
    
    int num_cls = sigma.length();
    NumericVector new_w(const_data.attrisize.length());
    NumericVector new_v(const_data.attrisize.length());
    
    // Determine clusters to process
    std::vector<int> clusters;
    if (clusters_to_update.size() == 0) {
        clusters.resize(num_cls);
        for (int i = 0; i < num_cls; i++) {
            clusters[i] = i;
        }
    } else {
        clusters = clusters_to_update;
    }
    
    // Process each cluster
    for (int c : clusters) {
        if (c >= num_cls) {
            warning("Skipping invalid cluster index: %d", c);
            continue;
        }
        
        // Get cluster data
        IntegerVector cluster_indices;
        for (int i = 0; i < c_i.length(); ++i) {
            if (c_i[i] == c) {
                cluster_indices.push_back(i);
            }
        }
        
        NumericMatrix cluster_data(cluster_indices.length(), const_data.data.ncol());
        for (int i = 0; i < cluster_indices.length(); ++i) {
            cluster_data(i, _) = const_data.data(cluster_indices[i], _);
        }
        
        int nm = cluster_indices.length();
        NumericVector centers_cluster = as<NumericVector>(centers[c]);
        
        // Update parameters
        for (int j = 0; j < const_data.attrisize.length(); ++j) {
            NumericVector col = cluster_data(_, j);
            double sumdelta = sum(col == centers_cluster[j]);
            new_w[j] = const_data.w[j] + nm - sumdelta;
            new_v[j] = const_data.v[j] + sumdelta;   
        }

        // Sample new sigmas
        sigma[c] = clone(sample_sigma_1_cluster(const_data.attrisize, new_v, new_w));
    }
}

#endif