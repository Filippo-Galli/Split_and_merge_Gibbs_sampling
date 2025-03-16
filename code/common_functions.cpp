#include "common_functions.hpp"
#include "hyperg.hpp"

bool debug_var = true;

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

// Print functions
void print_internal_state(const internal_state& state, int interest) {
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
        time_str << secs << "s"; // spazio per evitare che rimangano scritte vecchie
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

void validate_state(const internal_state& state, const std::string& message) {
    
    // Ensure consistency between c_i and total_cls
    IntegerVector unique_cls = unique_classes(state.c_i);
    if (unique_cls.length() != state.total_cls) {
        std::string error_message = "State validation failed: inconsistent cluster count from " + message;
        print_internal_state(state, 1);
        Rcpp::Rcout << "Unique classes: " << unique_cls << std::endl;
        Rcpp::Rcout << "Total classes: " << state.total_cls << std::endl;
        Rcpp::stop(error_message);
    }
    
    // Validate center and sigma lists
    if (state.center.length() != state.total_cls || 
        state.sigma.length() != state.total_cls) {
        std::string error_message = "State validation failed: inconsistent parameter lengths from " + message;
        print_internal_state(state, -1);
        Rcpp::Rcout << "Unique classes: " << unique_cls << std::endl;
        Rcpp::Rcout << "Total classes: " << state.total_cls << std::endl;
        Rcpp::stop(error_message);
    }
    
}

// Initialization functions
IntegerVector sample_initial_assignment(double K, int n) {
    /**
     * @brief Sample initial cluster assignments
     * @param K Number of clusters
     * @param n Number of observations
     */
    IntegerVector cluster_assignments = Rcpp::sample(K, n, true); 
    cluster_assignments = cluster_assignments - 1; 
    return cluster_assignments;
}

NumericVector sample_center_1_cluster(const IntegerVector & attrisize, const List & probs) {
    /**
     * @brief Sample center for a single cluster
     * @param attrisize Vector of attribute sizes
     * @return NumericVector containing the sampled center
     */
    NumericVector center(attrisize.length());
    for (int j = 0; j < attrisize.length(); j++) {
        if(probs.length() > 0){
            const IntegerVector & attribute_vec = seq_len(attrisize[j]);
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
        //sigma[j] = rhyper_sig(1, w[j], v[j], attrisize[j])[0];
        sigma[j] = rhig(1, v[j], w[j], attrisize[j])[0];
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
    
    return as<IntegerVector>(unique(c_i).sort());
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

    // Create a LogicalVector from the comparison
    //const LogicalVector & mask = (seq_len(c_i.length()) - 1) != index_to_del;
    //// Materialize the subset before passing to unique
    //const IntegerVector & subset = c_i[mask];
    //return unique(subset);
}

int count_cluster_members(const IntegerVector& c_i, int exclude_index, int cls) {
    /**
     * @brief Count members of a cluster excluding a specific index
     * @param c_i Cluster assignments
     * @param exclude_index Index to exclude
     * @param cls Cluster index
     * @return Number of members in the cluster
     */
   
    int n_i_z = 0;
    for (int i = 0; i < c_i.length(); i++) {
        if (i != exclude_index && c_i[i] == cls) {
            n_i_z++;
        }
    }
    return n_i_z;
}

void clean_var(internal_state & updated_state, 
              const internal_state current_state, 
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

    validate_state(updated_state, "clean_var");
}

double dhamming_pippo(int x, int c, double s, int attrisize) {
    /**
     * @brief Compute Log-Hamming distance between two values
     * @param x First value
     * @param c Second value
     * @param s Sigma parameter
     * @param attrisize Attribute size
     * @return Hamming distance between x and c
     */

    // Convert boolean to integer safely
    int diff = 1 - (x == c);
    double numerator = - diff / s;
    
    // Precompute common terms
    double exp_term = exp(1.0/s);
    double attr_ratio = (attrisize - 1.0) / exp_term;
    double denominator = log(1.0 + attr_ratio);
    
    return numerator - denominator; 
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
        const int cluster = state.c_i[i];
        const NumericVector & center = as<NumericVector>(state.center[cluster]);
        const NumericVector & sigma = as<NumericVector>(state.sigma[cluster]);
        
        for (int j = 0; j < const_data.attrisize.length(); j++) {
            loglikelihood += dhamming_pippo(const_data.data(i, j), 
                                    center[j], 
                                    sigma[j], 
                                    const_data.attrisize[j]);
        }
    }
    return loglikelihood;
}

NumericMatrix subset_data_for_cluster(const NumericMatrix& data, 
                                              int cluster, 
                                              const internal_state& state) {
    /**
     * @brief Subset data for a specific cluster
     * @param data NumericMatrix containing the data
     * @param cluster Cluster index
     * @param state Internal state of the MCMC sampler
     * @return NumericMatrix containing the subset of data
     * @note This function is used to subset the data for a specific cluster
     *    based on the current state of the MCMC sampler
     */

    // Pre-count cluster size
    int cluster_size = 0;
    for (int i = 0; i < state.c_i.length(); ++i) {
        if (state.c_i[i] == cluster) {
            ++cluster_size;
        }
    }
    
    // Pre-allocate matrix
    NumericMatrix cluster_data(cluster_size, data.ncol());
    int curr_idx = 0;
    
    // Single pass to fill data
    for (int i = 0; i < state.c_i.length(); ++i) {
        if (state.c_i[i] == cluster) {
            cluster_data(curr_idx++, _) = data(i, _);
        }
    }
    
    return cluster_data;
}


NumericVector compute_frequencies(const NumericVector& data_col, const int m_j) {
    /**
     * @brief Compute frequencies of values in a column
     * @param data_col NumericVector containing the column data
     * @param m_j Number of attribute levels
     * @return NumericVector containing the frequencies
     * @note This function is used to compute the frequencies of values in a column
     */
    // Pre-allocate with zeros
    NumericVector freqs(m_j, 0.0);
    
    // Single pass counting
    for(int i = 0; i < data_col.length(); i++) {
        int value = data_col[i];
        if(value >= 1 && value <= m_j) {
            freqs[value - 1]++;
        }
    }
    return freqs;
}

List Center_prob_pippo(const NumericMatrix& data, const IntegerVector& indices, 
                        const NumericVector& sigma, const IntegerVector& attrisize) {
    const int p = data.ncol(); // number of features
    const int n = indices.size(); // number of data points in the cluster
    List prob(p);

    for(int j = 0; j < p; j++) {
        const int m_j = attrisize[j];

        // Initialize frequency vector with zeros
        NumericVector freq(m_j, 0.0);

        // Count frequencies using indices
        for(int idx = 0; idx < n; idx++) {
            int i = indices[idx]; // Get the actual row index
            int value = data(i, j);// Get the value at the current row and column
            if(value >= 1 && value <= m_j) {
                freq[value - 1]++;
            }
        }

        // Create and transform probabilities using vectorized operations
        NumericVector prob_tmp = -(n - freq) / sigma[j];

        // Numerical stability: subtract max and exp
        const double max_val = max(prob_tmp);
        prob_tmp = exp(prob_tmp - max_val);

        // Normalize using Sugar's sum
        prob_tmp = prob_tmp / sum(prob_tmp);

        prob[j] = prob_tmp;
    }

    return prob;
}

void update_phi(internal_state& state, const aux_data& const_data, std::vector<int> cluster_indexes) {
    const int& num_cls = state.total_cls;
    const int n_rows = const_data.data.nrow();
    NumericVector new_w(const_data.attrisize.length());
    NumericVector new_v(const_data.attrisize.length());

    // Use Sugar for cluster mask
    LogicalVector cluster_mask = LogicalVector(num_cls, cluster_indexes.empty());
    if (!cluster_indexes.empty()) {
        IntegerVector idx = wrap(cluster_indexes);
        cluster_mask[idx] = true;
    }
    
    for (int i = 0; i < num_cls; i++) {
        if (!cluster_mask[i]) continue;
        
        // Create a vector of indices for this cluster
        std::vector<int> cluster_indices;
        for (int j = 0; j < n_rows; j++) {
            if (state.c_i[j] == i) {
                cluster_indices.push_back(j);
            }
        }
        
        int cluster_size = cluster_indices.size();
        if (cluster_size == 0) continue;
        
        // Convert to IntegerVector for use with Sugar functions
        IntegerVector indices = wrap(cluster_indices);

        /*
        * ------------------------ Update Centers ------------------------
        */
        
        // Update center using optimized functions that work with indices
        List prob_centers = Center_prob_pippo(const_data.data, indices, 
                                            as<NumericVector>(state.sigma[i]), 
                                            const_data.attrisize);
        state.center[i] = sample_center_1_cluster(const_data.attrisize, prob_centers);

        /*
        * ------------------------ Update Sigma ------------------------
        */
        
        const NumericVector& centers_cluster = as<NumericVector>(state.center[i]);
        
        // Initialize counters for matches with center values
        NumericVector match_counts(const_data.attrisize.length(), 0.0);
        
        // Count matches using indices
        for (int idx = 0; idx < cluster_size; idx++) {
            int j = indices[idx]; // Get the actual row index
            for (int k = 0; k < const_data.attrisize.length(); k++) {
                if (const_data.data(j, k) == centers_cluster[k]) {
                    match_counts[k]++;
                }
            }
        }
        
        // Update parameters
        for (int j = 0; j < const_data.attrisize.length(); j++) {
            double sumdelta = match_counts[j];
            new_w[j] = const_data.w[j] + cluster_size - sumdelta;
            new_v[j] = const_data.v[j] + sumdelta;   
        }

        // Sample new sigmas
        state.sigma[i] = sample_sigma_1_cluster(const_data.attrisize, new_v, new_w);
    }
}