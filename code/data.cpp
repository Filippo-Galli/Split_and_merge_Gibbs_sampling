#include "data.hpp"

Data::Data(const NumericMatrix & data, const IntVec & attrisize, const double gamma, const  DoubleVec & v, const  DoubleVec & w, const int m): 
            data(data), n(data.nrow()), p(data.ncol()), m(m), attrisize(attrisize), gamma(gamma), v(v), w(w) {
    /**
     * @brief Constructor for the Data class
     * @param data Input data matrix
     * @param attrisize Vector of attribute sizes
     * @param gamma Gamma parameter
     * @param v Vector of v parameters
     * @param w Vector of w parameters
     * @note This constructor reserve the memory for the data and initialize the parameters of the model
     */

    // Reserve space for the c vector
    c = IntVec(n);
}

void Data::set_c(const int & idx, const int & value) {
    /**
     * @brief Setter for the cluster assignment of a specific data point
     * @param idx Index of the data point
     * @param value Cluster assignment
     * @note This function is used to set the cluster assignment of a specific data point
     */

    c[idx] = value;
}

void Data::set_c(const IntVec & c) {
    /**
     * @brief Setter for the cluster assignments
     * @param c Cluster assignments
     * @note This function is used to set the cluster assignments
     */

    this->c = c;
    // update the total number of clusters
    set_total_cls(unique(c).length());
}

void Data::set_center(const int & idx, const  DoubleVec & center) {
    /**
     * @brief Setter for the cluster center of a specific cluster
     * @param idx Index of the cluster
     * @param center Cluster center
     * @note This function is used to set the cluster center of a specific cluster
     */

    this->center[idx] = center;
}

void Data::set_center(const DoubleMat & center) {
    /**
     * @brief Setter for the cluster centers
     * @param center DoubleMat of cluster centers
     * @note This function is used to set the cluster centers
     */

    this->center = center;
}

void Data::set_sigma(const int & idx, const  DoubleVec & sigma) {
    /**
     * @brief Setter for the cluster sigma of a specific cluster
     * @param idx Index of the cluster
     * @param sigma Cluster sigma
     * @note This function is used to set the cluster sigma of a specific cluster
     */

    this->sigma[idx] = sigma;
}

void Data::set_sigma(const DoubleMat & sigma) {
    /**
     * @brief Setter for the cluster sigmas
     * @param sigma DoubleMat of cluster sigmas
     * @note This function is used to set the cluster sigmas
     */

    this->sigma = sigma;
}

void Data::set_total_cls(const int total_cls) {
    /**
     * @brief Setter for the total number of clusters
     * @param total_cls Total number of clusters
     * @note This function is used to set the total number of clusters
     */

    this->total_cls = total_cls;
}

IntegerVector Data::get_c() const {
    /**
     * @brief Getter for the cluster assignments
     * @return IntVec containing the cluster assignments
     * @note This function is used to get the cluster assignments
     */

    return c;
}

double Data::get_c(const int & idx) const {
    /**
     * @brief Getter for the cluster assignment of a specific data point
     * @param idx Index of the data point
     * @return Cluster assignment of the data point
     * @note This function is used to get the cluster assignment of a specific data point
     */

    return c[idx];
}

NumericVector Data::get_center(const int & idx) const {
    /**
     * @brief Getter for the cluster center of a specific cluster
     * @param idx Index of the cluster
     * @return  DoubleVec containing the cluster center
     * @note This function is used to get the cluster center of a specific cluster
     */

    return center[idx];
}

DoubleMat Data::get_center() const {
    /**
     * @brief Getter for the cluster centers
     * @return DoubleMat containing the cluster centers
     * @note This function is used to get the cluster centers
     */

    return center;
}

NumericVector Data::get_sigma(const int & idx) const {
    /**
     * @brief Getter for the cluster sigma of a specific cluster
     * @param idx Index of the cluster
     * @return  DoubleVec containing the cluster sigma
     * @note This function is used to get the cluster sigma of a specific cluster
     */

    return sigma[idx];
}

DoubleMat Data::get_sigma() const {
    /**
     * @brief Getter for the cluster sigmas
     * @return DoubleMat containing the cluster sigmas
     * @note This function is used to get the cluster sigmas
     */

    return sigma;
}

int Data::get_total_cls() const {
    /**
     * @brief Getter for the total number of clusters
     * @return Total number of clusters
     * @note This function is used to get the total number of clusters
     */

    return total_cls;
}

NumericVector Data::get_data_row(const int & idx) const {
    /**
     * @brief Get a specific data point
     * @param idx Index of the data point
     * @return  DoubleVec containing the data point
     * @note This function is used to get a specific data point
     */

    return data(idx, _);
}

NumericVector Data::get_data_col(const int & feature) const {
    /**
     * @brief Get a specific feature
     * @param feature Index of the feature
     * @return  DoubleVec containing the feature
     * @note This function is used to get a specific feature
     */

    return data(_, feature);
}

double Data::get_data(const int & idx, const int & feature) const {
    /**
     * @brief Get a specific data point feature
     * @param idx Index of the data point
     * @param feature Index of the feature
     * @return Value of the feature
     * @note This function is used to get a specific data point feature
     */

    return data(idx, feature);
}

NumericMatrix Data::get_data() const {
    /**
     * @brief Getter for the data matrix
     * @return NumericMatrix containing the data matrix
     * @note This function is used to get the data matrix
     */

    return data;
}

int Data::get_n() const {
    /**
     * @brief Getter for the number of rows
     * @return Number of rows
     * @note This function is used to get the number of rows
     */

    return n;
}

int Data::get_p() const {
    /**
     * @brief Getter for the number of columns
     * @return Number of columns
     * @note This function is used to get the number of columns
     */

    return p;
}

int Data::get_m() const {
    /**
     * @brief Getter for the number of latent clusters
     * @return Number of latent clusters
     * @note This function is used to get the number of latent clusters
     */

    return m;
}

double Data::get_attrisize(const int & idx) const {
    /**
     * @brief Getter for the attribute size of a specific feature
     * @param idx Index of the feature
     * @return Attribute size of the feature
     * @note This function is used to get the attribute size of a specific feature
     */

    return attrisize[idx];
}

IntegerVector Data::get_attrisize() const {
    /**
     * @brief Getter for the attribute sizes
     * @return IntVec containing the attribute sizes
     * @note This function is used to get the attribute sizes
     */

    return attrisize;
}

double Data::get_gamma() const {
    /**
     * @brief Getter for the gamma parameter
     * @return Gamma parameter
     * @note This function is used to get the gamma parameter
     */

    return gamma;
}

double Data::get_v(const int & idx) const {
    /**
     * @brief Getter for the v parameter of a specific feature
     * @param idx Index of the feature
     * @return Value of the v parameter
     * @note This function is used to get the v parameter of a specific feature
     */

    return v[idx];
}

NumericVector Data::get_v() const {
    /**
     * @brief Getter for the v parameters
     * @return  DoubleVec containing the v parameters
     * @note This function is used to get the v parameters
     */

    return v;
}

double Data::get_w(const int & idx) const {
    /**
     * @brief Getter for the w parameter of a specific feature
     * @param idx Index of the feature
     * @return Value of the w parameter
     * @note This function is used to get the w parameter of a specific feature
     */

    return w[idx];
}

NumericVector Data::get_w() const {
    /**
     * @brief Getter for the w parameters
     * @return  DoubleVec containing the w parameters
     * @note This function is used to get the w parameters
     */

    return w;
}

IntegerVector Data::get_cluster_indices(const int & cluster) const {
    /**
     * @brief Get the indices of the data points in a cluster
     * @param cluster Cluster index
     * @return IntVec containing the indices of the data points in the cluster
     * @note This function is used to get the indices of the data points in a cluster
     */

    IntVec cluster_indices;
    for (int i = 0; i < as<NumericVector>(c).length(); ++i) 
        if ( as<NumericVector>(c)[i] == cluster) 
            cluster_indices.push_back(i);
    
    return cluster_indices;
}

NumericMatrix Data::get_cluster_data(const int & cluster) const {
    /**
     * @brief Get the data points in a cluster
     * @param cluster Cluster index
     * @return NumericMatrix containing the data points in the cluster
     * @note This function is used to get the data points in a cluster
     */
    
    IntVec cluster_indices = get_cluster_indices(cluster);
    
    NumericMatrix cluster_data(cluster_indices.length(), data.ncol());
    for (int i = 0; i < cluster_indices.length(); ++i) {
        cluster_data(i, _) = data(cluster_indices[i], _);
    }
    
    return cluster_data;
}

NumericVector Data::get_cluster_center(const int & cluster) const {
    /**
     * @brief Get the center of a cluster
     * @param cluster Cluster index
     * @return  DoubleVec containing the center of the cluster
     * @note This function is used to get the center of a cluster
     */
    
    return center[cluster];
}

NumericVector Data::get_cluster_sigma(const int & cluster) const {
    /**
     * @brief Get the sigma of a cluster
     * @param cluster Cluster index
     * @return  DoubleVec containing the sigma of the cluster
     * @note This function is used to get the sigma of a cluster
     */
    
    return sigma[cluster];
}

NumericVector Data::get_unique_clusters() const {
    /**
     * @brief Get the unique clusters sorted
     * @return  DoubleVec containing the unique clusters sorted
     * @note This function is used to get the unique clusters sorted
     */

    return  DoubleVec(unique(c).sort());
}

NumericVector Data::get_unique_clusters_without_idx(const int & idx) const {
    /**
     * @brief Get the unique clusters sorted without a specific index
     * @param idx Index to exclude
     * @return  DoubleVec containing the unique clusters sorted without the index
     * @note This function is used to get the unique clusters sorted without a specific index
     */

     DoubleVec unique_clusters = get_unique_clusters();
    int cls_idx = get_c(idx);
    // if there are more than one cluster, return all clusters
    if(get_cluster_indices(cls_idx).length() != 1) {
        return unique_clusters;
    }

    // remove the cluster of the index
     DoubleVec unique_clusters_without_idx = unique_clusters[unique_clusters != cls_idx];
    
    return unique_clusters_without_idx;
}

int Data::get_cluster_size(const int & cluster) const {
    /**
     * @brief Get the size of a cluster
     * @param cluster Cluster index
     * @return Size of the cluster
     * @note This function is used to get the size of a cluster
     */

    return get_cluster_indices(cluster).length();
}

NumericVector Data::get_cluster_sizes() const {
    /**
     * @brief Get the sizes of the clusters
     * @return  DoubleVec containing the sizes of the clusters
     * @note This function is used to get the sizes of the clusters
     */
     DoubleVec unique_clusters = get_unique_clusters();
     DoubleVec cluster_sizes(unique_clusters.length());
    for (int i = 0; i < cluster_sizes.length(); ++i) {
        cluster_sizes[i] = get_cluster_size(unique_clusters[i]);
    }
    
    return cluster_sizes;
}

void Data::clean_var(IntegerVector & c_to, DoubleMat & center_to, DoubleMat & sigma_to,
                    const IntVec & c_from, const DoubleMat & center_from, const DoubleMat & sigma_from,
                    const  DoubleVec& existing_cls) const {
    /**
     * @brief Remove empty clusters and update the internal state
     * @param c_to Cluster assignments final
     * @param center_to Cluster centers final
     * @param sigma_to Cluster sigmas final
     * @param c_from Cluster assignments original
     * @param center_from Cluster centers original
     * @param sigma_from Cluster sigmas original
     * @param existing_cls Vector of existing clusters
     */

    int num_existing_cls = existing_cls.length();
    std::unordered_map<int, int> cls_to_new_index;
    
    for(int i = 0; i < num_existing_cls; ++i) {
        int idx_temp = 0;
        // if the cluster is in the existing clusters, keep the index
        if(existing_cls[i] < num_existing_cls){
            cls_to_new_index[existing_cls[i]] = existing_cls[i];
        } 
        // if the cluster is not in the existing clusters, find the next available index
        else {
            while(cls_to_new_index.find(idx_temp) != cls_to_new_index.end() && idx_temp < num_existing_cls){
                ++idx_temp;
            }
            cls_to_new_index[existing_cls[i]] = idx_temp;
        }
    }
    
    // Clean the variables and reserve the memory
    center_to = DoubleMat(num_existing_cls);
    sigma_to = DoubleMat(num_existing_cls);

    // Copy the existing clusters
    for(int i = 0; i < num_existing_cls; ++i) {
        if(existing_cls[i] >= center_from.length()) {
            throw std::out_of_range("Invalid cluster index");
        }
        center_to[cls_to_new_index[existing_cls[i]]] = center_from[existing_cls[i]];
        sigma_to[cls_to_new_index[existing_cls[i]]] = sigma_from[existing_cls[i]];
    }

    // Update the cluster assignments keeping only the existing clusters
    for(int i = 0; i < get_n(); i++) {
        auto it = cls_to_new_index.find(c_from[i]);
        if(it != cls_to_new_index.end()) {
            c_to[i] = it->second;
        }
    }
}

void Data::clean_var(IntegerVector & c, DoubleMat & center, DoubleMat & sigma,const  DoubleVec& existing_cls) const {
    /**
     * @brief Remove empty clusters and update on the input variables
     * @param c Cluster assignments
     * @param center Cluster centers
     * @param sigma Cluster sigmas
     * @param existing_cls Vector of existing clusters
     */

    int num_existing_cls = existing_cls.length();
    std::unordered_map<int, int> cls_to_new_index;
    
    for(int i = 0; i < num_existing_cls; ++i) {
        int idx_temp = 0;
        // if the cluster is in the existing clusters, keep the index
        if(existing_cls[i] < num_existing_cls){
            cls_to_new_index[existing_cls[i]] = existing_cls[i];
        } 
        // if the cluster is not in the existing clusters, find the next available index
        else {
            while(cls_to_new_index.find(idx_temp) != cls_to_new_index.end() && idx_temp < num_existing_cls){
                ++idx_temp;
            }
            cls_to_new_index[existing_cls[i]] = idx_temp;
        }
    }
    
    // Clean the variables and reserve the memory
    IntVec c_to(c.length());
    DoubleMat center_to(num_existing_cls);
    DoubleMat sigma_to(num_existing_cls);

    // Copy the existing clusters
    for(int i = 0; i < num_existing_cls; ++i) {
        if(existing_cls[i] >= center.length()) {
            throw std::out_of_range("Invalid cluster index");
        }
        center_to[cls_to_new_index[existing_cls[i]]] = center[existing_cls[i]];
        sigma_to[cls_to_new_index[existing_cls[i]]] = sigma[existing_cls[i]];
    }

    // Update the cluster assignments keeping only the existing clusters
    for(int i = 0; i < get_n(); i++) {
        auto it = cls_to_new_index.find(c[i]);
        if(it != cls_to_new_index.end()) {
            c_to[i] = it->second;
        }
    }

    // Update the input variables
    c = std::move(c_to);
    center = std::move(center_to);
    sigma = std::move(sigma_to);
}

void Data::clean_var() {
    /**
     * @brief Remove empty clusters and update the internal state
     * @param c_to Cluster assignments final
     * @param center_to Cluster centers final
     * @param sigma_to Cluster sigmas final
     * @param c_from Cluster assignments original
     * @param center_from Cluster centers original
     * @param sigma_from Cluster sigmas original
     * @param existing_cls Vector of existing clusters
     */
    
    const  DoubleVec& existing_cls = get_unique_clusters();
    int num_existing_cls = existing_cls.length();
    
    IntVec c_to(n);
    DoubleMat center_to(num_existing_cls);
    DoubleMat sigma_to(num_existing_cls);

    const IntVec& c_from = get_c();
    const DoubleMat& center_from = get_center();
    const DoubleMat& sigma_from = get_sigma();

    std::unordered_map<int, int> cls_to_new_index;
    
    for(int i = 0; i < num_existing_cls; ++i) {
        int idx_temp = 0;
        // if the cluster is in the existing clusters, keep the index
        if(existing_cls[i] < num_existing_cls){
            cls_to_new_index[existing_cls[i]] = existing_cls[i];
        } 
        // if the cluster is not in the existing clusters, find the next available index
        else {
            while(cls_to_new_index.find(idx_temp) != cls_to_new_index.end() && idx_temp < num_existing_cls){
                ++idx_temp;
            }
            cls_to_new_index[existing_cls[i]] = idx_temp;
        }
    }
    
    // Clean the variables and reserve the memory
    center_to = DoubleMat(num_existing_cls);
    sigma_to = DoubleMat(num_existing_cls);

    // Copy the existing clusters
    for(int i = 0; i < num_existing_cls; ++i) {
        // possible error here: isn't it supposed to be center_to[cls_to_new_index[existing_cls[i]]] = center_from[existing_cls[i]]?
        center_to[i] = center_from[existing_cls[i]];
        sigma_to[i] = sigma_from[existing_cls[i]];
    }

    // Update the cluster assignments keeping only the existing clusters
    for(int i = 0; i < get_n(); i++) {
        auto it = cls_to_new_index.find(c_from[i]);
        if(it != cls_to_new_index.end()) {
            c_to[i] = it->second;
        }
    }

    // Update the internal state
    set_c(c_to);
    set_center(center_to);
    set_sigma(sigma_to);
}

void Data::initialize_memory() {
    /**
     * @brief Initialize the memory for the cluster centers and sigmas
     * @note This function is used to initialize the memory for the cluster centers and sigmas
     */

    // Reserve space for the center and sigma DoubleMats
    center = DoubleMat(total_cls);
    sigma = DoubleMat(total_cls);
}

void Data::print(const int interest) const {
    /**
     * @brief Print the state of the data object
     * @note This function is used to print the state of the data object
     */
    if(interest == 0 || interest == -1){
        Rcpp::Rcout << "Total Cluster: " << std::endl;
        Rcpp::Rcout << total_cls << std::endl;
    }

    if(interest == 1 || interest == -1){
        Rcpp::Rcout << "Cluster assignments: " << std::endl << "\t";
        Rcpp::Rcout << c << std::endl;
    }

    if(interest == 2 || interest == -1){
        Rcpp::Rcout << "Centers: " << std::endl;
        for (int i = 0; i < total_cls; i++) {
            Rcpp::Rcout << "\tCluster "<< i << " :" << std::setprecision(5) 
                    << as<NumericVector>(center[i]) << std::endl;
        }
    }

    if(interest == 3 || interest == -1){
        Rcpp::Rcout << "Dispersions: " << std::endl;
        for (int i = 0; i < total_cls; i++) {
            Rcpp::Rcout << "\tCluster "<< i << ": " 
                    << as<NumericVector>(sigma[i]) << std::endl;
        }
    }


}
