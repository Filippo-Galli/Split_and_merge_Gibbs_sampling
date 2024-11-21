/**
 * @file neal_sampler.cpp
 * @brief Implementation of Markov Chain Monte Carlo clustering algorithm based on Neal algorithm 8 and Argiento paper 2022 using Rcpp
 * @details This file contains the implementation of a MCMC-based clustering algorithm
 *          specifically designed for categorical data using Hamming distance.
 */

#include <Rcpp.h>
#include <set>
#include <random>
#include <gibbs_utility.cpp>
#include <RcppGSL.h>  // Add this explicit RcppGSL include
#include <gsl/gsl_sf_hyperg.h>
#include <hyperg.cpp>
using namespace Rcpp;

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

NumericVector unique_classes(const NumericVector & c_i) {
    /**
     * @brief Gets unique class labels
     * @param c_i Vector of class assignments
     * @param index_to_del Index to exclude from unique class calculation
     * @return NumericVector containing unique class labels
     * @note Exported to R using Rcpp
     */

    std::set<double> unique_classes;
    for (int i = 0; i < c_i.length(); ++i) {
        unique_classes.insert(c_i[i]);
    }

    // convert set to NumericVector
    NumericVector unique_classes_vector(unique_classes.size());
    std::copy(unique_classes.begin(), unique_classes.end(), unique_classes_vector.begin());

    return unique_classes_vector;
}

NumericVector unique_classes_without_index(const NumericVector & c_i, const int index_to_del) {
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

NumericVector sample_initial_assignment(double K = 4, int n = 10){
    /**
     * @brief Samples initial cluster assignments
     * @param K Number of initial clusters
     * @return NumericVector containing initial cluster assignments
     * @note Exported to R using Rcpp
     */

    // Sample initial cluster assignments
    NumericVector c_i(n);
    for (int i = 0; i < n; i++) {
        c_i[i] = sample(K, 1, true)[0];
    }
    return c_i;
}

NumericVector sample_center_1_cluster(const NumericVector & attrisize) {
    /**
     * @brief Sample initial cluster center
     * @param attrisize Vector of attribute sizes
     * @return NumericVector containing cluster centers
     */

    NumericVector center(attrisize.length());

    for (int i = 0; i < attrisize.length(); i++) {
        center[i] = sample(attrisize[i], 1, true)[0];
    } 

    return center;
}

List sample_centers(const NumericVector & attrisize, const int number_cls) {
    /**
     * @brief Sample initial cluster centers
     * @param attrisize Vector of attribute sizes
     * @param number_cls Number of clusters
     * @param num_cls_sample Number of clusters to sample (default: -1)
     * @return List of cluster centers for each attribute
     * @note The returned list contains a NumericVector for each attribute, So has dimension equal to the number of attributes x number_cls
     * @note We have some perfomance issues with this function? Can we change this using NumericMatrix?
     */

    List centers(attrisize.length());

    for (int i = 0; i < attrisize.length(); i++) {
        NumericVector attr_centers(number_cls);
        for (int i = 0; i < number_cls; i++) {
            attr_centers[i] = sample(attrisize[i], 1, true)[0];
        }
        centers[i] = attr_centers;
    }   

    return centers;
}

NumericVector sample_sigma_1_cluster(const NumericVector & attrisize, const NumericVector & u, const NumericVector & v){
    /**
     * @brief Sample initial cluster dispersion (sigma)
     * @param attrisize Vector of attribute sizes
     * @param number_cls Number of clusters
     * @param u Parameter for hypergeometric distribution
     * @param v Parameter for hypergeometric distribution
     * @return NumericVector containing cluster dispersions for each attribute
     */

    NumericVector sigma(attrisize.length());

    for (int i = 0; i < attrisize.length(); i++) {
        sigma[i] = rhyper_sig(1, v[i], u[i], attrisize[i])[0];
    } 

    return sigma;
}

List sample_sigmas(const NumericVector & attrisize, const int number_cls, const NumericVector & u, const NumericVector & v) {
    /**
     * @brief Sample initial cluster dispersions (sigma)
     * @param attrisize Vector of attribute sizes
     * @param number_cls Number of clusters
     * @param u Parameter for hypergeometric distribution
     * @param v Parameter for hypergeometric distribution
     * @return List of cluster dispersions for each attribute
     */
    List sigma(attrisize.length());

    for (int i = 0; i < attrisize.length(); i++) {
        NumericVector temp = rhyper_sig(number_cls, v[i], u[i], attrisize[i]);
        sigma[i] = temp;
    } 

    return sigma;
}

double calculate_b(const NumericVector & x_i, const NumericVector & c_i, const NumericMatrix & data, 
                   List center, List sigma, const NumericVector & attrisize, 
                   double gamma, int num_cls, const NumericVector & unique_classes_without_i, 
                   int index_i, int m) {
    /**
     * @brief Calculate the denominator (B) in the allocation probability calculation
     * @param x_i Current observation
     * @param c_i Current cluster assignments
     * @param data Full data matrix
     * @param center List of cluster centers
     * @param sigma List of cluster dispersions
     * @param attrisize Vector of attribute sizes
     * @param gamma Concentration parameter
     * @param num_cls Total number of clusters
     * @param unique_classes_without_i Unique classes excluding current observation
     * @param index_i Index of current observation
     * @param m Number of auxiliary parameters
     * @return Normalization constant for allocation probability
     */

    // Initialize numerator and denominator
    double num = data.nrow() - 1 + gamma;
    double den = 0;

    // used parameters
    int k_minus = unique_classes_without_i.length();

    NumericVector sigma_k(attrisize.length());
    NumericVector center_k(attrisize.length());

    // Compute denominator for existing clusters
    for (int z = 0; z < k_minus; z++) { // for each existing cluster
        double Hamming = 1.0;
        int n_i_z = 0;

        for(int j = 0; j < attrisize.length(); j++){
            NumericVector temp = as<NumericVector>(sigma[j]);// vettore attributo j
            sigma_k[j] = temp[z]; // prendo cluster k del attributo j
            temp = as<NumericVector>(center[j]);
            center_k[j] = temp[z];
        }

        // Count instances in this cluster excluding the current point
        for (int i = 0; i < data.nrow(); i++) {
            if (i != index_i && c_i[i] == z) {
                n_i_z++;
            }
        }

        // Compute Hamming distance product for this cluster
        for (int j = 0; j < data.ncol(); j++) {
            Hamming *= dhamming(x_i[j], center_k[j], sigma_k[j], attrisize[j]);
        }
        den += n_i_z * Hamming;
    }

    // Compute denominator for new/auxiliary clusters
    for (int z = k_minus; z < num_cls; z++) {
        double Hamming = 1.0;

        for(int j = 0; j < attrisize.length(); j++){
            NumericVector temp = as<NumericVector>(sigma[j]);// vettore attributo j
            sigma_k[j] = temp[z]; // prendo cluster k del attributo j
            temp = as<NumericVector>(center[j]);
            center_k[j] = temp[z];
        }

        // Compute Hamming distance product for this auxiliary cluster
        for (int j = 0; j < data.ncol(); j++) {
            Hamming *= dhamming(x_i[j], center_k[j], sigma_k[j], attrisize[j]);
        }
        den += (gamma/m) * Hamming;
    }
    return num / den;
}

int sample_allocation(const int index_i, const NumericVector & x_i, const NumericVector & c_i, const NumericMatrix & data, 
                        const List & center, const List & sigma,  const NumericVector & attrisize, 
                        const double gamma, const int num_cls, const NumericVector & unique_classes_without_i, const int m){
    /**
     * @brief Sample new cluster assignment for a given observation
     * @param index_i Index of current observation
     * @param x_i Current observation vector
     * @param c_i Current cluster assignments
     * @param data Full data matrix
     * @param center List of cluster centers
     * @param sigma List of cluster dispersions
     * @param attrisize Vector of attribute sizes
     * @param gamma Concentration parameter
     * @param num_cls Total number of clusters
     * @param unique_classes_without_i Unique classes excluding current observation
     * @param m Number of latent classes
     * @return New cluster assignment for the current observation
    */

    // Find normalizing constant
    double b = calculate_b(x_i, c_i, data, center, sigma, attrisize, gamma, num_cls, unique_classes_without_i, index_i, m);

    //std::cout << "[DEBUG] - " << 2.1 << " b: " << b << std::endl;

    // Calculate factor
    double factor = 1/(data.nrow() - 1 + gamma);

    // Calculate probabilities
    NumericVector probs(num_cls);

    // number of variable in cluster z without i
    int n_i_z = 0;

    for (int k = 0; k < num_cls; k++) {
        // Density calculation, in Neal paper is the F function
        double Hamming = 1;
        NumericVector sigma_k(attrisize.length());
        NumericVector center_k(attrisize.length());

        for(int j = 0; j < attrisize.length(); j++){
            NumericVector temp = as<NumericVector>(sigma[j]);// vettore attributo j
            sigma_k[j] = temp[k]; // prendo cluster k del attributo j
            temp = as<NumericVector>(center[j]);
            center_k[j] = temp[k];
        }

        for (int j = 0; j < data.ncol(); j++) {
            Hamming *= dhamming(x_i[j], center_k[j], sigma_k[j], attrisize[j]);
        }

        // probability calculation for existing clusters
        if (k < unique_classes_without_i.length()) {
            // Count instances in the z cluster excluding the current point i
            n_i_z = 0;
            for (int i = 0; i < data.nrow(); i++) {
                if (i != index_i && c_i[i] == c_i[index_i]) {
                    n_i_z++;
                }
            }
            // Calculate probability
            probs[k] = b * factor * n_i_z * Hamming ;
        }
        // probability calculation for latent clusters 
        else {
            probs[k] = b * factor * gamma/m * Hamming;
        }
    }

    // Create the vector from 0 to num_cls 
    NumericVector cls(num_cls);

    //std::cout << "[DEBUG] - " << 2.1 << " probs: " << probs<< std::endl;

    // Sample new cluster assignment using probabilities calculated before
    return sample(cls, 1, true, probs)[0];

}

List clean_var(List var, const NumericVector & c_i, const NumericVector & attrisize){
    /**
     * @brief Clean the variables that are not used
     * @param var List to clean, could be centers or sigma
     * @param c_i Vector of cluster assignments
     * @param attrisize Vector of attribute sizes
     * @return List containing cleaned variables
     */

    // Number of attributes
    int p = attrisize.length();
    // Vector of existing clusters
    NumericVector existig_cls = unique_classes(c_i);

    List new_var(p);

    for (int i = 0; i < p; i++) { // for each attribute
        NumericVector cls(existig_cls.length());
        NumericVector attribute_var = as<NumericVector>(var[i]);
        for (int j = 0; j < existig_cls.length(); j++) { // for each existing cluster
            cls[j] = attribute_var[existig_cls[j]]; // copy the value of the existing cluster inside the new vector
        }
        new_var[i] = cls; // add the new vector to the list
    }

    return new_var;
}

double sum_delta(const NumericMatrix & data, const int j, const int a) {
    /**
     * @brief Sum of occurrences of a specific attribute value
     * @param data Input data matrix
     * @param j Column index
     * @param a Attribute value
     * @return Sum of occurrences
     */
    double sumdelta = 0;
    for (int k = 0; k < data.nrow(); k++) { // for each observation
        if (data(k,j) == a) // if the attribute value is equal to the modality a a
            sumdelta++;
    }
    return sumdelta;
}

List update_centers(const NumericMatrix & data, const NumericVector & attrisize, 
                    const NumericVector & c_i, const List & sigma_prec) {
    /**
     * @brief Update cluster centers
     * @param data Input data matrix
     * @param attrisize Vector of attribute sizes
     * @param c_i Current cluster assignments
     * @param sigma_prec Previous sigma values
     * @return List of updated cluster centers
     */
    List prob_centers(attrisize.length());
    NumericVector clusters = unique_classes(c_i);
    
    /*
     * Calculation of probabilities for each attribute and modality to be chosen
     */
    for (int i = 0; i < attrisize.length(); i++) { // for each attribute
        NumericVector probs(attrisize[i]);
        double den=0;
        for (int a = 0; a < attrisize[i]; a++) { // for each modality
            double num = 0, sumdelta = 0;
            sumdelta = sum_delta(data, i, a);
            //std::cout << "\n[DEBUG] - " << 3.1 << " sumdelta: " << sumdelta << std::endl;
            // Calculate numerator of the full conditional
            double sigma_value = as<double>(sigma_prec[i]);
            
            //std::cout << "\n[DEBUG] - " << 3.1 << " sigmaprec: " << sigma_value << std::endl;
            
            num = pow((1 + (attrisize[a] - 1) / (exp(1 / sigma_value))), -data.nrow()) * 
                exp((data.nrow() - sumdelta) / (sigma_value));
            
            std::cout << "[DEBUG] - " << 3.2 << " num: " << num << std::endl;
            
            probs[a] = num;
            den+=num;
        }     
        
        std::cout << "\n[DEBUG] - " << 3.3 << " den: " << den << std::endl;
        
        for(int p = 0; p < probs.length(); p++){
            probs[p] = probs[p]/den;
            std::cout << "[DEBUG] - " << 3.4 << " prob["<<p<<"]: " << probs[p] << std::endl;
        }     
        
        // Store probabilities for this attribute
        prob_centers[i] = probs;
    }
    
    /*
     * Sample new cluster centers using the probabilities calculated before
     */
    List centers(attrisize.length());
    
    // for each attribute
    for (int i = 0; i < attrisize.length(); i++) { 
        // temporary vector to store centers for this attribute for each cluster
        NumericVector attr_centers(clusters.length());
        
        // for each cluster
        for (int j = 0; j < clusters.length(); j++) { 
            // Sample a modalities for this attribute using the probabilities calculated before
            attr_centers[j] = sample(attrisize[i], 1, true, as<NumericVector>(prob_centers[i]))[0];
        }
        centers[i] = attr_centers;
    }
    
    return centers;
}


List update_sigma(const List & centers, const NumericVector & w, const NumericVector & v, 
                  const NumericMatrix & data, const NumericVector & attrisize, const int num_cls) {
/**
 * @brief Update cluster dispersions using the original method
 * @param centers List of current cluster centers
 * @param w Parameter for dispersion not updated
 * @param v Parameter for dispersion not updated
 * @param data Input data matrix
 * @param attrisize Vector of attribute sizes
 * @param num_cls Total number of clusters
 * @return List of updated cluster dispersions
 */
    List sigmas(attrisize.length());

    for (int j = 0; j < attrisize.length(); j++) {
        
        double sumdelta = sum_delta(data, j, centers[j]);

        // Update v and w
        double new_v = v[j] + sumdelta;
        double new_w = w[j] + data.nrow() - sumdelta;
        
        // Append new sigma values
        sigmas[j] = rhyper_sig(num_cls, new_v, new_w, attrisize[j]);
    }

    return sigmas;
}

// [[Rcpp::export]]
List run_markov_chain(NumericMatrix data, NumericVector attrisize, double gamma,
                    NumericVector v, NumericVector u, int verbose = 0, int m = 5, int iterations = 1000) {
    /**
     * @brief Main Markov Chain Monte Carlo sampling function
     * @param data Input data matrix
     * @param attrisize Vector of attribute sizes
     * @param gamma Concentration parameter
     * @param v Parameter for sigma update
     * @param u Parameter for sigma update
     * @param m Number of latent classes (default: 5)
     * @param iterations Number of MCMC iterations (default: 1000)
     * @return List containing final clustering results
    */
    int n = data.nrow();

    // First cluster assignments
    int L = 4;

    // Initialize cluster assignments
    NumericVector c_i = sample_initial_assignment(L, n);

    if(verbose == 1){
        std::cout << "[DEBUG] - " << " c_i[i]: " << std::endl;
        for (int i = 0; i < n; i++) {  
            std::cout << c_i[i] << " ";
        }
        std::cout << std::endl;
    }

    
    // Initialize centers using Center_prob
    List center = sample_centers(attrisize, L + m);

    if(verbose == 2){
        std::cout << "[DEBUG] - " << " center[i]: " << std::endl;
        for (int i = 0; i < attrisize.length(); i++) {  
            NumericVector temp = as<NumericVector>(center[i]);
            for(int j = 0; j < temp.length(); j++){
                std::cout << temp[j] << " ";
            }
        }
    }

    // Initialize sigma
    List sigma = sample_sigmas(attrisize, L + m, u, v);
    List sigma_prec;

    if(verbose == 3){
        std::cout << "[DEBUG] - " << " sigma[i]: " << std::endl;
        for (int i = 0; i < attrisize.length(); i++) {  
            NumericVector temp = as<NumericVector>(sigma[i]);
            for(int j = 0; j < temp.length(); j++){
                std::cout << temp[j] << " ";
            }
        }
    }
    
    Rcpp::Rcout << "Starting Markov Chain sampling..." << std::endl;
    
    // Store results for each iteration
    List results(iterations);
    
    // Main MCMC loop
    for (int iter = 0; iter < iterations; iter++) {

        NumericVector prec_unique_classes = unique_classes(c_i);

        // For each observation
        for (int i = 0; i < n; i++) {
            // take the i-th observation
            NumericVector x_i = data(i, _);

            // Remove i-th observation from cluster
            NumericVector unique_classes_without_i = unique_classes_without_index(c_i, i);

            // Check if the i-th observation is the only one in the cluster
            int start_k = unique_classes_without_i.length() == prec_unique_classes.length() ? L : L + 1;

            // Re-Sample m the latent classes
            for (int k = start_k; k < L + m; k++){
                // sample a new center
                NumericVector new_center = sample_center_1_cluster(attrisize);
                
                // update the center list with the new center
                center[k] = new_center;
                
                // sample a new sigma
                NumericVector new_sigma = sample_sigma_1_cluster(attrisize, u, v);
                // update the sigma list with the new sigma
                sigma[k] = new_sigma;
                
            }
            // Compute the new allocation of the i-th observation
            c_i[i] = sample_allocation(i, x_i, c_i, data, center, sigma, attrisize, gamma, L + m, unique_classes_without_i, m);  
        }

        // Clean variables from unused latent classes
        center = clean_var(center, c_i, attrisize);
        sigma = clean_var(sigma, c_i, attrisize);

        // Update centers
        sigma_prec = clone(sigma);
        center = update_centers(data, attrisize, c_i, sigma_prec);
        // Update sigma
        sigma = update_sigma(center, u, v, data, attrisize, unique_classes(c_i).length());
        
        // Store current state
        List iteration_result = List::create(
            Named("assignments") = clone(c_i),
            Named("centers") = clone(center)
        );
        results[iter] = iteration_result;
        
        // Update progress bar
        print_progress_bar(iter + 1, iterations);
    }
    
    Rcpp::Rcout << "Sampling completed!" << std::endl;
    
    return List::create(
       Named("final_assignments") = c_i,
       Named("final_centers") = center,
       Named("sigma") = sigma,
       Named("all_iterations") = results
    );
}