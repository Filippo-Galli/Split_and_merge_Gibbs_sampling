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

NumericVector unique_classes_without_index(NumericVector c_i, int index_to_del) {
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

NumericMatrix convert_centers_to_matrix(List centers, int p) {
    /**
     * @brief Converts a List of centers to a NumericMatrix representation
     * @param centers List containing cluster centers
     * @param p Number of variables/dimensions
     * @return NumericMatrix containing the converted center representations
     * @note Helper function for internal use
     */
    int k = centers.length();  // Number of clusters
    NumericMatrix center_matrix(k, p);
    
    for(int i = 0; i < k; i++) {
        NumericVector cluster_probs = centers[i];
        for(int j = 0; j < p; j++) {
            center_matrix(i, j) = which_max(cluster_probs) + 1;
        }
    }
    return center_matrix;
}

NumericVector calculate_class_probabilities(
    NumericVector x_i,
    List centers,
    NumericVector sigma,
    NumericVector attrisize,
    double gamma,
    int m,
    const bool log_scale = false) {
    /**
     * @brief Calculates class probabilities using Hamming density
     * @param x_i Observation vector
     * @param centers List of cluster centers
     * @param sigma Vector of cluster dispersions
     * @param attrisize Vector of attribute sizes
     * @param gamma Concentration parameter for new clusters
     * @param m Number of auxiliary parameters
     * @param log_scale Boolean indicating if calculations should be in log scale
     * @return NumericVector of class probabilities
     * @note Exported to R using Rcpp
     */
    int k_minus = centers.length();
    int h = k_minus + m;
    int p = x_i.length();
    NumericMatrix center_matrix = convert_centers_to_matrix(centers, p);
    NumericVector probabilities(h);
    
    // Calculate probabilities for existing classes
    for (int z = 0; z < k_minus; z++) {
        double prob = 0;
        for (int j = 0; j < p; j++) {
            prob += dhamming(x_i[j], center_matrix(z, j), sigma[z], attrisize[j], log_scale);
        }
        probabilities[z] = prob;
    }
    
    // Calculate probabilities for new classes
    for (int z = k_minus; z < h; z++) {
        double prob = 0;
        for (int j = 0; j < p; j++) {
            prob += dhamming(x_i[j], center_matrix(z % k_minus, j), sigma[z % k_minus], attrisize[j], log_scale);
        }
        probabilities[z] = prob * (gamma/m);
    }
    
    return probabilities;
}

int sample_class_assignment(
    NumericVector x_i,
    NumericVector c_i,
    List centers,
    NumericVector sigma,
    NumericVector attrisize,
    double gamma,
    int m) {
    /**
     * @brief Samples new class assignment for an observation
     * @param x_i Observation vector
     * @param c_i Current class assignments
     * @param centers List of cluster centers
     * @param sigma Vector of cluster dispersions
     * @param attrisize Vector of attribute sizes
     * @param gamma Concentration parameter
     * @param m Number of auxiliary parameters
     * @return Integer representing new class assignment
     * @note Exported to R using Rcpp
     */
    
    NumericVector probs = calculate_class_probabilities(x_i, centers, sigma, attrisize, gamma, m);
    // Normalize probabilities
    probs = exp(probs - max(probs));
    probs = probs / sum(probs);
    
    // Sample new class
    IntegerVector classes = seq_len(probs.length());
    IntegerVector sampled = sample(classes, 1, true, probs);
    return sampled[0];
}

List update_centers(NumericMatrix data, NumericVector sigma, NumericVector attrisize) {
    /**
     * @brief Updates cluster centers based on current assignments
     * @param data Input data matrix
     * @param sigma Vector of cluster dispersions
     * @param attrisize Vector of attribute sizes
     * @return List of updated centers
     * @note Exported to R using Rcpp
     */
    return Center_prob(data, sigma, attrisize);
}

// [[Rcpp::export]]
List run_markov_chain(
    NumericMatrix data,
    NumericVector initial_assignments,
    NumericVector sigma,
    NumericVector attrisize,
    double gamma,
    int m,
    int iterations = 1000) {
    /**
     * @brief Main MCMC sampling function for clustering
     * @param data Input data matrix
     * @param initial_assignments Initial cluster assignments
     * @param sigma Vector of cluster dispersions
     * @param attrisize Vector of attribute sizes
     * @param gamma Concentration parameter
     * @param m Number of auxiliary parameters
     * @param iterations Number of MCMC iterations
     * @return List containing final assignments, centers, and all iteration results
     * @note Exported to R using Rcpp
     * @details This function implements the main MCMC sampling algorithm for
     *          clustering categorical data. It iteratively updates cluster
     *          assignments and centers until convergence or maximum iterations.
     */
    int n = data.nrow();
    NumericVector c_i = clone(initial_assignments);
    
    // Initialize centers using Center_prob
    List centers = update_centers(data, sigma, attrisize);
    
    Rcpp::Rcout << "Starting Markov Chain sampling..." << std::endl;
    
    // Store results for each iteration
    List results(iterations);
    
    // Main MCMC loop
    for (int iter = 0; iter < iterations; iter++) {
        // For each observation
        for (int i = 0; i < n; i++) {
            NumericVector x_i = data(i, _);
            c_i[i] = sample_class_assignment(x_i, c_i, centers, sigma, attrisize, gamma, m);
        }
        
        // Update centers
        centers = update_centers(data, sigma, attrisize);
        
        // Store current state
        List iteration_result = List::create(
            Named("assignments") = clone(c_i),
            Named("centers") = clone(centers)
        );
        results[iter] = iteration_result;
        
        // Update progress bar
        print_progress_bar(iter + 1, iterations);
    }
    
    Rcpp::Rcout << "Sampling completed!" << std::endl;
    
    return List::create(
        Named("final_assignments") = c_i,
        Named("final_centers") = centers,
        Named("all_iterations") = results
    );
}

// [[Rcpp::export]]
List example_usage() {
    /**
     * @brief Example usage function demonstrating the MCMC clustering implementation
     * @return List containing clustering results for example data
     * @note Exported to R using Rcpp
     * @details Creates a small example dataset with 5 observations and 2 variables,
     *          then runs the MCMC clustering algorithm with sample parameters
     */
    // Create example data
    NumericMatrix data(5, 2);
    
    // Fill the first column
    data(0,0) = 1; data(1,0) = 1; data(2,0) = 2; data(3,0) = 2; data(4,0) = 3;
    
    // Fill the second column
    data(0,1) = 1; data(1,1) = 1; data(2,1) = 2; data(3,1) = 2; data(4,1) = 3;
    
    NumericVector initial_assignments = NumericVector::create(1, 1, 2, 2, 3);
    NumericVector sigma = NumericVector::create(0.1, 0.1, 0.1);
    NumericVector attrisize = NumericVector::create(3, 3); // 3 possible values for each variable
    
    double gamma = 1.0;
    int m = 2;
    
    return run_markov_chain(data, initial_assignments, sigma, attrisize, gamma, m, 10000);
}