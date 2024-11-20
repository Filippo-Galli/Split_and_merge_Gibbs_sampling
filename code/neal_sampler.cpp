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
//#include <hyperg.cpp>
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

NumericVector unique_classes(NumericVector c_i) {
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
    return wrap(std::vector<double>(unique_classes.begin(), unique_classes.end()));
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
        c_i[i] = R::runif(0, K);
    }
    return c_i;
}

NumericMatrix sample_centers(NumericVector attrisize, int number_cls){
    /**
     * @brief Samples initial cluster centers
     * @param attrisize Vector of attribute sizes
     * @param number_cls Number of clusters
     * @return NumericVector containing initial cluster centers
     * @note Exported to R using Rcpp
     */

    NumericMatrix centers(attrisize.length(), number_cls);

    for  (int i = 0; i < attrisize.length(); i++) {
        for (int j = 0; j < number_cls; j++) {
            centers(i, j) = R::runif(1, attrisize[i]);
        }
    }   

    return centers;
}

NumericMatrix sample_sigma(NumericVector attrisize, int number_cls, NumericVector u, NumericVector v, bool posterior = false){
    /**
     * @brief Samples initial cluster centers
     * @param attrisize Vector of attribute sizes
     * @param number_cls Number of clusters
     * @return NumericVector containing initial cluster centers
     * @note Exported to R using Rcpp
     */

    NumericMatrix sigma(attrisize.length(), number_cls);

    for  (int i = 0; i < attrisize.length(); i++) {
        NumericVector temp = rhyper_sig(number_cls, v[i], u[i], attrisize[i]);
        for (int j = 0; j < number_cls; j++) {
            sigma(i, j) = temp(j);
        }
    } 

    return sigma;
}

double calculate_b(NumericVector x_i, NumericVector c_i, NumericMatrix data, 
                    NumericMatrix center, NumericMatrix sigma,  NumericVector attrisize, 
                    double gamma, int num_cls, NumericVector unique_classes_without_i, 
                    int index_i, int m){

    double num = data.nrow() - 1 + gamma;

    double den = 0;
    int n_i_z = 0;
    double Hamming = 1;
    int k_minus = unique_classes_without_i.length();

    // First summation of the denominator
    for (int z = 0; z < k_minus; z++) {
        double Hamming = 1;

        n_i_z = 0;
        for (int i = 0; i < data.nrow(); i++) {
            if (i != index_i && c_i[i] == z) {
                n_i_z++;
            }
        }

        for (int j = 0; j < data.ncol(); j++) {
            Hamming *= dhamming(x_i[j], center(z, j), sigma(z, j), attrisize[j]);
        }
        den += n_i_z * Hamming;
    }

    // Second summation of the denominator
    for (int z = k_minus ; z < num_cls; z++) {
        double Hamming = 1;

        for (int j = 0; j < data.ncol(); j++) {
            Hamming *= dhamming(x_i[j], center(z, j), sigma(z, j), attrisize[j]);
        }
        den += (gamma/m) * Hamming;
    }


    return num/den;
}

int sample_allocation(NumericVector x_i, NumericVector c_i, NumericMatrix data, 
                        NumericMatrix center, NumericMatrix sigma,  NumericVector attrisize, 
                        double gamma, int num_cls, NumericVector unique_classes_without_i, int m, int index_i){

    // Find normalizing constant
    double b = calculate_b(x_i, c_i, data, center, sigma, attrisize, gamma, num_cls, unique_classes_without_i, index_i, m);

    // Calculate factor
    double factor = 1/(data.nrow() - 1 + gamma);

    // Calculate probabilities
    NumericVector probs(num_cls);

    int n_i_z = 0;

    for (int k = 0; k < num_cls; k++) {
        // F nel paper Neal
        double Hamming = 1;
        for (int j = 0; j < data.ncol(); j++) {
            Hamming *= dhamming(x_i[j], center(k, j), sigma(k, j), attrisize[j]);
        }


        if (k < unique_classes_without_i.length()) {
            n_i_z = 0;
            for (int i = 0; i < data.nrow(); i++) {
                if (i != index_i && c_i[i] == c_i[index_i]) {
                    n_i_z++;
                }
            }

            probs[k] = b*factor* n_i_z * Hamming ;
        } else {
            probs[k] = b * factor * gamma/m * Hamming;
        }
    }

    // Create the vector from 1 to num_cls + 1
    NumericVector cls(num_cls);

    // Sample new cluster assignment using probabilities calculated before
    return sample(cls, 1, true, probs)[0];

}

NumericMatrix clean_var(NumericMatrix var, NumericVector c_i, NumericVector attrisize){
    /**
     * @brief Clean the variables that are not used
     * @param var Matrix of variables
     * @param c_i Vector of cluster assignments
     * @param attrisize Vector of attribute sizes
     * @return NumericMatrix containing cleaned variables
     * @note Exported to R using Rcpp
     */

    int p = var.nrow();
    int new_cls = unique_classes(c_i).length();

    NumericMatrix new_var(p, new_cls);

    for (int i = 0; i < p; i++) {
        for (int j = 0; j < new_cls; j++) {
            new_var(i, j) = var(i, c_i[j]);
        }
    }

    return new_var;
}

double sum_delta(NumericMatrix data, int j, int a){
    // Summation of delta
    double sumdelta=0;
    for(int k=0; k<data.nrow(); k++){
        if(data[k,j]==a) 
        sumdelta++;
    }
    return sumdelta;
}

NumericMatrix update_centers(NumericMatrix data, NumericVector attrisize,NumericVector c_i, NumericMatrix sigma_prec){
    /**
     * @brief Update cluster centers
     * @param data Input data matrix
     * @param attrisize Vector of attribute sizes
     * @return NumericMatrix containing updated cluster centers
     * @note Exported to R using Rcpp
     */
    NumericMatrix centers(attrisize.length(), unique_classes(c_i).length());
    List probs_list;
    
    double num, den, sumdelta;

    for(int i=0; i<attrisize.length(); i++){ // per ogni variabile
        NumericVector probs(attrisize[i]);
        for(int a = 0; a<attrisize[i]; a++){ // per ogni modalità
            num = 0;
            den = 0;
            sumdelta = sum_delta(data, i, a);
            num= pow((1+(attrisize[a]-1)/(exp(1/sigma_prec(a,i)))),-data.nrow())*exp((data.nrow()-sumdelta)/(sigma_prec(a,i)));
            
            for(int k=0; k<attrisize[i]; k++){
                sumdelta = sum_delta(data, i, k);
                den+=(pow((1+(attrisize[k]-1)/(exp(1/sigma_prec(k,i)))),-data.nrow()))*exp((data.nrow()-sumdelta)/(sigma_prec(k,i)));
            }
            probs[a] = num/den;
        }
        probs_list.push_back(probs);
    }
    return centers;
}

List update_sigma(NumericVector centers, NumericVector w, NumericVector v, NumericMatrix data, NumericVector attrisize){
    /**
     * @brief Update cluster dispersions
     * @param data Input data matrix
     * @param attrisize Vector of attribute sizes
     * @return NumericMatrix containing updated cluster dispersions
     * @note Exported to R using Rcpp
     */
    
    double num, den, sumdelta = 0;
    double new_v = 0, new_w = 0;

    List sigmas;

    for(int j = 0; j < centers.length(); ++j){
        sumdelta = sum_delta(data, j, centers[j]);
        // Update v and w
        new_v = v[j] + sumdelta;
        new_w = w[j] + data.nrow() - sumdelta;
        
        // Create prob vector
        sigmas.append(rhyper_sig(1, new_v, new_w, attrisize[i]));
    }


    return sigmas;
}

// [[Rcpp::export]]
List run_markov_chain(
    NumericMatrix data,
    NumericVector attrisize,
    double gamma,
    NumericVector v,
    NumericVector u,
    int m = 3,
    int iterations = 1000) {
    /**
     * @brief Main MCMC sampling function for clustering
     * @param data Input data matrix
     * @param initial_assignments Initial cluster assignments
     * @param sigma Vector of cluster dispersions
     * @param attrisize Vector of attribute sizes len p and each component is the number of modalities for each variable
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

    // Initialize cluster assignments
    int L = 4;

    // Initialize cluster assignments
    NumericVector c_i = sample_initial_assignment(L, n);
    
    // Initialize centers using Center_prob
    NumericMatrix center = sample_centers(attrisize, L + m);

    // Initialize sigma
    NumericMatrix sigma = sample_sigma(attrisize, L + m, u, v);
    NumericMatrix sigma_prec;

    
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
            if (unique_classes_without_i.length() == prec_unique_classes.length()){
                // Re-Sample m the latent classes
                for ( int k = L; k < L + m; k++){
                    center(_, k) = sample_centers(attrisize, 1);
                    sigma(_, k) = sample_sigma(attrisize, 1, u, v);
                }
            }
            else{
                // Re-Sample m-1 the latent classes 
                for ( int k = L + 1 ; k < L + m; k++){
                    center(_, k) = sample_centers(attrisize, 1);
                    sigma(_, k) = sample_sigma(attrisize, 1, u, v);
                }
            }

            // Compute the new allocation of the i-th observation
            c_i[i] = sample_allocation(x_i, c_i, data, center, sigma, attrisize, gamma, L + m, unique_classes_without_i, m, i);
            
        }

        // Clean variables from unused latent classes
        center = clean_var(center, c_i, attrisize);
        sigma = clean_var(sigma, c_i, attrisize);

        // Update centers
        sigma_prec = clone(sigma);
        centers = update_centers(data, attrisize, c_i, sigma_prec);
        // Update sigma _ CHECK IS WRONG!!!!!!!!!!
        sigma = update_sigma(centers, attrisize, u, v);
        
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
        Named("all_iterations") = results
    );
}