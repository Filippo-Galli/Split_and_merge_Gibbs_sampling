/**
 * @file neal_sampler.cpp
 * @brief Implementation of Markov Chain Monte Carlo clustering algorithm based on Neal algorithm 8 and Argiento paper 2022 using Rcpp
 * @details This file contains the implementation of a MCMC-based clustering algorithm
 *          specifically designed for categorical data using Hamming distance.
 */

#include <Rcpp.h>
#include <set>
#include <random>
#include <algorithm>
#include <gibbs_utility.cpp>
#include <RcppGSL.h>  // Add this explicit RcppGSL include
#include <gsl/gsl_sf_hyperg.h>
#include <hyperg.cpp>
#include <Rinternals.h>
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

    return wrap(std::vector<double>(unique_classes.begin(), unique_classes.end()));
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
        c_i[i] = sample(K, 1, true)[0] - 1; // with -1 we have the index starting from 0
    }
    return c_i;
}

int count_cluster_members(const NumericVector& c_i, int exclude_index) {
   
    int n_i_z = 0;
    for (int i = 0; i < c_i.length(); i++) {
        if (i != exclude_index && c_i[i] == c_i[exclude_index]) {
            n_i_z++;
        }
    }
    
    return n_i_z;
}

NumericVector sample_center_1_cluster(const NumericVector & attrisize) {
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

List sample_centers(const int number_cls, const NumericVector & attrisize) {
    /**
     * @brief Sample initial cluster centers
     * @param number_cls Number of clusters
     * @param attrisize Vector of attribute
     * @param num_cls_sample Number of clusters to sample (default: -1)
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

List sample_sigmas(const int number_cls, const NumericVector & attrisize, const NumericVector & u, const NumericVector & v) {
    /**
     * @brief Sample initial cluster dispersions (sigma)
     * @param attrisize Vector of attribute sizes
     * @param number_cls Number of clusters
     * @param u Parameter for hypergeometric distribution
     * @param v Parameter for hypergeometric distribution
     * @return List of cluster dispersions for each attribute
     */
    List sigma(number_cls);

    for (int i = 0; i < number_cls; i++) {
        sigma[i] = rhyper_sig(attrisize.length(), v[i], u[i], attrisize[i]);
    } 

    return sigma;
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

    // Calculate probabilities
    NumericVector probs(num_cls);
    NumericVector sigma_k;
    NumericVector center_k;

    // number of variable in cluster z without i
    int n_i_z = 0;

    for (int k = 0; k < num_cls; k++) {
        // Density calculation, in Neal paper is the F function
        double Hamming = 0;

        sigma_k = sigma[k]; // prendo le sigma del cluster k
        center_k = center[k]; // prendo i centri del cluster k

        for (int j = 0; j < x_i.length(); j++) {
            Hamming += dhamming(x_i[j], center_k[j], sigma_k[j], attrisize[j], true);
        }

        // probability calculation for existing clusters
        if (k < unique_classes_without_i.length()) {
            // Count instances in the z cluster excluding the current point i
            n_i_z = count_cluster_members(c_i, index_i);
            // Calculate probability
            probs[k] = n_i_z * std::exp(Hamming) ;
        }
        // probability calculation for latent clusters 
        else {
            probs[k] = gamma/m * std::exp(Hamming);
        }
    }


    // Create the vector from 0 to num_cls 
    NumericVector cls(num_cls);
    for (int i = 0; i < num_cls; ++i) {
        cls[i] = i;
    }

    double b = 1/(sum(probs));
    probs = probs * b;
    if(sum(probs) > 1.1){
        std::cout << "[DEBUG] - probability of cluster assignment > 1 - idx" << index_i << " : " << probs << std::endl;
    }
    
    if(std::accumulate(std::begin(probs), std::end(probs), 1.0, std::multiplies<double>()) < 0 ){
        std::cout << "[DEBUG] - Negative probability of cluster assignment  - idx" << index_i << " : " << probs << std::endl;
    }


    // Sample new cluster assignment using probabilities calculated before
    return sample(cls, 1, true, probs)[0];
}

void clean_var(List& center, List& sigma, NumericVector& c_i, NumericVector& attrisize) {
    // Vector of existing clusters
    NumericVector existing_cls = unique_classes(c_i);
    /*
    for(int k = 0; k < existing_cls.length(); k++){
        std::cout << "Existing cluster " << existing_cls[k] << " - " ;
    }
    std::cout << std::endl;
    */

    List new_center;
    List new_sigma;

    for(int i = 0; i < center.length(); ++i){
        if(std::find(existing_cls.begin(), existing_cls.end(), i) != existing_cls.end()){
            //std::cout << "[DEBUG] - Keeping cluster index " << i << std::endl;
            new_center.push_back(center[i]);
            new_sigma.push_back(sigma[i]);
        }
    }

    // Update the center with the cleaned values
    center = new_center;
    sigma = new_sigma;
}

void update_centers(List & centers,const NumericMatrix & data, const NumericVector & attrisize, 
                    const NumericVector & c_i, const List & sigma_prec) {
    /**
     * @brief Update cluster centers
     * @param data Input data matrix
     * @param attrisize Vector of attribute sizes
     * @param c_i Current cluster assignments
     * @param sigma_prec Previous sigma values
     * @return List of updated cluster centers
     */
    NumericVector clusters = unique_classes(c_i);
    int num_cls = sigma_prec.length();
    List prob_centers(num_cls);
    int n = data.nrow();
    //std::cout << std::endl << std::endl <<"[DEBUG] - New Call " << std::endl;
    
    /*
     * Calculation of probabilities for each attribute and modality to be chosen
     */
    for (int c = 0; c < num_cls; ++c){ // for each cluster
        //std::cout << "[DEBUG] ----------------------------------- " << "Cluster " << c << " of " << num_cls<< std::endl;
        NumericVector sigma_values = as<NumericVector>(sigma_prec[c]);
        List prob_center(attrisize.length());
        for (int i = 0; i < attrisize.length(); i++) { // for each attribute
            NumericVector probs(attrisize[i]);
            double den=0;
            double sigma_value = sigma_values[i];
            
            for (int a = 0; a < attrisize[i]; a++) { // for each modality

                double num = 0, sumdelta = 0, costante = 0;
                //std::cout << "[DEBUG] - " << " Modalità "<<a<<std::endl;
                sumdelta = std::count(data(_,i).begin(),data(_,i).end(), a + 1);
                //std::cout << "[DEBUG] - " << " sumdelta: " << sumdelta << std::endl;
                num = (sumdelta - n)/sigma_value;
                //std::cout << "[DEBUG] - " << " num: " << num << std::endl;
                
                probs[a] = num;
            }     

            probs = probs - max(probs); // Proposta loro - gibbs_utility.cpp riga 34
            probs = exp(probs);
            den = sum(probs);
            
            for(int p = 0; p < probs.length(); p++){
                probs[p] = probs[p]/den;
                //std::cout << "[DEBUG] - " << " prob modalità "<<p<<": " << probs[p] << std::endl;
            }     
            
            // Store probabilities for this attribute
            prob_center[i] = probs;
            //std::cout << std::endl;
        }
        prob_centers[c] = prob_center;
    }
    
    /*
     * Sample new cluster centers using the probabilities calculated before
     */
    NumericVector attr_centers(attrisize.length());
    List prob_centers_attribute;
    
    
    // for each cluster
    for (int i = 0; i < num_cls; i++) {         
        // for each cluster
        prob_centers_attribute = as<List>(prob_centers[i]);

        for (int j = 0; j < attrisize.length(); j++) { 
            //std::cout << std::endl<<"[DEBUG] - Cluster " << i <<" attribute " << j << std::endl;
            int attrisize_j = attrisize[j];
            
            NumericVector prob_centers_attribute_j = as<NumericVector>(prob_centers_attribute[j]);

            if(sum(prob_centers_attribute_j) > 1.1){
                std::cout << "[DEBUG] - probability of center > 1 - attribute " << j << " : " << prob_centers_attribute_j << std::endl;
            }
            
            if(std::accumulate(std::begin(prob_centers_attribute_j), std::end(prob_centers_attribute_j), 1.0, std::multiplies<double>()) < 0 ){
                std::cout << "[DEBUG] - Negative probability of center < 0  - attribute" << j << " : " << prob_centers_attribute_j << std::endl;
            }
            
            /*
            // To print the probabilities
            for(int k = 0; k < attrisize_j; k++){
                std::cout << "Count: " << std::count(data(_,i).begin(),data(_,i).end(), k + 1) << " Prob: "<<  prob_centers_attribute_j[k] << std::endl;
            }
            */
            
            
            attr_centers[j] = sample(attrisize_j, 1, true, prob_centers_attribute_j)[0];
            //std::cout << std::endl << "[DEBUG] - center: " << attr_centers[j] << std::endl;
        }
        centers[i] = attr_centers;
    }
}

void update_sigma(List & sigma, const List & centers, const NumericMatrix & data, const NumericVector & attrisize, 
                    const NumericVector & c_i, const NumericVector & v, const NumericVector & w) {
    /**
     * @brief Update cluster dispersions using the original method
     * @param sigma List of current cluster dispersions
     * @param centers List of current cluster centers
     * @param data Input data matrix
     * @param attrisize Vector of attribute sizes
     * @param c_i Current cluster assignments
     * @param v Parameter for dispersion not updated
     * @param w Parameter for dispersion not updated
     * @return List of updated cluster dispersions
     */
    int num_cls = sigma.length();

    List new_sigma(num_cls);

    for (int c = 0; c < num_cls; c++) { // for each cluster
        NumericVector sigmas_cluster = as<NumericVector>(sigma[c]);
        NumericVector centers_cluster = as<NumericVector>(centers[c]);

        NumericVector new_sigma_cluster(attrisize.length());
        for (int i = 0; i < attrisize.length(); ++i ){ // for each attribute
            double sumdelta = std::count(data(_,i).begin(),data(_,i).end(), centers_cluster[i]);
            double new_w = w[i] + data.nrow() - sumdelta;
            double new_v = v[i] + sumdelta;
            new_sigma_cluster[i] = rhyper_sig(1, new_v, new_w, attrisize[i])[0];
        }
        new_sigma[c] = new_sigma_cluster;
    }

    sigma = new_sigma;
}

// [[Rcpp::export]]
List run_markov_chain(NumericMatrix data, NumericVector attrisize, double gamma,
                    NumericVector v, NumericVector w, int verbose = 0, int m = 5, int iterations = 1000) {
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
    int n = data.nrow();

    // First cluster assignments
    int L = 4;
    int total_cls = L + m;

    // Initialize cluster assignments
    NumericVector c_i = sample_initial_assignment(L, n);

    if(verbose == 1){
        std::cout << "[DEBUG] - " << " c_i[i]: " << std::endl;
        for (int i = 0; i < n; i++) {  
            std::cout << c_i[i] << " ";
        }
        std::cout << std::endl;
    }
  
    // Initialize centers
    List center = sample_centers(L, attrisize);

    if(verbose == 1){
        std::cout << "\n[DEBUG] - " << " center[i]: " << std::endl;
        for (int i = 0; i < L + m; i++) {  
            std::cout << "Center of cluster " << i << " :"<< std::endl << "\t";
            NumericVector temp = as<NumericVector>(center[i]);
            for(int j = 0; j < attrisize.length(); j++){
                std::cout << temp[j] << " ";
            }
            std::cout << std::endl;
        }
    }

    // Initialize sigma
    List sigma = sample_sigmas(L, attrisize, v, w);
    List sigma_prec;

    if(verbose == 1){
        std::cout << "\n[DEBUG] - " << " sigma[i]: " << std::endl;
        for (int i = 0; i < L + m; i++) {  
            std::cout << "Sigma of cluster " << i << " :"<< std::endl << "\t";
            NumericVector temp = as<NumericVector>(sigma[i]);
            for(int j = 0; j < attrisize.length(); j++){
                std::cout << temp[j] << " ";
            }
            std::cout << std::endl;
        }
    }

    Rcpp::Rcout << "Starting Markov Chain sampling..." << std::endl;

    // Store results for each iteration
    List results(iterations);
    List iteration_result = List::create(
        Named("assignments") = clone(c_i),
        Named("centers") = clone(center),
        Named("sigma") = clone(sigma)
    );
    results[0] = iteration_result;
    
    for (int iter = 0; iter < iterations; iter++) {
        if(verbose != 0)
            std::cout << std::endl <<"[DEBUG] - Iteration " << iter << " of " << iterations << std::endl;

        // Add latent classes
        for(int i = 0; i < m; i++){
            center.push_back(sample_center_1_cluster(attrisize));
            sigma.push_back(sample_sigma_1_cluster(attrisize, v, w));
        }
        
        // update total number of clusters
        total_cls = center.length();

        for (int index_i = 0; index_i < n; index_i++) {
            NumericVector x_i = data(index_i, _);
            NumericVector unique_classes_without_i = unique_classes_without_index(c_i, index_i);
            NumericVector unique_classes_vec = unique_classes(c_i);

            c_i[index_i] = sample_allocation(index_i, x_i, c_i, data, center, sigma, attrisize, gamma, total_cls, unique_classes_without_i, m);
            
            if(verbose == 2)
                std::cout << "[DEBUG] - new cluster assignment - "<< index_i << " : " << c_i[index_i] << std::endl;
        } 

        /*
        std::cout << std::endl <<"[DEBUG] - Iteration " << iter + 1 << " of " << iterations << std::endl;
        std::cout << "[DEBUG] - c_i: " << std::endl;
        for(int i = 0; i < c_i.length(); i++){
            std::cout << c_i[i] << " ";
        }
        */

        // Clean variables
        clean_var(center, sigma, c_i, attrisize);
        
        // Update centers and sigmas
        update_centers(center, data, attrisize, c_i, sigma);
        update_sigma(sigma, center, data, attrisize, c_i, v, w);
        update_centers(center, data, attrisize, c_i, sigma);

        if(verbose == 2){
            std::cout << "\n[DEBUG] - " << " center[i]: " << std::endl;
            for (int i = 0; i < center.length(); i++) {  
                std::cout << "Center of cluster " << i << " :"<< std::endl << "\t";
                NumericVector temp = as<NumericVector>(center[i]);
                for(int j = 0; j < attrisize.length(); j++){
                    std::cout << temp[j] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "\n[DEBUG] - " << " sigma[i]: " << std::endl;
            for (int i = 0; i < sigma.length(); i++) {  
                std::cout << "sigma of cluster " << i << " :"<< std::endl << "\t";
                NumericVector temp = as<NumericVector>(sigma[i]);
                for(int j = 0; j < attrisize.length(); j++){
                    std::cout << temp[j] << " ";
                }
                std::cout << std::endl;
            }
        }

        // Update progress bar
        if(verbose == 0)
            print_progress_bar(iter + 1, iterations);
    }
    

    Rcpp::Rcout << std::endl << "Markov Chain sampling completed." << std::endl;
    return List::create(
       Named("final_assignments") = c_i,
       Named("final_centers") = center,
       Named("sigma") = sigma
       //Named("all_iterations") = results
    );
}