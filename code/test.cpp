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
    return sample(K, n, true) -1;
}

int count_cluster_members(const IntegerVector& c_i, int exclude_index, int cls) {
   
    int n_i_z = 0;
    for (int i = 0; i < c_i.length(); i++) {
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

int sample_allocation(const int index_i, const aux_data & constant_data, 
                        const internal_state & state, const int m, const IntegerVector & unique_classes_without_i) {
    /**
     * @brief Sample new cluster assignment for a single data point
     * @param index_i Index of the data point
     * @param constant_data Auxiliary data for the MCMC algorithm
     * @param state Internal state of the MCMC algorithm
     * @param m Number of latent classes
     * @return New cluster assignment for the data point
     */

    NumericVector x_i = constant_data.data(index_i, _);

    // Calculate probabilities
    NumericVector probs(state.total_cls);
    NumericVector sigma_k;
    NumericVector center_k;

    // number of variable in cluster z without i
    int n_i_z = 0;

    for (int k = 0; k < state.total_cls; k++) {
        // Density calculation, in Neal paper is the F function
        double Hamming = 0;

        sigma_k = state.sigma[k]; // prendo le sigma del cluster k
        center_k = state.center[k]; // prendo i centri del cluster k

        for (int j = 0; j < x_i.length(); j++) {
            Hamming += dhamming(x_i[j], center_k[j], sigma_k[j], constant_data.attrisize[j], true);
        }

        // probability calculation for existing clusters
        if (k < unique_classes_without_i.length()) {
            // Count instances in the z cluster excluding the current point i
            n_i_z = count_cluster_members(state.c_i, index_i, unique_classes_without_i[k]);
            // Calculate probability
            probs[k] = n_i_z * std::exp(Hamming) ;
        }
        // probability calculation for latent clusters 
        else {
            probs[k] = constant_data.gamma/m * std::exp(Hamming);
        }
    }

    // Create the vector from 0 to num_cls 
    NumericVector cls(state.total_cls);
    for (int i = 0; i < state.total_cls; ++i) {
        cls[i] = i;
    }

    // Normalize probabilities
    probs = probs / sum(probs);

    // Sample new cluster assignment using probabilities calculated before
    return sample(cls, 1, true, probs)[0];
}

void clean_var(internal_state & state, const IntegerVector& attrisize) {
    // Efficiently find unique existing clusters
    IntegerVector existing_cls = unique_classes(state.c_i);
    int num_existing_cls = existing_cls.length();

    // Create a mapping for efficient cluster index lookup
    std::unordered_map<int, int> cls_to_new_index;
    /*
    for(int i = 0; i < num_existing_cls; ++i) {
        cls_to_new_index[existing_cls[i]] = i;
    }
    */

    int idx_temp = 0;
    for(int i = 0; i < num_existing_cls; ++i) {
        if(existing_cls[i] < num_existing_cls){
            cls_to_new_index[existing_cls[i]] = existing_cls[i];
        }
        else{
            // find the first available index for new clusters
            while(cls_to_new_index.find(idx_temp) == cls_to_new_index.end()){
                idx_temp++;
            }
            
            cls_to_new_index[existing_cls[i]] = idx_temp;
        }
    }

    // Preallocate new containers with the correct size
    List new_center(num_existing_cls);
    List new_sigma(num_existing_cls);

    // Efficiently copy centers and sigmas
    for(int i = 0; i < num_existing_cls; ++i) {
        new_center[i] = state.center[existing_cls[i]];
        new_sigma[i] = state.sigma[existing_cls[i]];
    }

    // Update state with new centers, sigmas, and cluster count
    state.center = std::move(new_center);
    state.sigma = std::move(new_sigma);
    state.total_cls = num_existing_cls;

    // Vectorized cluster index update using the mapping
    for(int i = 0; i < state.c_i.length(); i++) {
        auto it = cls_to_new_index.find(state.c_i[i]);
        if(it != cls_to_new_index.end()) {
            state.c_i[i] = it->second;
        }
    }
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
    //List prob_centers(state.total_cls);
    List prob_centers = Center_prob(const_data.data, as<NumericVector>(state.sigma[state.sigma.length() -1]), as<NumericVector>(const_data.attrisize));
    
    /*
     * Calculation of probabilities for each attribute and modality to be chosen
     */
    /*for (int c = 0; c < num_cls; ++c){ // for each cluster
        NumericVector sigma_values = as<NumericVector>(state.sigma[c]);
        List prob_center(const_data.attrisize.length());
        for (int i = 0; i < const_data.attrisize.length(); i++) { // for each attribute
            NumericVector probs(const_data.attrisize[i]);
            double den=0;
            double sigma_value = sigma_values[i];
            
            for (int a = 0; a < const_data.attrisize[i]; a++) { // for each modality

                double num = 0, costante = 0;
                double sumdelta = sum(const_data.data(_, i) == (a + 1));
                num = (sumdelta - const_data.n)/sigma_value;
                
                probs[a] = num;
            }     

            probs = probs - max(probs); // Proposta loro - gibbs_utility.cpp riga 34
            probs = exp(probs);
            den = sum(probs);
            
            for(int p = 0; p < probs.length(); p++){
                probs[p] = probs[p]/den;
            }     
            
            // Store probabilities for this attribute
            prob_center[i] = probs;

        }
        prob_centers[c] = prob_center;
    }*/
    
    /*
     * Sample new cluster centers using the probabilities calculated before
     */
    NumericVector attr_centers(const_data.attrisize.length());
    List prob_centers_cluster;

    List attri_List = Attributes_List(const_data.data, const_data.data.ncol());
    
    for (int i = 0; i < num_cls; i++) { 
        state.center[i] = clone(Samp_Center(attri_List, prob_centers, const_data.attrisize.length()));
    }
    
    // for each cluster
    /*
    for (int i = 0; i < num_cls; i++) {         
        // for each cluster
        prob_centers_cluster = as<List>(prob_centers[i]);

        for (int j = 0; j < const_data.attrisize.length(); j++) { 
            //std::cout << std::endl<<"[DEBUG] - Cluster " << i <<" attribute " << j << std::endl;
            int attrisize_j = const_data.attrisize[j];
            
            NumericVector prob_centers_attribute_j = as<NumericVector>(prob_centers_cluster[j]);
            std::cout << std::endl << "[DEBUG] - prob_centers_attribute_j: " << prob_centers_attribute_j << std::endl;
            attr_centers[j] = sample(attrisize_j, 1, true, prob_centers_attribute_j)[0];
            //std::cout << std::endl << "[DEBUG] - center: " << attr_centers[j] << std::endl;
        }

        // move to avoid problems with pointers
        state.center[i] = clone(attr_centers);
    }*/
}

void update_sigma(List & sigma, const List & centers, const IntegerVector & c_i, const aux_data & const_data) {

    /**
     * @brief Update cluster dispersions
     * @param sigma List of cluster dispersions
     * @param centers List of cluster centers
     * @param c_i Current cluster assignments
     * @param const_data Auxiliary data for the MCMC algorithm
     */

    int num_cls = sigma.length();

    for (int c = 0; c < num_cls; c++) { // for each cluster
        NumericVector sigmas_cluster = as<NumericVector>(sigma[c]);
        NumericVector centers_cluster = as<NumericVector>(centers[c]);
        NumericVector new_w(const_data.attrisize.length());
        NumericVector new_v(const_data.attrisize.length());

        NumericVector new_sigma_cluster(const_data.attrisize.length());
        for (int i = 0; i < const_data.attrisize.length(); ++i ){ // for each attribute
            double sumdelta = std::count(const_data.data(_,i).begin(), const_data.data(_,i).end(), centers_cluster[i]);
            new_w[i] = const_data.w[i] + const_data.n - sumdelta;
            new_v[i] = const_data.v[i] + sumdelta;

            
        }
        new_sigma_cluster = sample_sigma_1_cluster(const_data.attrisize, new_v, new_w);
        sigma[c] = clone(new_sigma_cluster);
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
                    Rcpp::Nullable<Rcpp::IntegerVector> c_i = R_NilValue) {
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

    int burnin = 5000;
    aux_data const_data = {data, data.nrow(), attrisize, gamma, v, w};
    internal_state state = {IntegerVector(), List(), List(), L};

    // Initialize cluster assignments
    IntegerVector initial_c_i;
    if(c_i.isNotNull()) {
        Rcpp::Rcout << "Initial cluster assignments provided" << std::endl;
        initial_c_i = as<IntegerVector>(c_i);
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
                                Named("loglikelihood") = NumericVector(iterations));

    auto start_time = std::chrono::high_resolution_clock::now();
    Rcpp::Rcout << "Starting Markov Chain sampling..." << std::endl;
    
    int n_update_latent = 0; 
    for (int iter = 0; iter < iterations + burnin; iter++) {
        if(verbose != 0)
            std::cout << std::endl <<"[DEBUG] - Iteration " << iter << " of " << iterations << std::endl;

        L = unique_classes(state.c_i).length();

        // Add latent classes
        for(int i = 0; i < m; i++){
            state.center.push_back(sample_center_1_cluster(const_data.attrisize));
            state.sigma.push_back(sample_sigma_1_cluster(const_data.attrisize, const_data.v, const_data.w));
        }
        
        // update total number of clusters
        state.total_cls = state.center.length();

        // Sample new cluster assignments
        for (int index_i = 0; index_i < const_data.n; index_i++) {
            IntegerVector unique_classes_without_i = unique_classes_without_index(state.c_i, index_i);
            
            // If observation i create a new cluster update m-1 latent cls else m latent cls
            n_update_latent = L;
            if(unique_classes(state.c_i).length() != unique_classes_without_i.length()){
                n_update_latent = L + 1;
            }

            for(int i = n_update_latent; i < state.total_cls; i++){;
                as<List>(state.center)[i] = std::move(sample_center_1_cluster(const_data.attrisize));
                as<List>(state.sigma)[i] = std::move(sample_sigma_1_cluster(const_data.attrisize, const_data.v, const_data.w));
            }
            

            state.c_i[index_i] = sample_allocation(index_i, const_data, state, m, unique_classes_without_i);

            if(verbose == 3)
                std::cout << "[DEBUG] - new cluster assignment - "<< index_i << " : " << state.c_i[index_i] << std::endl;
        } 

        // Clean variables
        clean_var(state, const_data.attrisize);
        
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