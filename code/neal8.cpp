/**
 * @file neal8.cpp
 * @brief Implementation of Neal's Algorithm 8 for Dirichlet Process Mixture Models
 */

#include <chrono>
#include "split_merge.hpp"

using namespace Rcpp;

void sample_allocation(const int index_i, const aux_data & constant_data, 
                     internal_state & state, const int m, 
                     const IntegerVector & unique_classes_without_i) {
    /**
     * @brief Sample new cluster assignment for observation i
     * @param index_i Index of observation i
     * @param constant_data Auxiliary data for the MCMC algorithm
     * @param state Internal state of the MCMC algorithm
     * @param m Number of latent classes
     * @param unique_classes_without_i Vector of unique classes excluding the current observation
     */
    
    // Get data point
    const NumericVector & y_i = constant_data.data(index_i, _);

    const IntegerVector uni_clas = unique_classes(state.c_i);
    const int k_minus = unique_classes_without_i.length();
    const int m_temp = uni_clas.length() == k_minus ? m : m - 1;

    // Store current parameters for initialization
    NumericVector temp_center = as<List>(state.center)[state.c_i[index_i]];
    NumericVector temp_sigma = as<List>(state.sigma)[state.c_i[index_i]];

    // Create temporary state
    internal_state state_temp = {
        IntegerVector(state.c_i.length()), 
        List(state.center.length()), 
        List(state.sigma.length()), 
        0
    };
    
    // Clean and prepare temporary state
    clean_var(state_temp, state, unique_classes_without_i, constant_data.attrisize);

    // Handle case when point is alone in its cluster
    if(uni_clas.length() != k_minus){
        state_temp.center.push_back(temp_center);
        state_temp.sigma.push_back(temp_sigma);
    }

    // Add auxiliary parameters
    for(int i = 0; i < m_temp; i++){
        state_temp.center.push_back(sample_center_1_cluster(constant_data.attrisize));
        state_temp.sigma.push_back(sample_sigma_1_cluster(constant_data.attrisize, 
                                                        constant_data.v, 
                                                        constant_data.w));
    } 
    
    state_temp.total_cls = state_temp.center.length();
    
    // Calculate allocation probabilities
    NumericVector probs(state_temp.total_cls);
    NumericVector sigma_k(constant_data.attrisize.length());
    NumericVector center_k(constant_data.attrisize.length());
    
    // Probabilities for existing clusters
    for (int k = 0; k < k_minus; k++) {
        double log_likelihood = 0.0;
        
        sigma_k = state_temp.sigma[k];
        center_k = state_temp.center[k];

        // Calculate likelihood
        for (int j = 0; j < y_i.length(); j++) {
            log_likelihood += dhamming(y_i[j], center_k[j], sigma_k[j], 
                                    constant_data.attrisize[j], true);
        }

        // Count instances excluding current point
        int n_i_z = (k < unique_classes_without_i.length()) ? 
            count_cluster_members(state_temp.c_i, index_i, unique_classes_without_i[k]) : 0;

        // Set probability
        if(k < probs.length()){
            probs[k] = (n_i_z == 0) ? 0.0 : n_i_z * std::exp(log_likelihood);
        }
    }

    // Probabilities for auxiliary parameters
    for(int k = k_minus; k < state_temp.total_cls; k++){
        double log_likelihood = 0.0;
        
        sigma_k = state_temp.sigma[k];
        center_k = state_temp.center[k];

        // Calculate likelihood
        for (int j = 0; j < y_i.length(); j++) {
            log_likelihood += dhamming(y_i[j], center_k[j], sigma_k[j], 
                                    constant_data.attrisize[j], true);
        }

        probs[k] = constant_data.gamma/m * std::exp(log_likelihood);    
    }

    // Normalize probabilities
    NumericVector cls(state_temp.total_cls);
    for (int i = 0; i < state_temp.total_cls; ++i) {
        cls[i] = i;
    }
    probs = probs / max(probs);
    probs = probs / sum(probs);    

    // Sample new allocation
    state_temp.c_i[index_i] = sample(cls, 1, true, probs)[0];

    // Clean and update state
    clean_var(state, state_temp, unique_classes(state_temp.c_i), 
              constant_data.attrisize);
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
                                Named("accepted_merge") = IntegerVector(iterations), 
                                Named("accepted_split") = IntegerVector(iterations), 
                                Named("final_ass") = IntegerVector(data.nrow())                           
                                );

    auto start_time =  std::chrono::steady_clock::now();
    Rcpp::Rcout << "\nStarting Markov Chain sampling..." << std::endl;

    double acpt_ratio = 0.0;
    int accepted = 0; 
    int split_n = 0;
    int merge_n = 0;
    int accepted_merge = 0;
    int accepted_split = 0;

    for (int iter = 0; iter < iterations + burnin; iter++) {
        if(verbose != 0)
            std::cout << std::endl <<"[DEBUG] - Iteration " << iter << " of " << iterations + burnin << std::endl;

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
            split_and_merge(state, const_data, t, r, acpt_ratio, accepted, split_n, merge_n, accepted_merge, accepted_split);
        }

        if(verbose == 2){
            std::cout << "State after Split and Merge" << std::endl;
            print_internal_state(state);
        }

        // Calculate likelihood
        double loglikelihood = compute_loglikelihood(state, const_data);

        // Update progress bar
        if(verbose == 0)
            print_progress_bar(iter + 1, iterations + burnin, start_time);

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
            as<IntegerVector>(results["accepted_merge"])[iter - burnin] = accepted_merge;
            as<IntegerVector>(results["accepted_split"])[iter - burnin] = accepted_split;
            
        }
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    Rcpp::Rcout << std::endl << "Markov Chain sampling completed in: "<< duration.count() << " s " << std::endl;

    results["final_ass"] = state.c_i;
    
    return results;
}