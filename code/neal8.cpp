/**
 * @file neal8.cpp
 * @brief Implementation of Neal's Algorithm 8 for Dirichlet Process Mixture Models
 */

#include <chrono>
#include "split_merge.hpp"

using namespace Rcpp;

void sample_allocation(int idx, const aux_data & const_data, internal_state & state, const int m, 
                        const std::vector<NumericVector> & latent_center_reuse, const std::vector<NumericVector> & latent_sigma_reuse) {
    /**
     * @brief Sample new cluster assignment for observation i
     * @param idx Index of the observation
     * @param const_data Auxiliary data for the MCMC algorithm
     * @param state Internal state of the MCMC sampler
     * @param m Number of latent classes
     * @note This function is used to sample new cluster assignments for each observation
     *       based on the current state of the MCMC sampler
     */
    // Get data point
    const NumericVector & y_i = const_data.data(idx, _);
    const IntegerVector & uni_clas = unique_classes(state.c_i);
    const IntegerVector & unique_classes_without_i = unique_classes_without_index(state.c_i, idx);
    const int k = uni_clas.length(); // numeri di cluster attivi
    const int k_minus = unique_classes_without_i.length(); // numeri di cluster attivi escludendo l'osservazione corrente

    NumericVector probs(k + m);

    // Calculate allocation probabilities for existing clusters
    double log_likelihood = 0.0;
    for(int i = 0; i < k; ++i){
        log_likelihood = 0.0;
        
        const NumericVector & sigma_k = state.sigma[i];
        const NumericVector & center_k = state.center[i];

        // Calculate likelihood
        for (int j = 0; j < y_i.length(); j++) {
            log_likelihood += dhamming_pippo(y_i[j], center_k[j], sigma_k[j], const_data.attrisize[j]);
        }

        // Count instances of element in the cluster i excluding idx
        int n_i_z = sum(state.c_i == i) - (state.c_i[idx] == i);

        // Set probability
        probs[i] = n_i_z != 0 ? log(n_i_z) + log_likelihood : -INFINITY ; //if n_i_z == 0 then probs[i] = 0
    }

    // Sample Latent Cluster
    std::vector<NumericVector> latent_centers;
    latent_centers.reserve(m);
    std::vector<NumericVector> latent_sigmas;
    latent_sigmas.reserve(m);

    // initialize latent clusters
    for(int i = 0; i < m; ++i){
        int idx_latent = sample(latent_center_reuse.size(), 1, false)[0] - 1;
        latent_centers.push_back(latent_center_reuse[idx_latent]);
        latent_sigmas.push_back(latent_sigma_reuse[idx_latent]);
    }
    // Se l'osservazione è unica, il suo cluster diventa la prima classe latente
    if(k_minus < k){  
        latent_centers[0] = state.center[state.c_i[idx]];
        latent_sigmas[0] = state.sigma[state.c_i[idx]];
    }

    // Calculate allocation probabilities for latent clusters
    const double log_factor = std::log(const_data.gamma/m);
    for(int i = 0; i < m; ++i){
        log_likelihood = 0.0;
        
        const NumericVector & sigma_k = latent_sigmas[i];
        const NumericVector & center_k = latent_centers[i];

        // Calculate likelihood
        for (int j = 0; j < y_i.length(); j++) {
            log_likelihood += dhamming_pippo(y_i[j], center_k[j], sigma_k[j], const_data.attrisize[j]);
        }

        // probability calculation
        probs[k + i] = log_factor + log_likelihood;
    }

    // Normalize probabilities
    probs = exp((probs - max(probs)));
    probs = probs / sum(probs);
    //Rcpp::Rcout << "Probs:" << probs << std::endl;

    // Sample new allocation
    NumericVector cls(k + m);
    std::iota(cls.begin(), cls.end(), 0); // fill cls with 0, 1, 2, ..., k + m - 1

    int new_cls = sample(cls, 1, true, probs)[0];
    int old_cls = state.c_i[idx];

    // Update center and sigma
    // Caso 1: prendiamo una classe nota e non togliamo nessuna classe (non stiamo analizzando un'osservazione unica)
    if( sum(state.c_i == state.c_i[idx]) != 1 && new_cls < k){
        // Update allocation
        state.c_i[idx] = new_cls;
        validate_state(state, "Neal8 case 1");
        return;
    }

    // Caso 2: prendiamo una classe nota e togliamo una classe (stiamo analizzando un'osservazione unica)
    if( new_cls < k && sum(state.c_i == state.c_i[idx]) == 1 ){
        // Update allocation
        state.c_i[idx] = new_cls;
        // sostituisco la vecchia classe con l'ultima classe attiva
        state.center[old_cls] = std::move(state.center[k - 1]);
        state.sigma[old_cls] = std::move(state.sigma[k - 1]);

        // elimino l'ultima classe attiva
        state.center.erase(k - 1);
        state.sigma.erase(k - 1);

        // aggiorno le allocazioni 
        for(int i = 0; i < const_data.n; ++i){
            if(state.c_i[i] == (k - 1)){
                state.c_i[i] = old_cls;
            }
        }

        // Aggiorno il numero di classi attive
        state.total_cls = k - 1;
        validate_state(state, "Neal8 case 2");
        return;
    }

    // Caso 3: prendiamo una classe latente e non togliamo nessuna classe
    if( new_cls >= k && sum(state.c_i == state.c_i[idx]) != 1){
        // Update allocation
        state.c_i[idx] = k;
        state.center.push_back(latent_centers[new_cls - k]);
        state.sigma.push_back(latent_sigmas[new_cls - k]);

        // Aggiorno il numero di classi attive
        state.total_cls += 1;
        validate_state(state, "Neal8 case 3");
        return;
    }

    // Caso 4: prendiamo una classe latente e togliamo una classe
    if (new_cls >= k && sum(state.c_i == state.c_i[idx]) == 1){
        // sostituisco la vecchia classe con la classe latente
        state.center[old_cls] = std::move(latent_centers[new_cls - k]);
        state.sigma[old_cls] = std::move(latent_sigmas[new_cls - k]);
        validate_state(state, "Neal8 case 4");
        return;
    }
}

// [[Rcpp::export]]
List run_markov_chain(NumericMatrix data, IntegerVector attrisize, double gamma, NumericVector v, NumericVector w, 
                    int verbose = 0, int m = 5, int iterations = 1000, int L = 1, 
                    Rcpp::Nullable<Rcpp::IntegerVector> c_i = R_NilValue, int burnin = 5000, int t = 10, int r = 10, bool neal8=false, 
                    bool split_merge = true, int n8_step_size = 1, int sam_step_size = 1, int thinning = 1) {
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
                                Named("final_ass") = IntegerVector(data.nrow()),
                                Named("time") = IntegerVector(1),
                                Named("accepted") = IntegerVector(iterations)                          
                                );

    Rcpp::Rcout << "Sampling all the latent cluster in advance... " << std::endl; 

    const size_t latent_size = const_data.n*m*thinning;
    std::vector<NumericVector> latent_center_reuse;
    latent_center_reuse.reserve(latent_size);
    std::vector<NumericVector> latent_sigma_reuse;
    latent_sigma_reuse.reserve(latent_size);
    int accepted = 0;

    for(size_t i = 0; i < latent_size; ++i) {
        latent_center_reuse.emplace_back(sample_center_1_cluster(const_data.attrisize));
        latent_sigma_reuse.emplace_back(sample_sigma_1_cluster(const_data.attrisize, const_data.v, const_data.w));
    }

    auto start_time =  std::chrono::steady_clock::now();
    Rcpp::Rcout << "\nStarting Markov Chain sampling..." << std::endl;

    int idx_1_sm = 0;

    try{
        for (int iter = 0; iter < (iterations + burnin)*thinning; ++iter) {
            accepted = 0;
            
            if(verbose != 0)
                Rcpp::Rcout << std::endl <<"[DEBUG] - Iteration " << iter << " of " << iterations + burnin << std::endl;

            // Sample new cluster assignments for each observation
            if(neal8 && iter%n8_step_size == 0){
                for (int index_i = 0; index_i < const_data.n; index_i++) {
                    // Sample new cluster assignment for observation i
                    sample_allocation(index_i, const_data, state, m, latent_center_reuse, latent_sigma_reuse);       
                } 
                
                // Update centers and sigmas
                update_centers(state, const_data);
                update_sigma(state, const_data);
            }

            if(verbose == 2){
                std::cout << "State after Neal8" << std::endl;
                print_internal_state(state);
            }

            // Split and merge step
            if(split_merge && iter%sam_step_size==0){
                accepted = split_and_merge(state, const_data, t, r, idx_1_sm);
                idx_1_sm = (idx_1_sm + 1) % const_data.n; // reset to 0 if it reaches the end
            }

            if(verbose == 2){
                std::cout << "State after Split and Merge" << std::endl;
                print_internal_state(state);
            }

            // Resampling parameters
            if(iter % 1000 == 0)
                for(size_t i = 0; i < latent_size; ++i) {
                    latent_center_reuse[i] = std::move(sample_center_1_cluster(const_data.attrisize));
                    latent_sigma_reuse[i] = std::move(sample_sigma_1_cluster(const_data.attrisize, const_data.v, const_data.w));
                }

            // Calculate likelihood
            double loglikelihood = compute_loglikelihood(state, const_data);

            // Update progress bar
            if(verbose == 0)
                print_progress_bar(iter + 1, (iterations + burnin)*thinning, start_time);

            // Save results
            if(iter >= thinning*burnin and iter % thinning == 0){
                as<List>(results["total_cls"])[(iter)/thinning - burnin] = state.total_cls;
                as<List>(results["c_i"])[(iter)/thinning - burnin] = clone(state.c_i);
                as<List>(results["centers"])[(iter)/thinning - burnin] = clone(state.center);
                as<List>(results["sigmas"])[(iter)/thinning - burnin] = clone(state.sigma);
                as<NumericVector>(results["loglikelihood"])[(iter)/thinning - burnin] = loglikelihood;
                as<IntegerVector>(results["accepted"])[(iter)/thinning - burnin] = accepted;              
            }
        }
    } catch (const std::exception& e) {
        Rcpp::Rcout << "Error in the sampling: " << e.what() << std::endl;
        throw;
    } catch (...) {
        Rcpp::Rcout << "Unknown error during sampling" << std::endl;
        throw;
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    Rcpp::Rcout << std::endl << "Markov Chain sampling completed in: "<< duration.count() << " s " << std::endl;

    results["final_ass"] = state.c_i;
    results["time"] = duration.count();
    
    return results;
}
