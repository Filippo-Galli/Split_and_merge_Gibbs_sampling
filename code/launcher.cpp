#include "./common_functions.hpp"
#include "./split_merge.hpp"
#include "./neal8.hpp"
#include "./hyperg.hpp"

// [[Rcpp::export]]
List run_markov_chain(NumericMatrix data, IntegerVector attrisize, double gamma,
                      NumericVector v, NumericVector w, int verbose = 0,
                      int m = 5, int iterations = 1000, int L = 1,
                      Rcpp::Nullable<Rcpp::IntegerVector> c_i = R_NilValue,
                      int burnin = 5000, int t = 10, int r = 10,
                      bool neal8 = false, bool split_merge = true,
                      int n8_step_size = 1, int sam_step_size = 1,
                      int thinning = 1) {
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
  if (c_i.isNotNull()) {
    Rcpp::Rcout << "Initial cluster assignments provided" << std::endl;
    initial_c_i =
        as<IntegerVector>(c_i) -
        min(as<IntegerVector>(
            c_i)); // rescale input cluster assignments to start from 0
    state.total_cls = unique_classes(initial_c_i).length();

  } else {
    initial_c_i = sample_initial_assignment(L, const_data.n);
  }
  state.c_i = clone(initial_c_i);

  // Initialize centers
  state.center = sample_centers(state.total_cls, const_data.attrisize);
  // Initialize sigma
  state.sigma = sample_sigmas(state.total_cls, const_data);

  // Fit the parameters on the cluster assignments
  update_phi(state, const_data);

  if (verbose == 2 or verbose == 1) {
    print_internal_state(state);
  }

  List results = List::create(
      Named("total_cls") = List(iterations), Named("c_i") = List(iterations),
      Named("centers") = List(iterations), Named("sigmas") = List(iterations),
      Named("loglikelihood") = NumericVector(iterations),
      Named("final_ass") = IntegerVector(data.nrow()),
      Named("time") = IntegerVector(1),
      Named("accepted") = IntegerVector(iterations));

  Rcpp::Rcout << "Sampling all the latent cluster in advance... " << std::endl;

  const size_t latent_size = const_data.n * m * thinning;
  std::vector<NumericVector> latent_center_reuse;
  latent_center_reuse.reserve(latent_size);
  std::vector<NumericVector> latent_sigma_reuse;
  latent_sigma_reuse.reserve(latent_size);
  int accepted = 0;

  for (size_t i = 0; i < latent_size; ++i) {
    latent_center_reuse.emplace_back(sample_center_1_cluster(const_data.attrisize));
    latent_sigma_reuse.emplace_back(sample_sigma_1_cluster(const_data.attrisize, const_data.v, const_data.w));
  }

  auto start_time = std::chrono::steady_clock::now();
  Rcpp::Rcout << "\nStarting Markov Chain sampling..." << std::endl;

  int idx_1_sm = 0;

  try {
    for (int iter = 0; iter < (iterations + burnin) * thinning; ++iter) {
      accepted = 0;

      if (verbose != 0)
        Rcpp::Rcout << std::endl
                    << "[DEBUG] - Iteration " << iter << " of "
                    << iterations + burnin << std::endl;

      // Sample new cluster assignments for each observation
      if (neal8 && iter % n8_step_size == 0) {
        for (int index_i = 0; index_i < const_data.n; index_i++) {
          // Sample new cluster assignment for observation i
          sample_allocation(index_i, const_data, state, m, latent_center_reuse,
                            latent_sigma_reuse);
        }

        // Update centers and sigmas
        update_phi(state, const_data);
      }

      if (verbose == 2) {
        std::cout << "State after Neal8" << std::endl;
        print_internal_state(state);
      }

      // Split and merge step
      if (split_merge && iter % sam_step_size == 0) {
        accepted = split_and_merge(state, const_data, t, r, idx_1_sm);
        idx_1_sm =
            (idx_1_sm + 1) % const_data.n; // reset to 0 if it reaches the end
      }

      if (verbose == 2) {
        std::cout << "State after Split and Merge" << std::endl;
        print_internal_state(state);
      }

      // Resampling parameters
      if (iter % 1000 == 0)
        for (size_t i = 0; i < latent_size; ++i) {
          latent_center_reuse[i] =
              std::move(sample_center_1_cluster(const_data.attrisize));
          latent_sigma_reuse[i] = std::move(sample_sigma_1_cluster(
              const_data.attrisize, const_data.v, const_data.w));
        }

      // Calculate likelihood
      double loglikelihood = compute_loglikelihood(state, const_data);

      // Update progress bar
      if (verbose == 0)
        print_progress_bar(iter + 1, (iterations + burnin) * thinning,
                           start_time);

      // Save results
      if (iter >= thinning * burnin and iter % thinning == 0) {
        as<List>(results["total_cls"])[(iter) / thinning - burnin] =
            state.total_cls;
        as<List>(results["c_i"])[(iter) / thinning - burnin] = clone(state.c_i);
        as<List>(results["centers"])[(iter) / thinning - burnin] =
            clone(state.center);
        as<List>(results["sigmas"])[(iter) / thinning - burnin] =
            clone(state.sigma);
        as<NumericVector>(
            results["loglikelihood"])[(iter) / thinning - burnin] =
            loglikelihood;
        as<IntegerVector>(results["accepted"])[(iter) / thinning - burnin] =
            accepted;
      }
    }
  } catch (const std::exception &e) {
    Rcpp::Rcout << "Error in the sampling: " << e.what() << std::endl;
    throw;
  } catch (...) {
    Rcpp::Rcout << "Unknown error during sampling" << std::endl;
    throw;
  }

  auto end_time = std::chrono::steady_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
  Rcpp::Rcout << std::endl
              << "Markov Chain sampling completed in: " << duration.count()
              << " s " << std::endl;

  results["final_ass"] = state.c_i;
  results["time"] = duration.count();

  return results;
}
