#include "neal8.hpp"
#include <Rcpp.h>
#include <RcppGSL.h>
#include <chrono>
#include <sstream>
#include <iomanip>

using namespace Rcpp;

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
        time_str << secs << "s";
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

double minimum(const IntVec & vec) {
    /**
     * @brief Find the minimum value in a vector
     * @param vec Input vector
     * @return Minimum value
     */
    return *std::min_element(vec.begin(), vec.end());
}

// [[Rcpp::export]]
DoubleMat run_markov_chain(NumericMatrix data, IntVec attrisize, double gamma,  DoubleVec v,  DoubleVec w, 
                        IntVec c_i, int verbose = 0, int m = 5, int iterations = 1000,
                        int burnin = 5000, int t = 10, int r = 10, bool neal8 = false, bool split_merge = false) {
    /**
     * @brief Main Markov Chain Monte Carlo sampling function
     * @param data Input data matrix
     * @param attrisize Vector of attribute sizes
     * @param gamma Gamma parameter
     * @param v Vector of v parameters
     * @param w Vector of w parameters
     * @param verbose Verbosity level
     * @param m Number of latent classes
     * @param iterations Number of iterations
     * @param c_i Initial cluster assignments
     * @param burnin Number of burnin iterations
     * @param t Number of split attempts
     * @param r Number of merge attempts
     * @param neal8 Use Neal8 algorithm
     * @param split_merge Use split and merge step
     * @return DoubleMat containing final clustering results
    */

    // Initialize data
    Neal8 neal8_class(data, attrisize, gamma, v, w, m);

    // Set initial cluster assignments
    double temp = minimum(c_i);
    // Ensure cluster indices start from 0
    for(int i = 0; i < c_i.size(); i++)
        c_i[i] -= temp; 

    neal8_class.set_c(c_i);
    neal8_class.initialize_memory(); 

    // Initialize centers
    neal8_class.sample_center();

    // Initialize sigma
    neal8_class.sample_sigma();

    // Check if all is well set
    //neal8_class.print();
    
    DoubleMat results = DoubleMat::create(Named("total_cls") = DoubleMat(iterations),
                                Named("c_i") = DoubleMat(iterations),
                                Named("centers") = DoubleMat(iterations),
                                Named("sigmas") = DoubleMat(iterations), 
                                Named("loglikelihood") =  DoubleVec(iterations), 
                                Named("acceptance_ratio") =  DoubleVec(iterations));

    auto start_time =  std::chrono::steady_clock::now();
    Rcpp::Rcout << "\nStarting Markov Chain sampling..." << std::endl;

    for (int iter = 0; iter < iterations + burnin; iter++) {
        if(verbose != 0)
            std::cout << std::endl <<"[DEBUG] - Iteration " << iter << " of " << iterations + burnin << std::endl;

        // Sample new cluster assignments for each observation
        if(neal8){
            if(iter < burnin)
                neal8_class.step();
            else
                neal8_class.step(results, iter - burnin);

        }

        // Update progress bar
        if(verbose == 0)
            print_progress_bar(iter + 1, iterations + burnin, start_time);
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    Rcpp::Rcout << std::endl << "Markov Chain sampling completed in: "<< duration.count() << " s " << std::endl;
    
    return results;
}