#include "neal8.hpp"

Neal8::Neal8(const NumericMatrix & data, const IntVec & attrisize, const double gamma, const  DoubleVec & v, const  DoubleVec & w, const int & m): Updater(data, attrisize, gamma, v, w, m) {}

double Neal8::step() {
    /**
     * @brief Perform a single step of the Neal8 algorithm
     * @note This function performs a single step of the Neal8 algorithm
     */

    DEBUG_PRINT(1, "Starting Neal8 step");
    // Sample the cluster assignments
    sample_c();
    DEBUG_PRINT(1, "Passed sample_c");

    // Sample the centers
    update_center();
    DEBUG_PRINT(1, "Passed update_center");

    // Sample the sigmas
    update_sigma();
    DEBUG_PRINT(1, "Passed update_sigma");

    // Loglikelihood calculation
    return compute_loglikelihood();
}

void Neal8::step(DoubleMat & logs, const int & iter) {
    /**
     * @brief Perform a single step of the Neal8 algorithm
     * @param logs DoubleMat to store the loglikelihood
     * @note This function performs a single step of the Neal8 algorithm
     */

    // Sample the cluster assignments
    sample_c();

    // Sample the centers
    update_center();

    // Sample the sigmas
    update_sigma();

    // Save the results
    as<DoubleMat>(logs["total_cls"])[iter] = get_total_cls();
    as<DoubleMat>(logs["c_i"])[iter] = clone(get_c());
    as<DoubleMat>(logs["centers"])[iter] = clone(get_center());
    as<DoubleMat>(logs["sigmas"])[iter] = clone(get_sigma());
    as<DoubleMat>(logs["loglikelihood"])[iter] = compute_loglikelihood();
}   