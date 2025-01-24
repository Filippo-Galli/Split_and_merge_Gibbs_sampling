#ifndef NEAL8_HPP
#define NEAL8_HPP

#include "updater.hpp"

class Neal8 : public Updater {
    private:
    // Cache

    public:
    // Constructor
    Neal8(const NumericMatrix & data, const IntVec & attrisize, const double gamma, 
            const  DoubleVec & v, const  DoubleVec & w, const int & m);

    // Step functions
    double step();
    void step(DoubleMat & logs, const int & iter);

};

#endif