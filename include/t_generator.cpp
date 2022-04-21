#ifndef TEMPLATED_GENERATOR_IMPLEMENTATION
#define TEMPLATED_GENERATOR_IMPLEMENTATION

#include <memory>
#include <random>
#include <functional>
#include <algorithm>

#include "config.h"
#include "generator.h"


template <int Mean, int Variance> 
Matrix::Representation Matrix::Generation::Normal<Mean, Variance>::operator() (Matrix::Representation& m){

    std::random_device rd{};
    std::mt19937 gen{rd()};

    std::normal_distribution<> d{Mean, Variance};

    std::transform(m.scanStart(), m.scanEnd(), m.scanStart(), [&gen, &d](const auto _){ return DAMPEN * d(gen); });

    return m;
}


template <int Val> 
Matrix::Representation Matrix::Generation::Tester<Val>::operator() (Matrix::Representation& m) {
 
    std::transform(m.scanStart(), m.scanEnd(), m.scanStart(), [](auto _){ return Val; });

    return m;
}


#endif // TEMPLATED_GENERATOR_IMPLEMENTATION