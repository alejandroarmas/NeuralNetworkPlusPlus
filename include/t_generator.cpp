#ifndef TEMPLATED_GENERATOR_IMPLEMENTATION
#define TEMPLATED_GENERATOR_IMPLEMENTATION

#include <memory>
#include <random>
#include <functional>
#include <algorithm>

#include "config.h"
#include "generator.h"


template <int M, int V> 
std::unique_ptr<Matrix::Representation> Matrix::Generation::Normal<M, V>::operator() (std::unique_ptr<Matrix::Representation> m){

    std::random_device rd{};
    std::mt19937 gen{rd()};

    std::normal_distribution<> d{M, V};

    for (uint64_t c = 0; c < m->num_cols(); c++) {

        for (uint64_t r = 0; r < m->num_rows(); r++) {
            m->put(r, c, DAMPEN * d(gen));
        }
    }

    return m;
}

#endif // TEMPLATED_GENERATOR_IMPLEMENTATION