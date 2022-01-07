
#include <memory>
#include <random>
#include <functional>
#include <algorithm>

#include "config.h"
#include "generator.h"


std::unique_ptr<Matrix::Representation> Matrix::Generation::Normal::operator() (std::unique_ptr<Matrix::Representation> m) {

    std::random_device rd{};
    std::mt19937 gen{rd()};

    std::normal_distribution<> d{this->mean, this->variance};

    for (uint64_t c = 0; c < m->num_cols(); c++) {

        for (uint64_t r = 0; r < m->num_rows(); r++) {
            m->put(r, c, DAMPEN * d(gen));
        }
    }

    return m;
}


std::unique_ptr<Matrix::Representation> Matrix::Generation::Tester::operator() (std::unique_ptr<Matrix::Representation> m) {

    
    std::transform(m->scanStart(), m->scanEnd(), m->scanStart(), [](auto a){ return GENERATOR_TESTER_CONSTANT; });


    return m;
}
