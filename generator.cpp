
#include <memory>
#include <random>

#include "generator.h"


#define MEAN 1
#define VARIANCE 2
#define DAMPEN 0.01

template <class T>
std::unique_ptr<Matrix::Representation<T>> Matrix::Generator<T>::operator() (std::unique_ptr<Matrix::Representation<T>> m) {

    std::random_device rd{};
    std::mt19937 gen{rd()};

    std::normal_distribution<> d{MEAN, VARIANCE};

    for (uint64_t c = 0; c < m->num_cols(); c++) {

        for (uint64_t r = 0; r < m->num_rows(); r++) {
            m->put(r, c, DAMPEN * d(gen));
        }

    }

    return std::move(m);
}
