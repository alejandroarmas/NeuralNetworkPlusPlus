#ifndef MATRIX_GENERATOR_H
#define MATRIX_GENERATOR_H

#include <memory>

#include "mm.h"


namespace Matrix {

    template <class T>
    class BaseGenerator {

        public:
            virtual std::unique_ptr<Matrix::Representation<T>> operator() (std::unique_ptr<Matrix::Representation<T>> m) = 0;
    };


    template <class T>
    class Generator : public BaseGenerator<T> {

        public:
            std::unique_ptr<Matrix::Representation<T>> operator() (std::unique_ptr<Matrix::Representation<T>> m) override;
    };

    template class Generator<float>;
    template class Generator<double>;
    template class Generator<int8_t>;
    template class Generator<int16_t>;
    template class Generator<int32_t>;
    template class Generator<int64_t>;
    template class Generator<uint8_t>;
    template class Generator<uint16_t>;
    template class Generator<uint32_t>;
    template class Generator<uint64_t>;

}

#endif