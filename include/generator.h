#ifndef MATRIX_GENERATOR_H
#define MATRIX_GENERATOR_H

#include <memory>

#include "mm.h"


namespace Matrix {

    namespace Generation {

        template <class T>
        class Base {

            public:
                virtual std::unique_ptr<Matrix::Representation<T>> operator() (std::unique_ptr<Matrix::Representation<T>> m) = 0;
        };


        template <class T>
        class Normal : public Base<T> {

            public:
                Normal(double _m, double _v) : mean(_m), variance(_v) {}
                std::unique_ptr<Matrix::Representation<T>> operator() (std::unique_ptr<Matrix::Representation<T>> m) override;
            private:
                double  mean; 
                double variance;
        };

        template class Normal<float>;
        template class Normal<double>;
        template class Normal<int8_t>;
        template class Normal<int16_t>;
        template class Normal<int32_t>;
        template class Normal<int64_t>;
        template class Normal<uint8_t>;
        template class Normal<uint16_t>;
        template class Normal<uint32_t>;
        template class Normal<uint64_t>;


        template <class T>
        class Tester : public Base<T> {

            public:
                std::unique_ptr<Matrix::Representation<T>> operator() (std::unique_ptr<Matrix::Representation<T>> m) override;
            private:
        };

        template class Tester<float>;
        template class Tester<double>;
        template class Tester<int8_t>;
        template class Tester<int16_t>;
        template class Tester<int32_t>;
        template class Tester<int64_t>;
        template class Tester<uint8_t>;
        template class Tester<uint16_t>;
        template class Tester<uint32_t>;
        template class Tester<uint64_t>;

    }

}

#endif