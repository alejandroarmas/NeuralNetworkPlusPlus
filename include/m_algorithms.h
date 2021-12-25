#ifndef MATRIX_ALGORITHMS_H
#define MATRIX_ALGORITHMS_H

#include <algorithm>
#include <functional>

#include "mm.h"


namespace Matrix {


    namespace Operations {

        template <class T>
        class BaseOp {

            public:
                virtual std::unique_ptr<Matrix::Representation<T>> operator()(
                    Matrix::Representation<T> l, 
                    Matrix::Representation<T> r) = 0;

        };

        namespace Add {

            template <class T> 
            class BaseAdd : BaseOp<T> {

            };

            template <class T> 
            class Std : public BaseAdd<T> {
                public:
                    std::unique_ptr<Matrix::Representation<T>> operator()(
                        Matrix::Representation<T> l, 
                        Matrix::Representation<T> r) override;

            };

            template class Std<float>;
            template class Std<double>;
            template class Std<int8_t>;
            template class Std<int16_t>;
            template class Std<int32_t>;
            template class Std<int64_t>;
            template class Std<uint8_t>;
            template class Std<uint16_t>;
            template class Std<uint32_t>;
            template class Std<uint64_t>;
        }


        namespace HadamardProduct {

            template <class T> 
            class BaseHP : BaseOp<T> {

            };

            template <class T> 
            class Naive : public BaseHP<T> {

                public:
                    std::unique_ptr<Matrix::Representation<T>> operator()(
                        Matrix::Representation<T> l, 
                        Matrix::Representation<T> r) override;

            };

            template <class T> 
            class Std : public BaseHP<T> {

                public:
                    std::unique_ptr<Matrix::Representation<T>> operator()(
                        Matrix::Representation<T> l, 
                        Matrix::Representation<T> r) override;

            };



            template class Naive<float>;
            template class Naive<double>;
            template class Naive<int8_t>;
            template class Naive<int16_t>;
            template class Naive<int32_t>;
            template class Naive<int64_t>;
            template class Naive<uint8_t>;
            template class Naive<uint16_t>;
            template class Naive<uint32_t>;
            template class Naive<uint64_t>;

            template class Std<float>;
            template class Std<double>;
            template class Std<int8_t>;
            template class Std<int16_t>;
            template class Std<int32_t>;
            template class Std<int64_t>;
            template class Std<uint8_t>;
            template class Std<uint16_t>;
            template class Std<uint32_t>;
            template class Std<uint64_t>;
        }




        namespace Multiplication {

            template <class T> 
            class BaseHP : BaseOp<T> {

            };

            template <class T> 
            class Naive : public BaseHP<T> {

                public:
                    std::unique_ptr<Matrix::Representation<T>> operator()(
                        Matrix::Representation<T> l, 
                        Matrix::Representation<T> r) override;

            };



            template class Naive<float>;
            template class Naive<double>;
            template class Naive<int8_t>;
            template class Naive<int16_t>;
            template class Naive<int32_t>;
            template class Naive<int64_t>;
            template class Naive<uint8_t>;
            template class Naive<uint16_t>;
            template class Naive<uint32_t>;
            template class Naive<uint64_t>;
        }

    }

}

#endif // MATRIX_ALGORITHMS_H