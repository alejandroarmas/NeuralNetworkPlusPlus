#ifndef MATRIX_MULTIPLY_H
#define MATRIX_MULTIPLY_H

#include <vector>
#include <cstdint>
#include <stdexcept>
#include <memory>



namespace Matrix {


    template <class T>
    class Representation {

        public:
            Representation(u_int64_t _l, u_int64_t _w) : rows(_l), columns(_w), data(std::move(std::vector<T>(_l * _w, 0))) {}
            u_int64_t num_rows() { return rows; }
            u_int64_t num_cols() { return columns; }
            T get(u_int64_t r, u_int64_t c);
            void put(u_int64_t r, u_int64_t c, T val);
            u_int64_t rows;
            u_int64_t columns;
            std::vector<T> data;
        private:
    };


    template class Representation<float>;
    template class Representation<double>;
    template class Representation<int8_t>;
    template class Representation<int16_t>;
    template class Representation<int32_t>;
    template class Representation<int64_t>;
    template class Representation<uint8_t>;
    template class Representation<uint16_t>;
    template class Representation<uint32_t>;
    template class Representation<uint64_t>;
    
}


namespace Matrix {


    namespace Operations {

        template <class T>
        class BaseOp {

            public:
                virtual std::unique_ptr<Matrix::Representation<T>> operator()(
                    Matrix::Representation<T> l, 
                    Matrix::Representation<T> r) = 0;

        };

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


#endif