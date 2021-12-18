#include <iostream>
#include <iomanip>


#include "mm.h"

#define MAX_PRINT_WIDTH 5
#define MAX_PRINT_LENGTH 5



template <class T>
T Matrix::Representation<T>::get(u_int64_t r, u_int64_t c) {

                uint64_t calculated_index = c + r * columns; 

                if (r <= rows && c <= columns) {
                    return data.at(calculated_index);
                }
                else throw std::range_error("Index not accepted for this Matrix.");

            }


template <class T>
void Matrix::Representation<T>::put(u_int64_t r, u_int64_t c, T val) {

                uint64_t calculated_index = c + r * columns; 

                if (r <= rows && c <= columns) {
                    data.at(calculated_index) = val;
                }
                else throw std::range_error("Index not accepted for this Matrix.");

            }




template <class T> 
std::unique_ptr<Matrix::Representation<T>> MatrixMultiplication::Naive<T>::operator()(
        Matrix::Representation<T> l, 
        Matrix::Representation<T> r) {

    if (l.num_cols() != r.num_rows()) {
        throw std::length_error("Matrix A columns not equal to Matrix B rows.");
    }

    std::unique_ptr<Matrix::Representation<T>> output = std::make_unique<Matrix::Representation<T>>(l.num_rows(), r.num_cols());


    for (u_int64_t i = 0; i < l.num_rows(); i++) {
        
        for (u_int64_t j = 0; j < r.num_cols(); j++) {


            T val = 0;

            for (u_int64_t k = 0; k < l.num_cols(); k++) {
                val += l.get(i, k) * r.get(k, j);
            }

            output->put(i, j, val);

        }

    }



    return output;
}



