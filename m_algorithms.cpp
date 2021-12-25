#include "m_algorithms.h"


template <class T> 
std::unique_ptr<Matrix::Representation<T>> Matrix::Operations::Add::Std<T>::operator()(
        Matrix::Representation<T> l, 
        Matrix::Representation<T> r) {

        std::transform(l.scanStart(), l.scanEnd(), r.scanStart(), l.scanStart(), std::plus<T>());
        
        auto output = std::make_unique<Matrix::Representation<T>>(l);

    return output;
}


template <class T> 
std::unique_ptr<Matrix::Representation<T>> Matrix::Operations::HadamardProduct::Std<T>::operator()(
        Matrix::Representation<T> l, 
        Matrix::Representation<T> r) {

        std::transform(l.scanStart(), l.scanEnd(), r.scanStart(), l.scanStart(), std::multiplies<T>());
        
        auto output = std::make_unique<Matrix::Representation<T>>(l);

    return output;
}



template <class T> 
std::unique_ptr<Matrix::Representation<T>> Matrix::Operations::HadamardProduct::Naive<T>::operator()(
        Matrix::Representation<T> l, 
        Matrix::Representation<T> r) {

    if ((l.num_rows() != r.num_rows()) && (l.num_cols() != r.num_cols())) {
        throw std::length_error("Matrix A columns not equal to Matrix B rows.");
    }

    std::unique_ptr<Matrix::Representation<T>> output = std::make_unique<Matrix::Representation<T>>(l.num_rows(), r.num_cols());


    for (u_int64_t i = 0; i < l.num_rows(); i++) {
        
        for (u_int64_t j = 0; j < r.num_cols(); j++) {


            T val = l.get(i, j) * r.get(i, j);

            output->put(i, j, val);

        }

    }


    return output;
}



template <class T> 
std::unique_ptr<Matrix::Representation<T>> Matrix::Operations::Multiplication::Naive<T>::operator()(
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
