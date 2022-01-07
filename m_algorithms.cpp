#include "m_algorithms.h"


std::unique_ptr<Matrix::Representation> Matrix::Operations::Add::Std::operator()(
        Matrix::Representation l, 
        Matrix::Representation r) {

        std::transform(l.scanStart(), l.scanEnd(), r.scanStart(), l.scanStart(), std::plus<float>());
        
        auto output = std::make_unique<Matrix::Representation>(l);

    return output;
}



std::unique_ptr<Matrix::Representation> Matrix::Operations::HadamardProduct::Std::operator()(
        Matrix::Representation l, 
        Matrix::Representation r) {

        std::transform(l.scanStart(), l.scanEnd(), r.scanStart(), l.scanStart(), std::multiplies<float>());
        
        auto output = std::make_unique<Matrix::Representation>(l);

    return output;
}




std::unique_ptr<Matrix::Representation> Matrix::Operations::HadamardProduct::Naive::operator()(
        Matrix::Representation l, 
        Matrix::Representation r) {

    if ((l.num_rows() != r.num_rows()) && (l.num_cols() != r.num_cols())) {
        throw std::length_error("Matrix A columns not equal to Matrix B rows.");
    }

    std::unique_ptr<Matrix::Representation> output = std::make_unique<Matrix::Representation>(l.num_rows(), r.num_cols());


    for (u_int64_t i = 0; i < l.num_rows(); i++) {
        
        for (u_int64_t j = 0; j < r.num_cols(); j++) {


            float val = l.get(i, j) * r.get(i, j);

            output->put(i, j, val);

        }

    }


    return output;
}




std::unique_ptr<Matrix::Representation> Matrix::Operations::Multiplication::Naive::operator()(
        Matrix::Representation l, 
        Matrix::Representation r) {

    if (l.num_cols() != r.num_rows()) {
        throw std::length_error("Matrix A columns not equal to Matrix B rows.");
    }

    std::unique_ptr<Matrix::Representation> output = std::make_unique<Matrix::Representation>(l.num_rows(), r.num_cols());


    for (u_int64_t i = 0; i < l.num_rows(); i++) {
        
        for (u_int64_t j = 0; j < r.num_cols(); j++) {


            float val = 0;

            for (u_int64_t k = 0; k < l.num_cols(); k++) {
                val += l.get(i, k) * r.get(k, j);
            }

            output->put(i, j, val);

        }

    }



    return output;
}
