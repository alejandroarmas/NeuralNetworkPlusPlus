#include "m_algorithms.h"

#include <cilk/cilk.h>
#include <cilk/reducer.h>


namespace Matrix {

    namespace Operations {


        namespace Addition {

            std::unique_ptr<Matrix::Representation> Std::operator()(
                    Matrix::Representation l, 
                    Matrix::Representation r) {

                if ((l.num_rows() != r.num_rows()) && (l.num_cols() != r.num_cols())) {
                    throw std::length_error("Matrix A not same size as Matrix B.");
                }
                    
                auto output = std::make_unique<Matrix::Representation>(l.num_rows(), r.num_cols());

                std::transform(l.scanStart(), l.scanEnd(), r.scanStart(), output->scanStart(), std::plus<float>());

                return output;
            }
        }


        namespace HadamardProduct {

            std::unique_ptr<Matrix::Representation> Std::operator()(
                    Matrix::Representation l, 
                    Matrix::Representation r) {

                    std::transform(l.scanStart(), l.scanEnd(), r.scanStart(), l.scanStart(), std::multiplies<float>());
                    
                    auto output = std::make_unique<Matrix::Representation>(l);

                return output;
            }


            std::unique_ptr<Matrix::Representation> Naive::operator()(
                    Matrix::Representation l, 
                    Matrix::Representation r) {

                if ((l.num_rows() != r.num_rows()) && (l.num_cols() != r.num_cols())) {
                    throw std::length_error("Matrix A not same size as Matrix B.");
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
        } 


        namespace Multiplication {

            std::unique_ptr<Matrix::Representation> Naive::operator()(
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


            std::unique_ptr<Matrix::Representation> Square::operator()(
                    Matrix::Representation l, 
                    Matrix::Representation r) {

                if (l.num_cols() != r.num_rows()) {
                    throw std::length_error("Matrix A columns not equal to Matrix B rows.");
                }

                std::unique_ptr<Matrix::Representation> output = std::make_unique<Matrix::Representation>(l.num_rows(), r.num_cols());


                cilk_for (u_int64_t i = 0; i < l.num_rows(); i++) {
                    
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
        }

    }

}











