#include "m_algorithms.h"

#include <cilk/cilk.h>
#include <cilk/reducer.h>
#include <iostream>


namespace Matrix {

    namespace Operations {


        namespace Addition {

            std::unique_ptr<Matrix::Representation> Std::operator()(
                    std::unique_ptr<Matrix::Representation>& l, 
                    std::unique_ptr<Matrix::Representation>& r) {

                if ((l->num_rows() != r->num_rows()) && (l->num_cols() != r->num_cols())) {
                    throw std::length_error("Matrix A not same size as Matrix B.");
                }
                    
                auto output = std::make_unique<Matrix::Representation>(l->num_rows(), r->num_cols());

                std::transform(l->scanStart(), l->scanEnd(), r->scanStart(), output->scanStart(), std::plus<float>());

                return output;
            }
        }


        namespace HadamardProduct {

            std::unique_ptr<Matrix::Representation> Std::operator()(
                    std::unique_ptr<Matrix::Representation>& l, 
                    std::unique_ptr<Matrix::Representation>& r) {

                    auto output = std::make_unique<Matrix::Representation>(l->num_rows(), r->num_cols());

                    
                    std::transform(l->scanStart(), l->scanEnd(), r->scanStart(), output->scanStart(), std::multiplies<float>()); 
                    
                return output;
            }


            std::unique_ptr<Matrix::Representation> Naive::operator()(
                    std::unique_ptr<Matrix::Representation>& l, 
                    std::unique_ptr<Matrix::Representation>& r) {

                if ((l->num_rows() != r->num_rows()) && (l->num_cols() != r->num_cols())) {
                    throw std::length_error("Matrix A not same size as Matrix B.");
                }

                std::unique_ptr<Matrix::Representation> output = std::make_unique<Matrix::Representation>(l->num_rows(), r->num_cols());


                for (u_int64_t i = 0; i < l->num_rows(); i++) {
                    
                    for (u_int64_t j = 0; j < r->num_cols(); j++) {


                        float val = l->get(i, j) * r->get(i, j);

                        output->put(i, j, val);

                    }

                }


                return output;
            }
        } 


        namespace Multiplication {

            std::unique_ptr<Matrix::Representation> Naive::operator()(
                    std::unique_ptr<Matrix::Representation>& l, 
                    std::unique_ptr<Matrix::Representation>& r) {

                if (l->num_cols() != r->num_rows()) {
                    throw std::length_error("Matrix A columns not equal to Matrix B rows.");
                }

                std::unique_ptr<Matrix::Representation> output = std::make_unique<Matrix::Representation>(l->num_rows(), r->num_cols());


                for (u_int64_t i = 0; i < l->num_rows(); i++) {
                    
                    for (u_int64_t j = 0; j < r->num_cols(); j++) {


                        float val = 0;

                        for (u_int64_t k = 0; k < l->num_cols(); k++) {
                            val += l->get(i, k) * r->get(k, j);
                        }

                        output->put(i, j, val);

                    }

                }



                return output;
            }

            
                /*
                Adapted from https://ocw.mit.edu/courses/mathematics/18-335j-introduction-to-numerical-methods-spring-2019/week-5/MIT18_335JS19_lec12.pdf
                */
                void add_matmul_rec(std::vector<float>::iterator a, std::vector<float>::iterator b, std::vector<float>::iterator c, 
                    int m, int n, int p, int fdA, int fdB, int fdC) {
                    
                    if (m + n + p <= 48) {  
                        int i, j, k;
                        
                        for (i = 0; i < m; ++i) {
                            for (k = 0; k < p; ++k) { 
                                float sum = 0;
                                for (j = 0; j < n; ++j)
                                    sum += *(a + (i * fdA + j)) * *(b + (j * fdB + k));
                                *(c + (i * fdC + k)) += sum;
                    
                            }
                        }
                    }
                    else {  
                        int m2 = m/2, n2 = n/2, p2 = p/2;
                
                         cilk_spawn add_matmul_rec(a, b, c, m2, n2, p2, fdA, fdB, fdC); 
                         cilk_spawn add_matmul_rec(a, b + p2, c + p2, m2, n2, p - p2, fdA, fdB, fdC); 
                         cilk_spawn add_matmul_rec(a + m2*fdA + n2, b + n2*fdB, c + m2*fdC, m-m2, n - n2, p2, fdA, fdB, fdC);
                         add_matmul_rec(a + m2*fdA + n2, b + p2 + n2*fdB, c + m2*fdC + p2, m - m2, n - n2, p - p2, fdA, fdB, fdC);
                         cilk_sync;
              
                         cilk_spawn add_matmul_rec(a + n2, b + n2*fdB, c, m2, n - n2, p2, fdA, fdB, fdC);
                         cilk_spawn add_matmul_rec(a + m2*fdA, b, c + m2*fdC, m - m2, n2, p2, fdA, fdB, fdC); 
                         cilk_spawn add_matmul_rec(a + n2       , b + p2 + n2*fdB, c + p2, m2, n - n2, p - p2, fdA, fdB, fdC);
                         add_matmul_rec(a + m2*fdA, b + p2, c + m2*fdC + p2, m - m2, n2, p - p2, fdA, fdB, fdC);
                         cilk_sync;
                    }
                }


            std::unique_ptr<Matrix::Representation> ParallelDNC::operator()(
                    std::unique_ptr<Matrix::Representation>& l, 
                    std::unique_ptr<Matrix::Representation>& r) {

                if (l->num_cols() != r->num_rows()) {
                    throw std::length_error("Matrix A columns not equal to Matrix B rows.");
                }

                std::unique_ptr<Matrix::Representation> output = std::make_unique<Matrix::Representation>(l->num_rows(), r->num_cols());

                add_matmul_rec(l->scanStart(), r->scanStart(), output->scanStart(), l->num_rows(), l->num_cols(), r->num_cols(), l->num_cols(), r->num_cols(), r->num_cols());

                return output;
            }
    
    
            std::unique_ptr<Matrix::Representation> Square::operator()(
                    std::unique_ptr<Matrix::Representation>& l, 
                    std::unique_ptr<Matrix::Representation>& r) {

                if (l->num_cols() != r->num_rows()) {
                    throw std::length_error("Matrix A columns not equal to Matrix B rows.");
                }

                std::unique_ptr<Matrix::Representation> output = std::make_unique<Matrix::Representation>(l->num_rows(), r->num_cols());


                cilk_for (u_int64_t i = 0; i < l->num_rows(); i++) {
                    
                    for (u_int64_t j = 0; j < r->num_cols(); j++) {


                        float val = 0;

                        for (u_int64_t k = 0; k < l->num_cols(); k++) {
                            val += l->get(i, k) * r->get(k, j);
                        }

                        output->put(i, j, val);

                    }

                }



                return output;
            }
        }

    }

}











