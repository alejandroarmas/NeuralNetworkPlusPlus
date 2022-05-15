#include "m_algorithms.h"
#include "m_algorithms_utilities.h"

#include <cilk/cilk.h>
#include <iostream>
#include <math.h>


namespace Matrix {

    namespace Operations {


        namespace Unary {

   
            Matrix::Representation ReLU::operate(
                        const Matrix::Representation& m) const{

                Matrix::Representation output = Matrix::Representation(
                            Matrix::Rows(m.num_rows()), 
                            Matrix::Columns(m.num_cols())
                    );
                

                std::replace_copy_if(m.constScanStart(), m.constScanEnd(), output.scanStart(), 
                    [](float z){ return z < 0;}, 0);

                return output;
            }


            Matrix::Representation SoftMax::operate(
                        const Matrix::Representation& m) const{

                
                Matrix::Representation output = Matrix::Representation(
                            Matrix::Rows(m.num_rows()), 
                            Matrix::Columns(m.num_cols())
                    );

                auto max = std::max(m.constScanStart(), m.constScanEnd());

                std::transform(m.constScanStart(), m.constScanEnd(), output.scanStart(), [max](const auto val) { return val - *max; }); 
                std::transform(output.constScanStart(), output.constScanEnd(), output.scanStart(), [](const auto val) { return exp(val);}); 



                return output;
            }



        } // Unary

        namespace Metric {


            Matrix::Representation CrossEntropy::operate(
                        const Matrix::Representation& p, 
                        const Matrix::Representation& q) const {
                
                Matrix::Representation output = Matrix::Representation(
                            Matrix::Rows(1), 
                            Matrix::Columns(1)
                    );

                
                float entropy = 0;

                for (auto p_i = p.constScanStart(), q_i = q.constScanStart(); q_i != q.constScanEnd(); p_i++, q_i++) {
                    entropy += *p_i * log(*q_i);
                }


                output.put(0, 0, entropy);

                return output;
            }

        }


        namespace Binary {



            namespace Addition {

                Matrix::Representation Std::operate(
                        const Matrix::Representation& l, 
                        const Matrix::Representation& r) const {

                    if ((l.num_rows() != r.num_rows()) && (l.num_cols() != r.num_cols())) {
                        throw std::length_error(Utility::debug_message_2(l, r));
                    }
                        
                    auto output = Matrix::Representation(Rows(l.num_rows()), Columns(r.num_cols()));

                    std::transform(l.constScanStart(), l.constScanEnd(), r.constScanStart(), output.scanStart(), std::plus<float>());

                    return output;
                }
            }

            namespace Subtraction {

                Matrix::Representation Std::operate(
                        const Matrix::Representation& l, 
                        const Matrix::Representation& r) const {

                    if ((l.num_rows() != r.num_rows()) && (l.num_cols() != r.num_cols())) {
                        throw std::length_error(Utility::debug_message_2(l, r));
                    }
                        
                    auto output = Matrix::Representation(Rows(l.num_rows()), Columns(r.num_cols()));

                    std::transform(l.constScanStart(), l.constScanEnd(), r.constScanStart(), output.scanStart(), std::minus<float>());

                    return output;
                }
            }


            namespace OuterProduct {


                Matrix::Representation Naive::operate(
                        const Matrix::Representation& l, 
                        const Matrix::Representation& r) const {

                    if (l.num_rows() != r.num_rows() && l.num_cols() != r.num_cols()) {
                        throw std::length_error(Utility::debug_message_2(l, r));
                    }
                    if (l.num_rows() != 1 && l.num_cols() != 1) {
                        throw std::length_error("Operands are not Vectors.");
                    }
                    
                    u_int64_t dimension; 

                    if (l.num_rows() > l.num_cols()) {
                        dimension = l.num_rows(); 
                    }
                    else dimension = l.num_cols(); 

                    auto output = Matrix::Representation(Rows(dimension), Columns(dimension));

                    auto li = l.constScanStart();

                    for (int i = 0; li != l.constScanEnd(); li++, i++) {
                        auto ri = r.constScanStart();
                        
                        for (int j = 0; ri != r.constScanEnd(); ri++, j++) {
                            float val = *li * *ri;
                            output.put(i, j, val);
                        }
                    }
                    
                    return output;
                }

            }


            namespace HadamardProduct {

                Matrix::Representation Std::operate(
                        const Matrix::Representation& l, 
                        const Matrix::Representation& r) const {

                        auto output = Matrix::Representation(
                                Rows(l.num_rows()), 
                                Columns(r.num_cols()));

                        
                        std::transform(l.constScanStart(), l.constScanEnd(), r.constScanStart(), output.scanStart(), std::multiplies<float>()); 
                        
                    return output;
                }


                Matrix::Representation Naive::operate(
                        const Matrix::Representation& l, 
                        const Matrix::Representation& r) const {

                    if ((l.num_rows() != r.num_rows()) && (l.num_cols() != r.num_cols())) {
                        throw std::length_error("Matrix A not same size as Matrix B.");
                    }

                    Matrix::Representation output = Matrix::Representation(Rows(l.num_rows()), Columns(r.num_cols()));


                    for (u_int64_t i = 0; i < l.num_rows(); i++) {
                        
                        for (u_int64_t j = 0; j < r.num_cols(); j++) {


                            float val = l.get(i, j) * r.get(i, j);

                            output.put(i, j, val);

                        }

                    }


                    return output;
                }
            } 


            namespace Multiplication {

                Matrix::Representation Naive::operate(
                        const Matrix::Representation& l, 
                        const Matrix::Representation& r) const {

                    if (l.num_cols() != r.num_rows()) {
                        throw std::length_error(Utility::debug_message(l, r));

                    }

                    Matrix::Representation output = Matrix::Representation(Rows(l.num_rows()), Columns(r.num_cols()));


                    for (u_int64_t i = 0; i < l.num_rows(); i++) {
                        
                        for (u_int64_t j = 0; j < r.num_cols(); j++) {


                            float val = 0;

                            for (u_int64_t k = 0; k < l.num_cols(); k++) {
                                val += l.get(i, k) * r.get(k, j);
                            }

                            output.put(i, j, val);

                        }

                    }



                    return output;
                }

                
                    /*
                    Adapted from https://ocw.mit.edu/courses/mathematics/18-335j-introduction-to-numerical-methods-spring-2019/week-5/MIT18_335JS19_lec12.pdf
                    */
                    void add_matmul_rec(std::vector<float>::const_iterator a, std::vector<float>::const_iterator b, std::vector<float>::iterator c, 
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


                Matrix::Representation ParallelDNC::operate(
                        const Matrix::Representation& l, 
                        const Matrix::Representation& r) const {

                    if (l.num_cols() != r.num_rows()) {
                        
                        throw std::length_error(Utility::debug_message(l, r));
                    }

                    Matrix::Representation output = Matrix::Representation(Rows(l.num_rows()), Columns(r.num_cols()));

                    add_matmul_rec(l.constScanStart(), r.constScanStart(), output.scanStart(), l.num_rows(), l.num_cols(), r.num_cols(), l.num_cols(), r.num_cols(), r.num_cols());

                    return output;
                }
        
        
                Matrix::Representation Square::operate(
                        const Matrix::Representation& l, 
                        const Matrix::Representation& r) const {

                    if (l.num_cols() != r.num_rows()) {
                        throw std::length_error(Utility::debug_message(l, r));

                    }

                    Matrix::Representation output = Matrix::Representation(Rows(l.num_rows()), Columns(r.num_cols()));


                    cilk_for (u_int64_t i = 0; i < l.num_rows(); i++) {
                        
                        for (u_int64_t j = 0; j < r.num_cols(); j++) {


                            float val = 0;

                            for (u_int64_t k = 0; k < l.num_cols(); k++) {
                                val += l.get(i, k) * r.get(k, j);
                            }

                            output.put(i, j, val);

                        }

                    }



                    return output;
                }
        
            } // namespace Multiplication

        }  // namespace Binary

    } // namespace Operations

} // namespace Matrix

