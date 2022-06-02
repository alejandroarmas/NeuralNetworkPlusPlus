#include "m_algorithms.h"
#include "m_algorithms_utilities.h"
#include "matrix_printer.h"

#include <cilk/cilk.h>
#include <iostream>
#include <math.h>
#include <numeric>
#include <assert.h>


namespace Matrix {

    namespace Operations {


        namespace Unary {

   
            Matrix::Representation ReLU::operate(
                        const Matrix::Representation& m) const noexcept{

                Matrix::Representation output = Matrix::Representation{
                            Matrix::Rows(m.num_rows()), 
                            Matrix::Columns(m.num_cols())
                };
                

                std::replace_copy_if(m.constScanStart(), m.constScanEnd(), output.scanStart(), 
                    [](float z){ return z < 0;}, 0);

                return Matrix::Representation{output};
            }

            Matrix::Representation Sign::operate(
                        const Matrix::Representation& m) const noexcept{

                Matrix::Representation output = Matrix::Representation(
                            Matrix::Rows(m.num_rows()), 
                            Matrix::Columns(m.num_cols())
                    );
                
                std::transform(m.constScanStart(), m.constScanEnd(), output.scanStart(), [](const auto val) { return val >= 0 ? 1 : 0;}); 

                return Matrix::Representation{output};
            }


            /*

            DESCRIPTION:
                Is a form of normalising arbitrary values using exponential distribution.

                exp(x_i) / SUM(exp(x)) is numerically unstable if dividing large terms,
                therefore we divide all intermediate terms by constant C = Max(exp(x)). 

                https://cs231n.github.io/linear-classify/#softmax

            */
            Matrix::Representation SoftMax::operate(
                        const Matrix::Representation& m) const noexcept{

                
                Matrix::Representation output = Matrix::Representation(
                            Matrix::Rows(m.num_rows()), 
                            Matrix::Columns(m.num_cols())
                    );
                

                auto max = std::max(m.constScanStart(), m.constScanEnd());

                std::transform(m.constScanStart(), m.constScanEnd(), output.scanStart(), [max](auto val) { return exp(val - *max); });

                double sum = std::accumulate(output.constScanStart(), output.constScanEnd(), 0.0);
        
                std::transform(output.constScanStart(), output.constScanEnd(), output.scanStart(), 
                    [sum](auto val) { return val / sum; }
                ); 

                return Matrix::Representation{output};
            }

            Matrix::Representation Transpose::operate(
                        const Matrix::Representation& m) const noexcept {

                Matrix::Representation output = Matrix::Representation{
                            Matrix::Rows(m.num_cols()), 
                            Matrix::Columns(m.num_rows())
                };

                transpose_helper(
                    m.constScanStart(), 
                    output.scanStart(), 
                    0, m.num_rows(), 
                    0, m.num_cols(), 
                    m.num_rows(), m.num_cols());

                return Matrix::Representation{output};
            }

            void transpose_helper(
                std::vector<float>::const_iterator in, 
                std::vector<float>::iterator out, 
                int rb, int re, int cb, int ce, int rows, int cols) noexcept {
                
                int r = re - rb, c = ce - cb;
                if (r <= 16 && c <= 16) {
                    for (int i = rb; i < re; i++) {
                        for (int j = cb; j < ce; j++) {
                            *(out + (j * rows + i)) = *(in + (i * cols + j));
                        }
                    }
                } else if (r >= c) {
                    cilk_spawn transpose_helper(in, out, rb, rb + (r / 2), cb, ce, rows, cols);
                    transpose_helper(in, out, rb + (r / 2), re, cb, ce, rows, cols);
                    cilk_sync;
                } else {
                    cilk_spawn transpose_helper(in, out, rb, re, cb, cb + (c / 2), rows, cols);
                    transpose_helper(in, out, rb, re, cb + (c / 2), ce, rows, cols);
                    cilk_sync;
                }
            }


        } // Unary


        /*

        DESCRIPTION:
            The cross-entropy between a “true” distribution p and an
             estimated distribution q is defined as:

             H(p, q) = -SUM(p(x), log(q(x)) ) 

            Logit function takes a probability and produces a real number 
            between negative and positive infinity.

            taking the log of the odds ratio brings about a certain 
            symmetricity in the results, making it easier to 
            interpret and use in various statistics
        */
        namespace Metric {


            Matrix::Representation CrossEntropy::operate(
                        const Matrix::Representation& p, 
                        const Matrix::Representation& q) const noexcept {
                
                Matrix::Representation output = Matrix::Representation(
                            Matrix::Rows(1), 
                            Matrix::Columns(1)
                    );
                Matrix::Operations::Unary::SoftMax softmax;

                Matrix::Representation theta = softmax(q); 
                
                double entropy = 0;

                for (auto p_i = p.constScanStart(), q_i = theta.constScanStart(); q_i != theta.constScanEnd(); p_i++, q_i++) {
                    entropy -= *p_i * log(*q_i);
                }                

                output.put(0, 0, entropy);

                return Matrix::Representation{output};
            }

        }


        namespace Binary {



            namespace Addition {

                Matrix::Representation Std::operate(
                        const Matrix::Representation& l, 
                        const Matrix::Representation& r) const noexcept {


#if DEBUG
                    if ((l.num_rows() != r.num_rows()) && (l.num_cols() != r.num_cols()))
                        std::cout << Utility::debug_message_2(l, r) << endl;
#endif
                    assert((l.num_rows() == r.num_rows()) && (l.num_cols() == r.num_cols()));


                        
                    auto output = Matrix::Representation(Rows(l.num_rows()), Columns(r.num_cols()));

                    std::transform(l.constScanStart(), l.constScanEnd(), r.constScanStart(), output.scanStart(), std::plus<float>());

                    return Matrix::Representation{output};
                }
            }

            namespace Subtraction {

                Matrix::Representation Std::operate(
                        const Matrix::Representation& l, 
                        const Matrix::Representation& r) const noexcept {

#if DEBUG
                    if ((l.num_rows() != r.num_rows()) && (l.num_cols() != r.num_cols()))
                        std::cout << Utility::debug_message_2(l, r) << endl;
#endif
                    assert((l.num_rows() == r.num_rows()) && (l.num_cols() == r.num_cols()));

                        
                    auto output = Matrix::Representation(Rows(l.num_rows()), Columns(r.num_cols()));

                    std::transform(l.constScanStart(), l.constScanEnd(), r.constScanStart(), output.scanStart(), std::minus<float>());

                    return Matrix::Representation{output};
                }
            }


            namespace OuterProduct {


                Matrix::Representation Naive::operate(
                        const Matrix::Representation& l, 
                        const Matrix::Representation& r) const noexcept {



#if DEBUG
                    if ( 
                        l.get_type() =! Matrix::Representation::Type::COLUMN_VECTOR && 
                        l.get_type() =! Matrix::Representation::Type::ROW_VECTOR ||
                        r.get_type() =! Matrix::Representation::Type::COLUMN_VECTOR && 
                        r.get_type() =! Matrix::Representation::Type::ROW_VECTOR
                    )
                        std::cout << Utility::debug_message_2(l, r) << endl;
#endif
                    assert(
                        l.get_type() == Matrix::Representation::Type::COLUMN_VECTOR || 
                        l.get_type() == Matrix::Representation::Type::ROW_VECTOR &&
                        r.get_type() == Matrix::Representation::Type::COLUMN_VECTOR || 
                        r.get_type() == Matrix::Representation::Type::ROW_VECTOR &&
                        "Operands are not Vectors.");
                    
                    u_int64_t x_dimension = l.num_rows() > r.num_rows() ? l.num_rows() : r.num_rows(); 
                    u_int64_t y_dimension = r.num_cols() > l.num_cols() ? r.num_cols() : l.num_cols();

                    auto output = Matrix::Representation(Rows(x_dimension), Columns(y_dimension));

                    auto li = l.constScanStart();

                    for (int i = 0; li != l.constScanEnd(); li++, i++) {
                        auto ri = r.constScanStart();
                        
                        for (int j = 0; ri != r.constScanEnd(); ri++, j++) {
                            float val = *li * *ri;
                            output.put(i, j, val);
                        }
                    }
                    
                    return Matrix::Representation{output};
                }

                

            }


            namespace HadamardProduct {

                Matrix::Representation Std::operate(
                        const Matrix::Representation& l, 
                        const Matrix::Representation& r) const noexcept {

                        auto output = Matrix::Representation(
                                Rows(l.num_rows()), 
                                Columns(r.num_cols()));

                        
                        std::transform(l.constScanStart(), l.constScanEnd(), r.constScanStart(), output.scanStart(), std::multiplies<float>()); 
                        
                    return Matrix::Representation{output};
                }


                Matrix::Representation Naive::operate(
                        const Matrix::Representation& l, 
                        const Matrix::Representation& r) const noexcept {



#if DEBUG
                    if (l.get_type() != r.get_type() || 
                        l.get_type() =! Matrix::Representation::Type::COLUMN_VECTOR && 
                        l.get_type() =! Matrix::Representation::Type::ROW_VECTOR)
                        std::cout << Utility::debug_message_2(l, r) << endl;
#endif
                    assert(l.get_type() == r.get_type() && 
                        l.get_type() == Matrix::Representation::Type::COLUMN_VECTOR || 
                        l.get_type() == Matrix::Representation::Type::ROW_VECTOR &&
                        "Operands are not Vectors.");

                    Matrix::Representation output = Matrix::Representation(Rows(l.num_rows()), Columns(r.num_cols()));


                    for (u_int64_t i = 0; i < l.num_rows(); i++) {
                        
                        for (u_int64_t j = 0; j < r.num_cols(); j++) {


                            float val = l.get(i, j) * r.get(i, j);

                            output.put(i, j, val);

                        }

                    }


                    return Matrix::Representation{output};
                }
            } 


            namespace Multiplication {

                Matrix::Representation Naive::operate(
                        const Matrix::Representation& l, 
                        const Matrix::Representation& r) const noexcept {


#if DEBUG
                    if (l.num_cols() != r.num_rows())
                        std::cout << Utility::debug_message(l, r) << endl;
#endif
                    assert(l.num_cols() == r.num_rows());

                    // if (l.num_cols() != r.num_rows()) {
                    //     throw std::length_error(Utility::debug_message(l, r));

                    // }

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



                    return Matrix::Representation{output};
                }

                
                    /*
                    Adapted from https://ocw.mit.edu/courses/mathematics/18-335j-introduction-to-numerical-methods-spring-2019/week-5/MIT18_335JS19_lec12.pdf
                    
                    We need to divide the data until it fits into lowest cache.
                    */
                    void add_matmul_rec(std::vector<float>::const_iterator a, std::vector<float>::const_iterator b, std::vector<float>::iterator c, 
                        int m, int n, int p, int fdA, int fdB, int fdC) noexcept {
                        
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
                        const Matrix::Representation& r) const noexcept {

                    
#if DEBUG
                    if (l.num_cols() != r.num_rows())
                        std::cout << Utility::debug_message(l, r) << endl;
#endif
                    assert(l.num_cols() == r.num_rows());


                    Matrix::Representation output = Matrix::Representation(Rows(l.num_rows()), Columns(r.num_cols()));

                    add_matmul_rec(l.constScanStart(), r.constScanStart(), output.scanStart(), l.num_rows(), l.num_cols(), r.num_cols(), l.num_cols(), r.num_cols(), r.num_cols());

                    return Matrix::Representation{output};
                }
        
        
                Matrix::Representation Square::operate(
                        const Matrix::Representation& l, 
                        const Matrix::Representation& r) const noexcept {

                    

#if DEBUG
                    if (l.num_cols() != r.num_rows())
                        std::cout << Utility::debug_message(l, r) << endl;
#endif
                    assert(l.num_cols() == r.num_rows());


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



                    return Matrix::Representation{output};
                }
        
            } // namespace Multiplication

        }  // namespace Binary

    } // namespace Operations

} // namespace Matrix

