#ifndef MATRIX_BENCHMARKER_H
#define MATRIX_BENCHMARKER_H

#include <memory>
#include <iostream>
#include <chrono>
#include <time.h>

#include "matrix.h"
#include "m_algorithms.h"

namespace Matrix {


    namespace Operations {


            /*
            DESCRIPTION:

                Decorator for BaseInterface() class Function objects, used to benchmark algorithm performance.

            USAGE:
                    
                using matrix_t = Matrix::Representation; 

                matrix_t ma = matrix_t(Matrix::Rows(5000), Matrix::Columns(5000));
                matrix_t mb = matrix_t(Matrix::Rows(5000), Matrix::Columns(5000));
                Matrix::Generation::Normal<0, 1> normal_distribution_init;

                ma = normal_distribution_init(ma);
                mb = normal_distribution_init(mb);

                Matrix::Operations::Timer mul_bm_r(
                    Matrix::Operations::Binary::Multiplication::ParallelDNC{}
                );

                matrix_t mf = mul_bm_r(ma, mb);
                
                std::cout << std::endl << "Performed in " << mul_bm_r.get_computation_duration_ms() << " ms." << std::endl;
            */

            template <Matrix::Operations::MatrixOperatable T>
            class Timer {

                public:
                    explicit Timer(T _m) : 
                        matrix_operation(_m) {} 
                    
                    int get_computation_duration_ms() { 
                        return std::chrono::duration_cast<std::chrono::duration<int, std::micro>>(end - start).count(); }
                    std::chrono::steady_clock::time_point get_start() { return start; }
                    std::chrono::steady_clock::time_point get_end()   { return end;   }
                

                    template <Matrix::Operations::BinaryMatrixOperatable U = T>
                    Representation operator()(
                            const Matrix::Representation& l, 
                            const Matrix::Representation& r) noexcept {

                        start = std::chrono::steady_clock::now();                
                        Matrix::Representation mc = matrix_operation(l, r);
                        end   = std::chrono::steady_clock::now();

                        return Matrix::Representation{mc};
                    }

                    template <Matrix::Operations::UnaryMatrixOperatable U = T>
                    Representation operator()(
                            const Matrix::Representation& l) noexcept {

                        start = std::chrono::steady_clock::now();                
                        Matrix::Representation mc = matrix_operation(l);
                        end   = std::chrono::steady_clock::now();
                        
                        return Matrix::Representation{mc};
                    }

                private:

                    T matrix_operation;
                    std::chrono::steady_clock::time_point start;
                    std::chrono::steady_clock::time_point end;
            };


    

        }

        
    }



#endif //MATRIX_BENCHMARKER_H