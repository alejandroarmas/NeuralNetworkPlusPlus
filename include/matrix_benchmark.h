#ifndef MATRIX_BENCHMARKER_H
#define MATRIX_BENCHMARKER_H

#include <memory>
#include <iostream>

#include "matrix.h"
#include "m_algorithms.h"

namespace Matrix {


    namespace Operations {


            /*
            DESCRIPTION:

                Decorator for BaseInterface() class Function objects, used to benchmark algorithm performance.

            USAGE:
                    
                using matrix_t = Matrix::Representation; 
                std::unique_ptr<matrix_t> ma = std::make_unique<matrix_t>(5000, 5000);
                std::unique_ptr<matrix_t> mb = std::make_unique<matrix_t>(5000, 5000);
                Matrix::Generation::Normal<0, 1> normal_distribution_init;

                ma = normal_distribution_init(std::move(ma));
                mb = normal_distribution_init(std::move(mb));

                Matrix::Operations::Timer mul_bm_r(
                        std::make_unique<Matrix::Operations::Binary::Multiplication::ParallelDNC>());
                std::unique_ptr<matrix_t> mf = mul_bm_r(std::move(ma), std::move(mb));
                

                std::cout << "Performed in " << mul_bm_r.get_computation_duration_ms() << " ms." << std::endl;
            */
            template <Matrix::Operations::MatrixOperatable Operator>
            class Timer {

                public:
                    Timer(Operator _m) : 
                        matrix_operation(_m) {} 
                    
                    std::unique_ptr<Representation> operator()(
                            const std::unique_ptr<Representation>& l,
                            const std::unique_ptr<Representation>& r = nullptr);

                    int get_computation_duration_ms() { 
                        return std::chrono::duration_cast<std::chrono::duration<int, std::micro>>(end - start).count(); }
                    std::chrono::steady_clock::time_point get_start() { return start; }
                    std::chrono::steady_clock::time_point get_end()   { return end;   }
                    
                private:
                    Operator matrix_operation;
                    std::chrono::steady_clock::time_point start;
                    std::chrono::steady_clock::time_point end;
            };

    

        }

        
    }



#endif //MATRIX_BENCHMARKER_H