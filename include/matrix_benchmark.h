#ifndef MATRIX_BENCHMARKER_H
#define MATRIX_BENCHMARKER_H

#include <memory>

#include "matrix.h"
#include "m_algorithms.h"

namespace Matrix {


    /* EXAMPLE USAGE:    
    
    using matrix_t = Matrix::Representation; 

    std::unique_ptr<matrix_t> ma = std::make_unique<matrix_t>(4000, 2000);
    std::unique_ptr<matrix_t> mb = std::make_unique<matrix_t>(2000, 3000);


    Matrix::Generation::Normal<0, 1> normal_distribution_init;


    ma = normal_distribution_init(std::move(ma));
    mb = normal_distribution_init(std::move(mb));

    std::unique_ptr<Matrix::Operations::Multiplication::Square> mul_ptr = std::make_unique<Matrix::Operations::Multiplication::Square>();
    Matrix::Benchmark mul_bm(std::move(mul_ptr));
    std::unique_ptr<matrix_t> mc = mul_bm(ma, mb);
    
    */
    class Benchmark {

        public:
            Benchmark(std::unique_ptr<Operations::BaseOp> _m) : matrix_operation(std::move(_m)) {}
            
            std::unique_ptr<Representation> operator()(
                    std::unique_ptr<Matrix::Representation>& l, 
                    std::unique_ptr<Matrix::Representation>& r); 
        private:
            std::unique_ptr<Operations::BaseOp> matrix_operation;

    };


}

#endif //MATRIX_BENCHMARKER_H