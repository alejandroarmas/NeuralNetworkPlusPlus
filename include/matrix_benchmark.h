#ifndef MATRIX_BENCHMARKER_H
#define MATRIX_BENCHMARKER_H

#include <memory>

#include "matrix.h"
#include "m_algorithms.h"

namespace Matrix {


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