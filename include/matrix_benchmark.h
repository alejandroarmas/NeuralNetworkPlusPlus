#ifndef MATRIX_BENCHMARKER_H
#define MATRIX_BENCHMARKER_H

#include <memory>

#include "matrix.h"
#include "m_algorithms.h"

namespace Matrix {


    template <class MatrixOperation> 
    class Benchmark {

            public:
                Benchmark(std::unique_ptr<Operations::BaseOp<MatrixOperation>> _m) : matrix_operation(std::move(_m)) {}
            protected:
                virtual std::unique_ptr<Representation> operator()(
                    std::unique_ptr<Representation>& l, 
                    std::unique_ptr<Representation>& r) = 0;
                virtual ~Benchmark() = default;
                std::unique_ptr<Operations::BaseOp<MatrixOperation>> matrix_operation;

   };



    /*
    DESCRIPTION:

        Decorator for BaseOp() class Function objects, used to benchmark algorithm performance.

    USAGE:
      
        using matrix_t = Matrix::Representation; 
        std::unique_ptr<matrix_t> ma = std::make_unique<matrix_t>(5000, 5000);
        std::unique_ptr<matrix_t> mb = std::make_unique<matrix_t>(5000, 5000);
        Matrix::Generation::Normal<0, 1> normal_distribution_init;

        ma = normal_distribution_init(std::move(ma));
        mb = normal_distribution_init(std::move(mb));

        std::unique_ptr<Matrix::Operations::Multiplication::ParallelDNC> mul_ptr_r       = std::make_unique<Matrix::Operations::Multiplication::ParallelDNC>();
        Matrix::Timer<Matrix::Operations::Multiplication::ParallelDNC> mul_bm_r(std::move(mul_ptr_r));
        std::unique_ptr<matrix_t> mf = mul_bm_r(ma, mb);
    
    */
    template <class MatrixOperation>
    class Timer : public Benchmark<MatrixOperation> {

        public:
            Timer(std::unique_ptr<Operations::BaseOp<MatrixOperation>> _m) : Benchmark<MatrixOperation>(std::move(_m)) {}
            std::unique_ptr<Representation> operator()(
                    std::unique_ptr<Representation>& l, 
                    std::unique_ptr<Representation>& r) override; 

    };


// #ifdef CILKSCALE
    template <class MatrixOperation>
    class ParallelMeasurer : public Benchmark<MatrixOperation> {

        public:
            ParallelMeasurer(std::unique_ptr<Operations::BaseOp<MatrixOperation>> _m) : Benchmark<MatrixOperation>(std::move(_m)) {}
            std::unique_ptr<Representation> operator()(
                    std::unique_ptr<Representation>& l, 
                    std::unique_ptr<Representation>& r) override; 

    };
// #endif


}

#include "t_matrix_benchmark.cpp"

#endif //MATRIX_BENCHMARKER_H