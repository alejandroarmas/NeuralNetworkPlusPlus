#ifndef MATRIX_BENCHMARKER_H
#define MATRIX_BENCHMARKER_H

#include <memory>

#include "matrix.h"
#include "m_algorithms.h"

namespace Matrix {



   class Benchmark {

            protected:
                Benchmark(std::unique_ptr<Operations::BaseOp> _m) : matrix_operation(std::move(_m)) {}
                virtual std::unique_ptr<Representation> operator()(
                    std::unique_ptr<Representation>& l, 
                    std::unique_ptr<Representation>& r) = 0;
                virtual ~Benchmark() = default;
            protected:
                std::unique_ptr<Operations::BaseOp> matrix_operation;

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
        Matrix::Timer mul_bm_r(std::move(mul_ptr_r));
        std::unique_ptr<matrix_t> mf = mul_bm_r(ma, mb);

    */
    class Timer : public Benchmark {

        public:
            Timer(std::unique_ptr<Operations::BaseOp> _m) : Benchmark(std::move(_m)) {}
            std::unique_ptr<Representation> operator()(
                    std::unique_ptr<Representation>& l, 
                    std::unique_ptr<Representation>& r) override; 

    };


// #ifdef CILKSCALE
    class ParallelMeasurer : public Benchmark {

        public:
            ParallelMeasurer(std::unique_ptr<Operations::BaseOp> _m) : Benchmark(std::move(_m)) {}
            std::unique_ptr<Representation> operator()(
                    std::unique_ptr<Representation>& l, 
                    std::unique_ptr<Representation>& r) override; 

    };
// #endif


}

#endif //MATRIX_BENCHMARKER_H