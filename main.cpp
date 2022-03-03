#include <memory>
#include <iostream>

#include "matrix.h"
#include "generator.h"
#include "matrix_printer.h"
// #include "network_layer.h"
#include "m_algorithms.h"
#include "matrix_benchmark.h"


int main(void) {

    using matrix_t = Matrix::Representation; 

    std::unique_ptr<matrix_t> ma = std::make_unique<matrix_t>(4000, 4000);
    std::unique_ptr<matrix_t> mb = std::make_unique<matrix_t>(4000, 4000);


    Matrix::Generation::Normal<0, 1> normal_distribution_init;


    ma = normal_distribution_init(std::move(ma));
    mb = normal_distribution_init(std::move(mb));


    std::unique_ptr<Matrix::Operations::Multiplication::RecursiveParallel> mul_ptr_r = std::make_unique<Matrix::Operations::Multiplication::RecursiveParallel>();
    std::unique_ptr<Matrix::Operations::Multiplication::Square> mul_ptr_s            = std::make_unique<Matrix::Operations::Multiplication::Square>();
    std::unique_ptr<Matrix::Operations::Multiplication::Naive> mul_ptr_n             = std::make_unique<Matrix::Operations::Multiplication::Naive>();
    

    Matrix::Benchmark mul_bm_r(std::move(mul_ptr_r));
    Matrix::Benchmark mul_bm_s(std::move(mul_ptr_s));
    Matrix::Benchmark mul_bm_n(std::move(mul_ptr_n));
    std::unique_ptr<matrix_t> mc = mul_bm_n(ma, mb);
    std::unique_ptr<matrix_t> me = mul_bm_s(ma, mb);
    std::unique_ptr<matrix_t> mf = mul_bm_r(ma, mb);

    // Matrix::Printer p;

    // p(*ma);
    // p(*mb);
    // p(*mc);



    return 0;
}