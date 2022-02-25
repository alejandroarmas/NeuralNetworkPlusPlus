#include <memory>
#include <time.h>
#include <iostream>

#include "matrix.h"
#include "generator.h"
#include "matrix_printer.h"
#include "network_layer.h"
#include "m_algorithms.h"
#include "matrix_benchmark.h"


int main(void) {

    using matrix_t = Matrix::Representation; 

    std::unique_ptr<matrix_t> ma = std::make_unique<matrix_t>(4000, 2000);
    std::unique_ptr<matrix_t> mb = std::make_unique<matrix_t>(2000, 3000);


    Matrix::Generation::Normal<0, 1> normal_distribution_init;


    ma = normal_distribution_init(std::move(ma));
    mb = normal_distribution_init(std::move(mb));


    // Matrix::Printer m_printer;


    // Matrix::Operations::Multiplication::Square mul;

    std::unique_ptr<Matrix::Operations::Multiplication::Square> mul_ptr = std::make_unique<Matrix::Operations::Multiplication::Square>();
    std::unique_ptr<Matrix::Operations::Multiplication::Naive> nmul_ptr = std::make_unique<Matrix::Operations::Multiplication::Naive>();



    // std::unique_ptr<matrix_t> mc = mul_ptr->operator()(ma, mb);
    
    Matrix::Benchmark mul_bm(std::move(mul_ptr));

    std::unique_ptr<matrix_t> mc = mul_bm(ma, mb);
    Matrix::Benchmark nmul_bm(std::move(nmul_ptr));
    std::unique_ptr<matrix_t> md = nmul_bm(ma, mb);

    
    // m_printer(*mc);


    return 0;
}