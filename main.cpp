#include <memory>
#include <time.h>
#include <iostream>

#include "matrix.h"
#include "generator.h"
#include "matrix_printer.h"
#include "network_layer.h"
#include "m_algorithms.h"



int main(void) {

    using matrix_t = Matrix::Representation; 

    struct timespec start, end;
    
    std::unique_ptr<matrix_t> ma = std::make_unique<matrix_t>(4000, 2000);
    std::unique_ptr<matrix_t> mb = std::make_unique<matrix_t>(2000, 3000);


    Matrix::Generation::Normal<0, 1> normal_distribution_init;


    ma = normal_distribution_init(std::move(ma));
    mb = normal_distribution_init(std::move(mb));


    Matrix::Printer m_printer;


    Matrix::Operations::Multiplication::Naive mul;
    clock_gettime(CLOCK_MONOTONIC, &start);

    std::unique_ptr<matrix_t> mc = mul(*ma, *mb);
    
    clock_gettime(CLOCK_MONOTONIC, &end);

    double tdiff = (end.tv_sec - start.tv_sec) + 1e-9*(end.tv_nsec - start.tv_nsec);

    m_printer(*mc);

    std::cout << "Matrix Multiply took: " << tdiff << " Seconds." << std::endl;
    

    return 0;
}