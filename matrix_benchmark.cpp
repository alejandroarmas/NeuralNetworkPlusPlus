#include <iostream>
#include <time.h>

#include "matrix_benchmark.h"


namespace Matrix {


    std::unique_ptr<Representation> Benchmark::operator()(std::unique_ptr<Matrix::Representation>& l, 
            std::unique_ptr<Matrix::Representation>& r) {


                struct timespec start, end;

                clock_gettime(CLOCK_MONOTONIC, &start);
                std::unique_ptr<Matrix::Representation> mc = matrix_operation->operator()(l, r);
                clock_gettime(CLOCK_MONOTONIC, &end);

                double tdiff = (end.tv_sec - start.tv_sec) + 1e-9*(end.tv_nsec - start.tv_nsec);

                std::cout << "Performed in: " << tdiff << " Seconds." << std::endl;

                return mc;
            }



}