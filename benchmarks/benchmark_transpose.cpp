#include <iostream>

#include "../include/m_algorithms.h"
#include "../include/matrix.h"
#include "../include/generator.h"
#include "../include/matrix_benchmark.h"
#include "../include/matrix_printer.h"

int main(void) {

    std::cout << "[10000, 9000] Matrix Transpose Benchmark:" << std::endl << std::endl ;
    std::cout << "..." << std::endl;

    using matrix_t = Matrix::Representation; 

    matrix_t ma = matrix_t(Matrix::Rows(10000), Matrix::Columns(9000));
    Matrix::Generation::Normal<0, 1> normal_distribution_init;

    ma = normal_distribution_init(ma);

    Matrix::Operations::Timer timer(
        Matrix::Operations::Unary::Transpose{}
    );

    matrix_t mb = timer(ma);
    

    std::cout << std::endl << "Performed in " << timer.get_computation_duration_ms() << " ms." << std::endl;


    return 0;
}