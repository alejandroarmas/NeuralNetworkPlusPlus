#include <iostream>

#include "../include/m_algorithms.h"
#include "../include/matrix.h"
#include "../include/generator.h"
#include "../include/matrix_benchmark.h"

int main(void) {

    std::cout << "[5000, 1] x [5000, 1] Hadamard Product Benchmark:" << std::endl << std::endl ;
    std::cout << "..." << std::endl;

    using matrix_t = Matrix::Representation; 

    matrix_t ma = matrix_t(Matrix::Rows(5000), Matrix::Columns(1));
    matrix_t mb = matrix_t(Matrix::Rows(5000), Matrix::Columns(1));
    Matrix::Generation::Normal<0, 1> normal_distribution_init;

    ma = normal_distribution_init(ma);
    mb = normal_distribution_init(mb);

    Matrix::Operations::Timer timer(
        Matrix::Operations::Binary::HadamardProduct::Std{}
    );

    matrix_t mf = timer(ma, mb);
    
    std::cout << std::endl << "Performed in " << timer.get_computation_duration_ms() << " ms." << std::endl;


    return 0;
}