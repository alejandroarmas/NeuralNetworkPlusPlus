#include <iostream>

#include "../include/m_algorithms.h"
#include "../include/matrix.h"
#include "../include/generator.h"
#include "../include/matrix_benchmark.h"

int main(void) {

    std::cout << "[5000, 5000] x [5000, 5000] Parallel Matrix Multiply Benchmark:" << std::endl << std::endl ;
    std::cout << "..." << std::endl;

    using matrix_t = Matrix::Representation; 

    matrix_t ma = matrix_t(Matrix::Rows(5000), Matrix::Columns(5000));
    matrix_t mb = matrix_t(Matrix::Rows(5000), Matrix::Columns(5000));
    Matrix::Generation::Normal<0, 1> normal_distribution_init;

    ma = normal_distribution_init(ma);
    mb = normal_distribution_init(mb);

    Matrix::Operations::Timer mul_bm_r(
        Matrix::Operations::Binary::Multiplication::ParallelDNC{}
    );

    matrix_t mf = mul_bm_r(ma, mb);
    
    std::cout << std::endl << "Performed in " << mul_bm_r.get_computation_duration_ms() << " ms." << std::endl;


    return 0;
}