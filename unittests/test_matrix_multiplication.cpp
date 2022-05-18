#include "../deps/doctest.h"

#include "../include/matrix.h"
#include "../include/generator.h"
#include "../include/m_algorithms.h"


TEST_CASE("Matrix Multiplication")
{
    Matrix::Representation mb = Matrix::Representation(
        Matrix::Rows(1000), Matrix::Columns(300));
    Matrix::Representation ma = Matrix::Representation(
        Matrix::Rows(200), Matrix::Columns(1000));
    

    Matrix::Generation::Normal<0, 1> normal_distribution_init;
    
    ma = normal_distribution_init(ma);
    mb = normal_distribution_init(mb);


    Matrix::Operations::Binary::Multiplication::Naive naive_mul;
    Matrix::Operations::Binary::Multiplication::Square c_mul;
    Matrix::Operations::Binary::Multiplication::ParallelDNC r_mul;

    Matrix::Representation mc = naive_mul(ma, mb);
    Matrix::Representation md = c_mul(ma, mb);
    Matrix::Representation me = r_mul(ma, mb);



    SUBCASE("Cilk-for Multiplication")
    {

        CHECK((Matrix::Representation{mc} == Matrix::Representation{md}) == true);
    }



    SUBCASE("Recursive Parallel Multiplication")
    {
        CHECK((Matrix::Representation{mc} == Matrix::Representation{me}) == true);
    }

}