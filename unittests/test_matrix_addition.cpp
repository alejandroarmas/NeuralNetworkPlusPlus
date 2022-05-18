#include "../deps/doctest.h"

#include "../include/matrix.h"
#include "../include/generator.h"
#include "../include/m_algorithms.h"


TEST_CASE("Matrix Addition")
{
    using matrix_t = Matrix::Representation; 

    matrix_t matrix_with_ones = matrix_t(Matrix::Rows(20), Matrix::Columns(100));
    matrix_t test_output      = matrix_t(Matrix::Rows(20), Matrix::Columns(100));
    

    Matrix::Generation::Tester<1> init_as_one;
    Matrix::Generation::Tester<2> init_as_two;
    
    matrix_with_ones = init_as_one(matrix_with_ones);
    test_output = init_as_two(test_output);


    Matrix::Operations::Binary::Addition::Std naive_add;

    matrix_t sum = naive_add(matrix_with_ones, matrix_with_ones);


    SUBCASE("Cilk-for Multiplication")
    {

        CHECK((matrix_t{sum} == matrix_t{test_output}) == true);
    }

}