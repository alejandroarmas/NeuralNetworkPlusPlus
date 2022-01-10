#include "../deps/catch.hpp"

#include "../include/mm.h"
#include "../include/generator.h"
#include "../include/m_algorithms.h"


TEST_CASE("Matrix Addition", "[arithmetic]")
{
    using matrix_t = Matrix::Representation; 

    std::unique_ptr<matrix_t> matrix_with_ones = std::make_unique<matrix_t>(20, 100);
    std::unique_ptr<matrix_t> test_output = std::make_unique<matrix_t>(20, 100);
    

    Matrix::Generation::Tester<1> init_as_one;
    Matrix::Generation::Tester<2> init_as_two;
    
    matrix_with_ones = init_as_one(std::move(matrix_with_ones));
    test_output = init_as_two(std::move(test_output));


    Matrix::Operations::Addition::Std naive_add;

    std::unique_ptr<Matrix::Representation> sum = naive_add(*matrix_with_ones, *matrix_with_ones);


    SECTION("Cilk-for Multiplication")
    {

        REQUIRE((*sum == *test_output) == true);
    }

}