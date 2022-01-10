#include "../deps/catch.hpp"

#include "../include/mm.h"
#include "../include/generator.h"
#include "../include/m_algorithms.h"


TEST_CASE("Matrix Multiplication", "[arithmetic]")
{
    using matrix_t = Matrix::Representation; 

    std::unique_ptr<matrix_t> ma = std::make_unique<matrix_t>(20, 100);
    std::unique_ptr<matrix_t> mb = std::make_unique<matrix_t>(100, 30);
    

    Matrix::Generation::Normal<0, 1> normal_distribution_init;
    
    ma = normal_distribution_init(std::move(ma));
    mb = normal_distribution_init(std::move(mb));


    Matrix::Operations::Multiplication::Naive naive_mul;
    Matrix::Operations::Multiplication::Square c_mul;

    std::unique_ptr<Matrix::Representation> mc = naive_mul(*ma, *mb);
    std::unique_ptr<Matrix::Representation> md = c_mul(*ma, *mb);



    SECTION("Cilk-for Multiplication")
    {

        REQUIRE((*mc == *md) == true);
    }

}