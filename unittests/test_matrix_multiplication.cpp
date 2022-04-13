#include "../deps/catch.hpp"

#include "../include/matrix.h"
#include "../include/generator.h"
#include "../include/m_algorithms.h"


TEST_CASE("Matrix Multiplication", "[arithmetic]")
{
    std::unique_ptr<Matrix::Representation> mb = std::make_unique<Matrix::Representation>(Matrix::Rows(1000), Matrix::Columns(300));
    std::unique_ptr<Matrix::Representation> ma = std::make_unique<Matrix::Representation>(Matrix::Rows(200), Matrix::Columns(1000));
    

    Matrix::Generation::Normal<0, 1> normal_distribution_init;
    
    ma = normal_distribution_init(std::move(ma));
    mb = normal_distribution_init(std::move(mb));


    Matrix::Operations::Binary::Multiplication::Naive naive_mul;
    Matrix::Operations::Binary::Multiplication::Square c_mul;
    Matrix::Operations::Binary::Multiplication::ParallelDNC r_mul;

    std::unique_ptr<Matrix::Representation> mc = naive_mul(std::move(ma), std::move(mb));
    std::unique_ptr<Matrix::Representation> md = c_mul(std::move(ma), std::move(mb));
    std::unique_ptr<Matrix::Representation> me = r_mul(std::move(ma), std::move(mb));



    SECTION("Cilk-for Multiplication")
    {

        REQUIRE((*mc == *md) == true);
    }



    SECTION("Recursive Parallel Multiplication")
    {
        REQUIRE((*mc == *me) == true);
    }

}