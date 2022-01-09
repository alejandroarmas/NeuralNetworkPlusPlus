#include "../deps/catch.hpp"

#include "../include/mm.h"
#include "../include/generator.h"


TEST_CASE("Equality Test", "[arithmetic]")
{
    using matrix_t = Matrix::Representation; 

    std::unique_ptr<matrix_t> ma = std::make_unique<matrix_t>(2000, 100);
    std::unique_ptr<matrix_t> mb = std::make_unique<matrix_t>(100, 3000);
    std::unique_ptr<matrix_t> mc = std::make_unique<matrix_t>(10, 30);
    std::unique_ptr<matrix_t> md = std::make_unique<matrix_t>(10, 30);

    Matrix::Generation::Normal<0, 1> normal_distribution_init;
    
    Matrix::Generation::Tester<3> constant_init; 

    ma = normal_distribution_init(std::move(ma));
    mb = normal_distribution_init(std::move(mb));

    mc = constant_init(std::move(mc));
    md = constant_init(std::move(md));


    SECTION("Normal Cases")
    {

        REQUIRE((*ma == *ma) == true);
        REQUIRE((*mb == *mb) == true);
        REQUIRE((*ma != *mb) == true);
        REQUIRE((*ma != *mb) == !(*ma == *mb));
        REQUIRE((*mc == *md) == true);
    }

}