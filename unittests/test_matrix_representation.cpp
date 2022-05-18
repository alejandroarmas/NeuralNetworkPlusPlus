#include "../deps/doctest.h"

#include "../include/matrix.h"
#include "../include/generator.h"


TEST_CASE("Matrix Equality Test")
{
    using matrix_t = Matrix::Representation; 

    matrix_t ma = matrix_t(Matrix::Rows(2000), Matrix::Columns(100));
    matrix_t mb = matrix_t(Matrix::Rows(100), Matrix::Columns(3000));
    matrix_t mc = matrix_t(Matrix::Rows(10), Matrix::Columns(30));
    matrix_t md = matrix_t(Matrix::Rows(10), Matrix::Columns(30));

    Matrix::Generation::Normal<0, 1> normal_distribution_init;
    
    Matrix::Generation::Tester<3> constant_init; 

    ma = normal_distribution_init(ma);
    mb = normal_distribution_init(mb);

    mc = constant_init(mc);
    md = constant_init(md);


    SUBCASE("Normal Cases")
    {

        CHECK((ma == ma) == true);
        CHECK((mb == mb) == true);
        CHECK((ma != mb) == true);
        CHECK((ma != mb) == !(ma == mb));
        CHECK((mc == md) == true);
    }

}