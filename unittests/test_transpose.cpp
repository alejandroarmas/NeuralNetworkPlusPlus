#include "../deps/doctest.h"

#include "../include/matrix.h"
#include "../include/generator.h"
#include "../include/m_algorithms.h"


TEST_CASE("Matrix Transpose")
{

    Matrix::Representation ma = Matrix::Representation(
        Matrix::Rows(200), Matrix::Columns(1000));
    

    Matrix::Generation::Normal<0, 1> normal_distribution_init;
    
    ma = normal_distribution_init(ma);


    Matrix::Operations::Unary::Transpose operation;

    Matrix::Representation mb = operation(ma);
    Matrix::Representation mc = operation(mb);



    SUBCASE("Cache Oblivious Transpose")
    {

        CHECK((Matrix::Representation{ma} == Matrix::Representation{mc}) == true);
    }


}