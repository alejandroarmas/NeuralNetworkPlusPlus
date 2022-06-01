#include "../deps/doctest.h"

#include "../include/matrix.h"
#include "../include/generator.h"
#include "../include/m_algorithms.h"
#include "../include/functions.h"

#include <numeric>

TEST_CASE("Softmax Operation")
{

    Matrix::Generation::Normal<0, 1> normal_distribution_init;
    Matrix::Operations::Unary::SoftMax softmax;


    SUBCASE("Column Vector Sum")
    {
        
        Matrix::Representation ma = Matrix::Representation(
            Matrix::Rows(10), Matrix::Columns(1));
            ma = normal_distribution_init(ma);

        Matrix::Representation mb = softmax(ma);

        double total = std::accumulate(mb.scanStart(), mb.scanEnd(), 0.0);

        CHECK(Functions::Utility::compare_float(total, 1.0) == true);
    }

    SUBCASE("One Cell Matrix Sum"){
        
        Matrix::Representation mc = Matrix::Representation(
            Matrix::Rows(1), Matrix::Columns(1));
        mc = normal_distribution_init(mc);

        Matrix::Representation md = softmax(mc);
        double total = std::accumulate(md.constScanStart(), md.constScanEnd(), 0.0);

        CHECK(Functions::Utility::compare_float(total, 1.0) == true);

    }


}