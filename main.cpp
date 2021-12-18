#include <iostream>
#include <vector>
#include <memory>

#include "mm.h"
#include "generator.h"
#include "matrix_printer.h"


int main(void) {

    using matrix_t = Matrix::Representation<double>; 

    std::unique_ptr<matrix_t> ma = std::make_unique<matrix_t>(6, 2);
    std::unique_ptr<matrix_t> mb = std::make_unique<matrix_t>(2, 4);


    Matrix::Generator<double> matrix_init;

    ma = matrix_init(std::move(ma));

    mb = matrix_init(std::move(mb));

    Matrix::Printer<double> m_printer;
    
    MatrixMultiplication::Naive<double> mm;

    m_printer(*ma);
    m_printer(*mb);

    std::unique_ptr<matrix_t> out = mm(*ma, *mb);

    m_printer(*out);

    return 0;
}