#include <iostream>
#include <vector>
#include <memory>

#include "mm.h"
#include "generator.h"
#include "matrix_printer.h"
#include "network_layer.h"



int main(void) {

    using matrix_t = Matrix::Representation<double>; 

    std::unique_ptr<matrix_t> ma = std::make_unique<matrix_t>(20, 1);


    Matrix::Generator<double> matrix_init;

    ma = matrix_init(std::move(ma));

    // mb = matrix_init(std::move(mb));

    Matrix::Printer<double> m_printer;

    m_printer(*ma);

    NeuralNetwork::Layer<double> l(10, 20);

    l.predict(std::move(ma));

 
    

 

 
    return 0;
}