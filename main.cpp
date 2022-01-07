#include <memory>

#include "mm.h"
#include "generator.h"
#include "matrix_printer.h"
#include "network_layer.h"



int main(void) {

    using matrix_t = Matrix::Representation; 

    std::unique_ptr<matrix_t> ma = std::make_unique<matrix_t>(20, 1);


    Matrix::Generation::Normal normal_distribution_init(0, 1);
    Matrix::Generation::Tester vec_init;



    ma = vec_init(std::move(ma));


    Matrix::Printer m_printer;

    m_printer(*ma);

    NeuralNetwork::Layer l(10, 20, normal_distribution_init);

    l.predict(std::move(ma));


    return 0;
}