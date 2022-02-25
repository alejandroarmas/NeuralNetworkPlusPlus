
#include <memory>

#include "matrix.h"
#include "network_layer.h" 
#include "m_algorithms.h"
#include "matrix_printer.h"
#include "config.h"


std::unique_ptr<Matrix::Representation> NeuralNetwork::Layer::predict(std::unique_ptr<Matrix::Representation> input) {

    Matrix::Operations::Multiplication::Naive mm;
    Matrix::Operations::Addition::Std add;


    auto out = mm(this->weights, input);

    z = add(out, this->bias);



#if DEBUG
    Matrix::Printer m_printer;
    m_printer(*z);
#endif


    

    return std::move(z);
}