
#include <memory>
#include <algorithm>

#include "matrix.h"
#include "network_layer.h" 
#include "m_algorithms.h"
#include "matrix_printer.h"
#include "config.h"


std::unique_ptr<Matrix::Representation> NeuralNetwork::Layer::forward(std::unique_ptr<Matrix::Representation> input) {

    Matrix::Operations::Multiplication::ParallelDNC mm;
    Matrix::Operations::Addition::Std add;


    auto out = mm(input, this->weights);

    auto z = add(this->bias, out);


#if DEBUG
    Matrix::Printer m_printer;
    m_printer(*z);
#endif


    return z;
}


std::unique_ptr<Matrix::Representation> NeuralNetwork::Sequential::forward(std::unique_ptr<Matrix::Representation> input) {

    std::unique_ptr<Matrix::Representation> current_value = std::move(input);

    std::for_each(this->_modules.begin(), this->_modules.end(), 
        [&current_value](std::pair<const unsigned int, std::unique_ptr<ComputationalStep>>& _layer){

            current_value = _layer.second->forward(std::move(current_value));
        });


    return current_value;
}


void NeuralNetwork::Sequential::add(std::unique_ptr<ComputationalStep> layer) {
    this->_modules.emplace(this->last_key++, std::move(layer));
}
