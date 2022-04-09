
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

    if (input == nullptr) {
        throw std::invalid_argument("Matrix has no data (pointing to null).");
    }


    auto out = mm(input, this->weights);

    auto z = add(this->bias, out);


#if DEBUG
    Matrix::Printer m_printer;
    z = m_printer(std::move(z));
#endif


    return z;
}


std::unique_ptr<Matrix::Representation> NeuralNetwork::Sequential::forward(std::unique_ptr<Matrix::Representation> input) {

    std::unique_ptr<Matrix::Representation> current_value = std::move(input);

    std::for_each(this->_modules.begin(), this->_modules.end(), 
        [&current_value](std::pair<const unsigned int, std::unique_ptr<StepInterface>>& _layer){

            current_value = _layer.second->forward(std::move(current_value));
        });


    return current_value;
}


void NeuralNetwork::Sequential::add(std::unique_ptr<StepInterface> layer) {
    this->_modules.emplace(this->last_key++, std::move(layer));
}
