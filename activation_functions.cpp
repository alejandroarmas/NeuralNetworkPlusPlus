#include <memory>
#include <algorithm>

#include "activation_functions.h"
#include "matrix.h"
#include "m_algorithms.h"
#include "matrix_printer.h"
#include "matrix_benchmark.h"
#include "config.h"



std::unique_ptr<Matrix::Representation> NeuralNetwork::ActivationFunctions::ReLU::forward(std::unique_ptr<Matrix::Representation> input) {

    if (input == nullptr) {
        throw std::invalid_argument("Matrix has no data (pointing to null).");
    }
    
    Matrix::Operations::Timer relu(
        std::make_unique<Matrix::Operations::Unary::ReLU>());



    // Matrix::Operations::Unary::ReLU relu;

    std::unique_ptr<Matrix::Representation> output = relu(std::move(input));

#if DEBUG
    Matrix::Printer m_printer;
    output = m_printer(std::move(output));
#endif


    return output;
}

