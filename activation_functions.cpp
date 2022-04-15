#include <memory>

#include "activation_functions.h"
#include "m_algorithms.h"
// #include "matrix_printer.h"
#include "matrix_benchmark.h"
#include "config.h"


namespace NeuralNetwork {

    std::shared_ptr<Computation::Graph::Tensor> NeuralNetwork::ActivationFunctions::ReLU::forward(std::shared_ptr<Computation::Graph::Tensor> input) {

        if (input == nullptr) {
            throw std::invalid_argument("Matrix has no data (pointing to null).");
        }
        
        
        
        Computation::Graph::TensorOp relu(std::make_unique<Matrix::Operations::Timer>(
            std::make_unique<Matrix::Operations::Unary::ReLU>()));



        std::shared_ptr<Computation::Graph::Tensor> output = relu(input);

    // #if DEBUG
    //     Matrix::Printer m_printer;
    //     output = m_printer(std::move(output));
    // #endif


        return output;
    }

}



