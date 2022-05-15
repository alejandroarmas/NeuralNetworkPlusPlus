#include <memory>
#include <assert.h>

#include "tensor_forward_wrapper.h"
#include "activation_layer.h"
#include "m_algorithms.h"
// #include "matrix_printer.h"
#include "matrix_benchmark.h"
#include "config.h"


namespace NeuralNetwork {

    std::shared_ptr<Computation::Graph::Tensor> NeuralNetwork::ActivationFunctions::ReLU::doForward(std::shared_ptr<Computation::Graph::Tensor> input) noexcept{

        assert(input != nullptr && "Matrix has no data (pointing to null).");
                
        Computation::Graph::TensorOp relu(Matrix::Operations::Unary::ReLU{});


        std::shared_ptr<Computation::Graph::Tensor> output = relu(input);

    // #if DEBUG
    //     Matrix::Printer m_printer;
    //     output = m_printer(std::move(output));
    // #endif


        return output;
    }
    
    std::shared_ptr<Computation::Graph::Tensor> NeuralNetwork::ActivationFunctions::SoftMax::doForward(std::shared_ptr<Computation::Graph::Tensor> input) noexcept{

        assert(input != nullptr && "Matrix has no data (pointing to null).");
                
        Computation::Graph::TensorOp softmax(Matrix::Operations::Unary::SoftMax{});


        std::shared_ptr<Computation::Graph::Tensor> output = softmax(input);


        return output;
    }

    


}



