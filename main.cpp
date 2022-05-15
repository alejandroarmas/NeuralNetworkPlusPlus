#include <memory>
#include <iostream>

#include "tensor_forward_wrapper.h"
#include "generator.h"
#include "matrix_printer.h"
#include "network_layer.h"
#include "activation_layer.h"
#include "m_algorithms.h"
#include "matrix_benchmark.h"
#include "context_object.h"
#include "function_object.h"



int main(void) {


    auto ma = NeuralNetwork::Computation::Graph::TensorConstructor::create(Matrix::Rows(1), Matrix::Columns(2000));
    auto ground_truth = NeuralNetwork::Computation::Graph::TensorConstructor::create(Matrix::Rows(1), Matrix::Columns(10), 
        NeuralNetwork::Computation::Graph::IsTrackable(true), 
        NeuralNetwork::Computation::Graph::IsLeaf(true), 
        NeuralNetwork::Computation::Graph::IsRecordable(true));

    NeuralNetwork::Sequential model;

    model.add(std::make_unique<NeuralNetwork::Layer>(
            std::make_unique<NeuralNetwork::MatrixMultiplyStep>(Matrix::Rows(2000), Matrix::Columns(1000)),
            std::make_unique<NeuralNetwork::AddStep>(Matrix::Columns(1000))    
    ));
    model.add(std::make_unique<NeuralNetwork::ActivationFunctions::ReLU>());
    model.add(std::make_unique<NeuralNetwork::Layer>(
            std::make_unique<NeuralNetwork::MatrixMultiplyStep>(Matrix::Rows(1000), Matrix::Columns(10)),
            std::make_unique<NeuralNetwork::AddStep>(Matrix::Columns(10))    
    ));
    model.add(std::make_unique<NeuralNetwork::ActivationFunctions::SoftMax>());
    
    auto CE = NeuralNetwork::Computation::Graph::TensorOp(Matrix::Operations::Metric::CrossEntropy{});

    for (int i = 0; i < 10; i++) {
        auto out  = model.forward(ma);
        auto loss = CE(out, ground_truth);
        loss->backwards();
    }





    return 0;
}