#include <memory>
#include <iostream>

#include "matrix.h"
#include "generator.h"
#include "matrix_printer.h"
#include "network_layer.h"
#include "activation_functions.h"
#include "m_algorithms.h"
#include "matrix_benchmark.h"
#include "context_object.h"


/*

model = nn.Sequential(
    nn.Linear(28*28, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)

*/

int main(void) {

    using matrix_t = Matrix::Representation; 

    std::unique_ptr<matrix_t> ma = std::make_unique<matrix_t>(Matrix::Rows(1), Matrix::Columns(2000));
    Matrix::Generation::Normal<0, 1> normal_distribution_init;
    ma = normal_distribution_init(std::move(ma));

    NeuralNetwork::Sequential model;

    model.add(std::make_unique<NeuralNetwork::Layer>(
            std::make_unique<NeuralNetwork::MatrixMultiplyStep>(Matrix::Rows(2000), Matrix::Columns(1000), normal_distribution_init),
            std::make_unique<NeuralNetwork::AddStep>(Matrix::Columns(1000), normal_distribution_init)    
    ));
    model.add(std::make_unique<NeuralNetwork::ActivationFunctions::ReLU>());
    model.add(std::make_unique<NeuralNetwork::Layer>(
            std::make_unique<NeuralNetwork::MatrixMultiplyStep>(Matrix::Rows(1000), Matrix::Columns(10), normal_distribution_init),
            std::make_unique<NeuralNetwork::AddStep>(Matrix::Columns(10), normal_distribution_init)    
    ));
    model.add(std::make_unique<NeuralNetwork::ActivationFunctions::ReLU>());
    

    auto out = model.forward(std::move(ma));

    // NeuralNetwork::Computation::Tree::ComputeOperation handler;
    // handler.setNextHandler(std::make_unique<NeuralNetwork::Computation::Tree::TimerHandler>());
    // handler.setNextHandler(std::make_unique<NeuralNetwork::Computation::Tree::CreateAutoDiffNode>());
    // handler.handle("test");


    return 0;
}