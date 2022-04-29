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

    auto ma = std::make_shared<NeuralNetwork::Computation::Graph::Tensor>(Matrix::Rows(1), Matrix::Columns(2000));

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
    model.add(std::make_unique<NeuralNetwork::ActivationFunctions::ReLU>());
    

    auto out = model.forward(ma);

    out->backwards();



    return 0;
}