#include <memory>
#include <algorithm>

#include "activation_functions.h"
#include "matrix.h"
#include "m_algorithms.h"
#include "matrix_printer.h"
#include "config.h"



std::unique_ptr<Matrix::Representation> NeuralNetwork::ActivationFunctions::ReLU::forward(std::unique_ptr<Matrix::Representation> input) {

    
    std::unique_ptr<Matrix::Representation> output = std::make_unique<Matrix::Representation>(input->num_rows(), input->num_cols());


    std::replace_copy_if(input->scanStart(), input->scanEnd(), output->scanStart(), 
        [](float z){ return z < 0;}, 0);


    return output;
}

