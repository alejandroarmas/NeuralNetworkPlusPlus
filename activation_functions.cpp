#include <memory>
#include <algorithm>

#include "activation_functions.h"
#include "matrix.h"
#include "m_algorithms.h"
#include "matrix_printer.h"
#include "config.h"



std::unique_ptr<Matrix::Representation> NeuralNetwork::ActivationFunctions::ReLU::forward(std::unique_ptr<Matrix::Representation> input) {

    if (input == nullptr) {
        throw std::invalid_argument("Matrix has no data (pointing to null).");
    }
    

    auto f = [](std::unique_ptr<Matrix::Representation> input) {

        std::unique_ptr<Matrix::Representation> output = std::make_unique<Matrix::Representation>(
                    Matrix::Rows(input->num_rows()), 
                    Matrix::Columns(input->num_cols())
            );

        std::replace_copy_if(input->scanStart(), input->scanEnd(), output->scanStart(), 
            [](float z){ return z < 0;}, 0);

        return output;
    };

    std::unique_ptr<Matrix::Representation> output = f(std::move(input));

#if DEBUG
    Matrix::Printer m_printer;
    output = m_printer(std::move(output));
#endif


    return output;
}

