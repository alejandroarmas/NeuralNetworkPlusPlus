#include <memory>
#include <algorithm>  // std::for_each

#include "tensor.h"
#include "network_layer.h" 
#include "m_algorithms.h"
// #include "matrix_printer.h"
#include "matrix_benchmark.h"
#include "config.h"
#include <chrono>

namespace NeuralNetwork {



    std::shared_ptr<Tensor> MatrixMultiplyStep::_doForward(std::shared_ptr<Tensor> input) {

        TensorOp mm(Matrix::Operations::Binary::Multiplication::ParallelDNC{});



        auto out = mm(input, this->matrix);

    // #if DEBUG
    //     Matrix::Printer m_printer;
    //     out = m_printer(std::move(out));
    // #endif


        return out;
    }



    std::shared_ptr<Tensor> AddStep::_doForward(std::shared_ptr<Tensor> input) {



        TensorOp add(Matrix::Operations::Binary::Addition::Std{});



        auto z = add(this->matrix, input);


    // #if DEBUG
    //     Matrix::Printer m_printer;
    //     z = m_printer(std::move(z));
    // #endif

        return z;
    }


    std::shared_ptr<Tensor> Layer::doForward(std::shared_ptr<Tensor> input) {

        if (input == nullptr) {
            throw std::invalid_argument("Matrix has no data (pointing to null).");
        }


        auto out = this->weights->forward(input);
        
        auto z = this->bias->forward(out);


        return z;
    }


    void Layer::_add(std::unique_ptr<StepInterface> matrix) {

        if (matrix == nullptr) {
            throw std::invalid_argument("Matrix has no data (pointing to null).");
        }

        if (this->weights == nullptr) {
            this->weights = std::move(matrix);
        }
        else if (this->bias == nullptr) {
            this->bias    = std::move(matrix);
        }

    }


    std::shared_ptr<Tensor> Sequential::doForward(std::shared_ptr<Tensor> input) {

        std::shared_ptr<Tensor> current_value = input;

        std::for_each(this->_modules.begin(), this->_modules.end(), 
            [&current_value](std::pair<const unsigned int, std::unique_ptr<StepInterface>>& _layer){

                current_value = _layer.second->forward(current_value);
            });


        return current_value;
    }


    void Sequential::_add(std::unique_ptr<StepInterface> layer) {
        this->_modules.emplace(this->last_key++, std::move(layer));
    }

}