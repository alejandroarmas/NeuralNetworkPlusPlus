#ifndef NETWORK_LAYER_H
#define NETWORK_LAYER_H

#include <cstdint>
#include <memory>
#include <functional>

#include "functions.h"
#include "generator.h"
#include "matrix.h"

#include <map>

#define FLAT 1
    

namespace NeuralNetwork {



    class ComputationalStep {

        public:
            virtual ~ComputationalStep() = default;
            virtual std::unique_ptr<Matrix::Representation> forward(std::unique_ptr<Matrix::Representation> input) = 0;            
    };


    class Layer: public ComputationalStep {

        public:
            Layer(u_int64_t _l, u_int64_t _w, Matrix::Generation::Base& matrix_init) : 
                weights(std::make_unique<Matrix::Representation>(_l, _w)), 
                bias(std::make_unique<Matrix::Representation>(FLAT, _w)) {
                    
                    this->weights = matrix_init(std::move(this->weights));
                    this->bias    = matrix_init(std::move(this->bias));
            }            
            std::unique_ptr<Matrix::Representation> forward(std::unique_ptr<Matrix::Representation> input);
        private:
            std::unique_ptr<Matrix::Representation> weights;
            std::unique_ptr<Matrix::Representation> bias;
    };


    /*
    DESCRIPTION:


    USAGE:
    
    using matrix_t = Matrix::Representation; 
    std::unique_ptr<matrix_t> ma = std::make_unique<matrix_t>(1, 2000);
    Matrix::Generation::Normal<0, 1> normal_distribution_init;
    NeuralNetwork::Sequential model;
    model.add(std::make_unique<NeuralNetwork::Layer>(2000, 1000, normal_distribution_init));
    model.add(std::make_unique<NeuralNetwork::Layer>(1000, 10, normal_distribution_init));
    model.forward(std::move(ma));
    */
    class Sequential: public ComputationalStep {
        public:
            Sequential() : last_key(0) {}
            std::unique_ptr<Matrix::Representation> forward(std::unique_ptr<Matrix::Representation> input);
            void add(std::unique_ptr<ComputationalStep> layer);
        private:
            std::map<const unsigned int, std::unique_ptr<ComputationalStep>> _modules;
            unsigned int last_key;



    };



}



#endif // NETWORK_LAYER_H