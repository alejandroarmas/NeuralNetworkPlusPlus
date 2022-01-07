#ifndef NETWORK_LAYER_H
#define NETWORK_LAYER_H

#include <cstdint>
#include <memory>
#include <functional>

#include "functions.h"
#include "generator.h"
#include "mm.h"


#define FLAT 1


namespace NeuralNetwork {


    class BaseLayer {

        public:
            virtual ~BaseLayer() = default;
            virtual std::unique_ptr<Matrix::Representation> predict(std::unique_ptr<Matrix::Representation> input) = 0;            
    };


    class Layer: public BaseLayer {

        public:
            Layer(u_int64_t _l, u_int64_t _w, Matrix::Generation::Base& matrix_init) : 
                weights(std::make_unique<Matrix::Representation>(_l, _w)), 
                bias(std::make_unique<Matrix::Representation>(_l, FLAT)),
                z(nullptr) {
                    
                    this->weights = matrix_init(std::move(this->weights));
                    this->bias    = matrix_init(std::move(this->bias));
            }            
            std::unique_ptr<Matrix::Representation> predict(std::unique_ptr<Matrix::Representation> input);
        private:
            std::unique_ptr<Matrix::Representation> weights;
            std::unique_ptr<Matrix::Representation> bias;
            std::unique_ptr<Matrix::Representation> z;
            // std::unique_ptr<ActivationFunctions::BaseFunction> activation;
    };



}



#endif // NETWORK_LAYER_H