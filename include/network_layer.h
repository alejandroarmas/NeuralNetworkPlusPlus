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

    template <class T>
    class BaseLayer {

        public:
            virtual ~BaseLayer() = default;
            virtual std::unique_ptr<Matrix::Representation<T>> predict(std::unique_ptr<Matrix::Representation<T>> input) = 0;            
    };


    template <class T>
    class Layer: public BaseLayer<T> {

        public:
            Layer(u_int64_t _l, u_int64_t _w, Matrix::Generation::Base<T>& matrix_init) : 
                weights(std::move(std::make_unique<Matrix::Representation<T>>(_l, _w))), 
                bias(std::move(std::make_unique<Matrix::Representation<T>>(_l, FLAT))),
                z(nullptr) {
                    
                    this->weights = matrix_init(std::move(this->weights));
                    this->bias    = matrix_init(std::move(this->bias));
            }            
            std::unique_ptr<Matrix::Representation<T>> predict(std::unique_ptr<Matrix::Representation<T>> input);
        private:
            std::unique_ptr<Matrix::Representation<T>> weights;
            std::unique_ptr<Matrix::Representation<T>> bias;
            std::unique_ptr<Matrix::Representation<T>> z;
            // std::unique_ptr<ActivationFunctions::BaseFunction> activation;
    };

    template class Layer<float>;
    template class Layer<double>;



}



#endif // NETWORK_LAYER_H