#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <memory>

#include "network_layer.h"
#include "tensor.h"


#define FLAT 1
    

namespace NeuralNetwork {

    namespace ActivationFunctions {

        class ReLU: public ComputationalStep<ReLU> {

            public:     
                std::shared_ptr<Computation::Graph::Tensor> doForward(std::shared_ptr<Computation::Graph::Tensor> input);
        };

    }

}



#endif // ACTIVATION_FUNCTIONS_H