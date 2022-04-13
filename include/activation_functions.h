#ifndef ACTIVATION_FUNCTIONS
#define ACTIVATION_FUNCTIONS

#include <cstdint>
#include <memory>
#include <functional>

#include "network_layer.h"
#include "functions.h"
#include "matrix.h"

#include <map>

#define FLAT 1
    

namespace NeuralNetwork {

    namespace ActivationFunctions {

        class ReLU: public ComputationalStep<ReLU> {

            public:     
                std::unique_ptr<Matrix::Representation> forward(std::unique_ptr<Matrix::Representation> input);
        };

    }

}



#endif // ACTIVATION_FUNCTIONS