
#include <math.h>

#include "functions.h"




template <class T>
T ActivationFunctions::Sigmoid<T>::operate(T x) {

    T denominator = 1 + ::exp(x); 
    T output = 1 / denominator; 

    return output;
}

template <class T>
T ActivationFunctions::Differentiable::Sigmoid<T>::operate(T x) {

    ActivationFunctions::Sigmoid<T> sigmoid;

    T result = sigmoid.operate(x);
 
    T output = result * (1 - result);

    return output;
}
