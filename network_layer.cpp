
#include <memory>

#include "mm.h"
#include "network_layer.h" 
#include "m_algorithms.h"
#include "matrix_printer.h"
#include "config.h"


template <class T>
std::unique_ptr<Matrix::Representation<T>> NeuralNetwork::Layer<T>::predict(std::unique_ptr<Matrix::Representation<T>> input) {

    Matrix::Operations::Multiplication::Naive<T> mm;
    Matrix::Operations::Add::Std<T> add;


    auto out = mm(*this->weights, *input);

    z = add(*out, *this->bias);



#if DEBUG
    Matrix::Printer<T> m_printer;
    m_printer(*z);
#endif


    

    return std::move(z);
}