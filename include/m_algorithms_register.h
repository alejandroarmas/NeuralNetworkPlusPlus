#ifndef TENSOR_ALGORITHM_REGISTER_H
#define TENSOR_ALGORITHM_REGISTER_H


// #include "tensor.h"
#include "matrix.h"
#include "m_algorithms.h"
#include "strong_types.h"

#include <utility>    // std::pair
#include <optional>

/*

Visitor interface for all Computational Steps
and then Tensor inherits from that class.

Visitor Polymorphism depending on task,
reading for creating graph, writing data back,
*/

namespace NeuralNetwork {

    namespace Computation {

        namespace Graph {

            class Tensor;


            /*
                Compile time behavior based on passing tag into constructor.
                This is based on MatrixOperatable concept, which supports
                BinaryOperable or UnaryOperable behavior.
                Type erasure comes at a price, so we don't want to wrap our
                ops .

            */
            class RegisteredTensor { 

                public:
                    // RegisteredTensor(RegisteredTensor&) = default; 
                    // RegisteredTensor(RegisteredTensor&&) = default; 
                    // RegisteredTensor& operator=(const RegisteredTensor&) = default; 
                    // RegisteredTensor& operator=(RegisteredTensor&&) = default; 

                    RegisteredTensor operator=(const RegisteredTensor& other) {
                        result = other.result;
                        return *this;
                    }
                    constexpr TensorID get_tensor_id() const noexcept { return result; }
                    RegisteredTensor(TensorID _res
                         = TensorID(0)) : 
                        result(_res) {}
                protected:
                    TensorID result;
                
            };

            class RegisteredUnaryOperation : public RegisteredTensor { 

                public:
                    // RegisteredUnaryOperation(RegisteredUnaryOperation&) = default; 
                    // RegisteredUnaryOperation(RegisteredUnaryOperation&&) = default; 
                    // RegisteredUnaryOperation& operator=(const RegisteredUnaryOperation&) = default; 
                    // RegisteredUnaryOperation& operator=(RegisteredUnaryOperation&&) = default; 
                    RegisteredUnaryOperation operator=(const RegisteredUnaryOperation& other) {
                        result = other.result;
                        operand = other.operand;
                        return *this;
                    }
                    RegisteredUnaryOperation(TensorID _res, 
                        TensorID _op) : 
                        RegisteredTensor(_res), operand(_op) {}
                    
                    TensorID left_op_id()    const { return operand; }

                protected:
                    TensorID operand;
                
            };

            class RegisteredBinaryOperation : public RegisteredUnaryOperation { 

                public:

                    // RegisteredBinaryOperation(RegisteredBinaryOperation&) = default; 
                    // RegisteredBinaryOperation(RegisteredBinaryOperation&&) = default; 
                    // RegisteredBinaryOperation& operator=(const RegisteredBinaryOperation&) = default; 
                    // RegisteredBinaryOperation& operator=(RegisteredBinaryOperation&&) = default; 

                    RegisteredBinaryOperation operator=(const RegisteredBinaryOperation& other) {
                        result = other.result;
                        operand = other.operand;
                        bin_operand = other.bin_operand;
                        return *this;
                    }
                    RegisteredBinaryOperation(TensorID _res, 
                        TensorID _op, 
                        TensorID _op2) : 
                        RegisteredUnaryOperation(_res, _op), bin_operand(_op2) {}
                    
                    TensorID right_op_id()    const { return bin_operand; }

                private:
                    TensorID bin_operand;
                
            };
         
        }

    }

}




#endif // TENSOR_ALGORITHM_REGISTER_H