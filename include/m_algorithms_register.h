#ifndef TENSOR_ALGORITHM_REGISTER_H
#define TENSOR_ALGORITHM_REGISTER_H


#include "tensor.h"

#include <utility>    // std::pair

/*

Visitor interface for all Computational Steps
and then Tensor inherits from that class.

Visitor Polymorphism depending on task,
reading for creating graph, writing data back,
...
transfer number: 7192 
1-800-829-1040
*/

namespace NeuralNetwork {

    namespace Computation {

        namespace Graph {

            class Tensor;

            class RegisteredOperation { 

                using cgNode   = std::shared_ptr<RegisteredOperation>; 
                using T        = Tensor&;
                using NodePair = std::pair<cgNode, cgNode>;

                public:
                    RegisteredOperation(Matrix::Operations::Code _typ, 
                        T _res, cgNode _op = nullptr, 
                        cgNode _op2 = nullptr) : 
                        m_type(_typ), result(_res), 
                        operand(_op), bin_operand(_op2) {}

                    T share_tensor() const { return result; }
                    const Matrix::Operations::Code get_code() const { return m_type; }
                    NodePair get_operands(void) const;
                
                private:
                    const Matrix::Operations::Code m_type;
                    T result;
                    cgNode operand;
                    cgNode bin_operand;
                
            };


            class OperationFactory {

                using cgNode   = std::shared_ptr<RegisteredOperation>; 
                using T        = Tensor&;
                using NodePair = std::pair<cgNode, cgNode>;
                
                public:
                    static std::shared_ptr<RegisteredOperation> create(
                            const Matrix::Operations::Code _typ, T _res, 
                            cgNode _op = nullptr, cgNode _op2 = nullptr) {

                            return std::make_shared<RegisteredOperation>(
                                    _typ, 
                                    _res, 
                                    _op, 
                                    _op2
                                );
                            
                    }
            };

         
        }

    }

}




#endif // TENSOR_ALGORITHM_REGISTER_H