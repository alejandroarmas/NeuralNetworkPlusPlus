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
*/

namespace NeuralNetwork {

    namespace Computation {

        namespace Graph {

            class Tensor;

            class RegisteredOperation : std::enable_shared_from_this<RegisteredOperation> {

                using cgNode   = std::shared_ptr<RegisteredOperation>; 
                using T        = std::shared_ptr<Tensor>;
                using NodePair = std::pair<cgNode, cgNode>;

                public:

                    constexpr Matrix::Operations::Code get_operation_code(void) { return m_type; }
                    T share_tensor () { return result; }

                    static std::shared_ptr<RegisteredOperation> create(
                            const Matrix::Operations::Code _typ, T _res, 
                            cgNode _op = nullptr, cgNode _op2 = nullptr) {

                            return std::shared_ptr<RegisteredOperation>(
                                new RegisteredOperation(_typ, _res, _op, _op2)
                                );
                            
                    }

                    std::shared_ptr<RegisteredOperation> get_operation(void) {
                        return shared_from_this();
                    }


                    NodePair get_operands(void) {

                        if (operand && bin_operand) {
                            return {
                                    this->operand->get_operation(), 
                                    this->bin_operand->get_operation()
                            };
                        }
                        else if (operand) {
                            return {
                                    this->operand->get_operation(), 
                                    nullptr
                            };
                        }
                        else if (bin_operand) {
                            return {
                                    nullptr,
                                    this->bin_operand->get_operation() 
                            };
                        }

                        return {
                                nullptr,
                                nullptr 
                        };


                     }
                
                
                protected:
                    const Matrix::Operations::Code m_type;
                    T result;
                    cgNode operand;
                    cgNode bin_operand;
                private:
                    RegisteredOperation(const Matrix::Operations::Code _typ, T _res, 
                        cgNode _op, cgNode _op2) : 
                        m_type(_typ), result(_res), 
                        operand(std::move(_op)), 
                        bin_operand(std::move(_op2)) {}
                
            };

         
        }

    }

}




#endif // TENSOR_ALGORITHM_REGISTER_H