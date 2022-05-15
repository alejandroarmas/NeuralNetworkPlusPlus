#ifndef TENSOR_FUNCTION_OBJECT_FACTOR_H  
#define TENSOR_FUNCTION_OBJECT_FACTOR_H

#include "strong_types.h" 
#include "m_algorithms_register.h"
#include "m_algorithms_concepts.h"
#include "function_object.h"

#include <optional>

namespace NeuralNetwork {

    namespace Computation {

        namespace Graph {

            class Tensor;

            // class DereferenceRegistry {
            //     static std::optional<std::shared_ptr<Tensor>> from(RegisteredOperation _op);

            // };



            // class OperationFactory {

            //     using T        = std::shared_ptr<Tensor>; 
            //     using NodePair = std::pair<TensorID, TensorID>;
                
            //     public:
            //         static void create(
            //                 const Matrix::Operations::Code _typ, T _res, 
            //                 TensorID _op = TensorID(0), TensorID _op2 = TensorID(0));            
                            
            // };


            class FunctionObjectFactory {

                using T = std::shared_ptr<Tensor>; 
                
                public:
                    static FunctionObject create(T _res);            

                template <Matrix::Operations::BinaryMatrixOperatable RegisteryType>
                    static FunctionObject create(
                        RegisteryType operation, T _res, 
                        TensorID _operand_id, TensorID _operand_id_two);

                template <Matrix::Operations::UnaryMatrixOperatable RegisteryType>
                    static FunctionObject create(
                        RegisteryType operation, T _res, TensorID _operand_id);

            };




        }
    }
}


#endif // TENSOR_FUNCTION_OBJECT_FACTOR_H  
