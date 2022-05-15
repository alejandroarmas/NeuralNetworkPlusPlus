#include "m_algorithms_register.h"
#include "computational_graph_map.h"

#include "m_algorithms.h"
#include "tensor.h"
#include <iostream>
#include <optional>

#include <memory>


namespace NeuralNetwork {

    namespace Computation {

        namespace Graph {

            // RegisteredOperation::T RegisteredOperation::share_tensor() const { 
                
            // ComputationalGraphMap& map = ComputationalGraphMap::get();
                
            //     return map._get_tensor(result);    
            // }


            // RegisteredOperation::NodePair RegisteredOperation::get_operands() const {

            //     ComputationalGraphMap& map = ComputationalGraphMap::get();

            //     if (operand.get() && bin_operand.get()) {
            //         auto t1 = map._get_tensor(operand);
            //         auto t2 = map._get_tensor(bin_operand);
            //         return {t1, t2};
            //     }
            //     else if (operand.get()) {
            //         auto t1 = map._get_tensor(operand);
            //         return {t1 , nullptr};                        
            //         }
            //     else if (bin_operand.get()) {
            //         auto t2 = map._get_tensor(bin_operand);
            //         return {nullptr, t2};                        
            //         }
            //     return {nullptr, nullptr};

            // }


            // std::optional<std::shared_ptr<Tensor>> DereferenceRegistry::from(RegisteredOperation _op) {
            //         if (_op.get_tensor_id() == TensorID(0)) return {};

            //         ComputationalGraphMap& map = ComputationalGraphMap::get();
            //         return map._get_tensor(_op.get_tensor_id());
                    
            //     }


            // void OperationFactory::create(
            //     const Matrix::Operations::Code _typ, T _res, 
            //     TensorID _op, TensorID _op2) {



            //     ComputationalGraphMap& map = ComputationalGraphMap::get();

            //     auto op = RegisteredOperation(
            //             _typ,
            //             _res->get_tensor_id(),  
            //             _op, 
            //             _op2
            //         );

            //     TensorID tensor_id = map._register_operation(_res, op);
            //     op.result = tensor_id;
            // }


        }

    }

}
