#include "function_object_factory.h"
#include "function_object.h"
#include "computational_graph_map.h"
#include "tensor.h"

#include <optional>



namespace NeuralNetwork {

    namespace Computation {

        namespace Graph {


            // std::optional<FunctionObject> DereferenceRegistry::from(RegisteredOperation _op) {
            //     if (_op.get_tensor_id() == TensorID(0)) return {};

            //     ComputationalGraphMap& map = ComputationalGraphMap::get();
            //     return map._get_tensor(_op.get_tensor_id());
                
            // }


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

    
            /*
                Creates a Function Object signifying NOP expression.
            */
            FunctionObject FunctionObjectFactory::create(T _res) {

                ComputationalGraphMap& map = ComputationalGraphMap::get();

                auto tensor_identification = _res->get_tensor_id();

                auto fn_object = FunctionObject(
                            RegisteredTensor(tensor_identification)
                        );

                map._register_operation(_res, fn_object);

                return fn_object;
            }



            /*
                1) access _res NOP
                2) create Instantiate based on operation
                3) fill step 2 with TensorID of operands
                4) 

            */

            template <Matrix::Operations::BinaryMatrixOperatable RegisteryType>
            FunctionObject FunctionObjectFactory::create(
                RegisteryType operation, T _res, 
                TensorID _operand_id, TensorID _operand_id_two) {

                ComputationalGraphMap& map = ComputationalGraphMap::get();

                auto res_tensor_id = _res->get_tensor_id();
                
                auto fn_object = FunctionObject();

                auto binaryRegistry = States::BinaryRegistered(
                            RegisteredBinaryOperation(res_tensor_id, 
                                _operand_id, _operand_id_two)
                        );

                auto instantiate_event = Events::Instantiate(operation, binaryRegistry);

                
                fn_object.process_event(instantiate_event);
                fn_object.stringify_type();

                // transition _res default NOP state to unary-state 
                // update computational graph OP state 

                
                map._register_operation(_res, fn_object);
                
                return fn_object;
            }


            template <Matrix::Operations::UnaryMatrixOperatable RegisteryType>
            FunctionObject FunctionObjectFactory::create(
                RegisteryType operation, T _res, TensorID _operand_id) {

                ComputationalGraphMap& map = ComputationalGraphMap::get();

                auto res_tensor_id = _res->get_tensor_id();

                auto fn_object = FunctionObject();

                auto unaryRegistry = States::UnaryRegistered(
                            RegisteredUnaryOperation(res_tensor_id, _operand_id)
                        );

                auto instantiate_event = Events::Instantiate(operation, unaryRegistry);

                fn_object.process_event(instantiate_event);
                fn_object.stringify_type();

                // transition _res default NOP state to unary-state 
                // update computational ~graph OP state 

                map._register_operation(_res, fn_object);


                return fn_object;
            }


            template FunctionObject FunctionObjectFactory::create<Matrix::Operations::Unary::ReLU>(
                Matrix::Operations::Unary::ReLU operation,
                T _res, 
                TensorID _operand_id);

            template FunctionObject FunctionObjectFactory::create<Matrix::Operations::Unary::SoftMax>(
                Matrix::Operations::Unary::SoftMax operation,
                T _res, 
                TensorID _operand_id);

            template FunctionObject FunctionObjectFactory::create<Matrix::Operations::Binary::HadamardProduct::Std>(
                Matrix::Operations::Binary::HadamardProduct::Std operation,
                T _res, 
                TensorID _operand_id, 
                TensorID _operand_id_two);

            template FunctionObject FunctionObjectFactory::create<Matrix::Operations::Binary::Multiplication::ParallelDNC>(
                Matrix::Operations::Binary::Multiplication::ParallelDNC operation,
                T _res, 
                TensorID _operand_id, 
                TensorID _operand_id_two);

            template FunctionObject FunctionObjectFactory::create<Matrix::Operations::Binary::Multiplication::Naive>(
                Matrix::Operations::Binary::Multiplication::Naive operation,
                T _res, 
                TensorID _operand_id, 
                TensorID _operand_id_two);

            template FunctionObject FunctionObjectFactory::create<Matrix::Operations::Binary::Multiplication::Square>(
                Matrix::Operations::Binary::Multiplication::Square operation,
                T _res, 
                TensorID _operand_id, 
                TensorID _operand_id_two);

            template FunctionObject FunctionObjectFactory::create<Matrix::Operations::Binary::Addition::Std>(
                Matrix::Operations::Binary::Addition::Std operation,
                T _res, 
                TensorID _operand_id, 
                TensorID _operand_id_two);

            template FunctionObject FunctionObjectFactory::create<Matrix::Operations::Binary::OuterProduct::Naive>(
                Matrix::Operations::Binary::OuterProduct::Naive operation,
                T _res, 
                TensorID _operand_id, 
                TensorID _operand_id_two);

            template FunctionObject FunctionObjectFactory::create<Matrix::Operations::Metric::CrossEntropy>(
                Matrix::Operations::Metric::CrossEntropy operation,
                T _res, 
                TensorID _operand_id, 
                TensorID _operand_id_two);


        }
    }
}