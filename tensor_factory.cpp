
#include "tensor_factory.h"
#include "function_object_factory.h"
#include "m_algorithms_concepts.h"


namespace NeuralNetwork {

    namespace Computation {

        namespace Graph {



            std::shared_ptr<Tensor> TensorConstructor::create(
                Matrix::Rows _l, Matrix::Columns _w, 
                IsTrackable _t, 
                IsLeaf _f,
                IsRecordable _r) {
                
                auto tensor = std::make_shared<Tensor>(
                        _l, _w, _t, _f, _r);
                
                FunctionObjectFactory::create(tensor);
                
                return tensor;
            }


            template <Matrix::Operations::MatrixOperatable Operator>
            std::shared_ptr<Tensor> TensorConstructor::create(
                Operator _operator,
                const Matrix::Representation& _m,
                TensorID _op, 
                TensorID _op2,  
                IsTrackable _t, 
                IsLeaf _f,
                IsRecordable _r) {
                

                auto tensor = std::make_shared<Tensor>(
                        _m, _t, _f, _r);

                if constexpr (Matrix::Operations::UnaryMatrixOperatable<Operator>) {
                    FunctionObjectFactory::create(
                        _operator, tensor, _op);
                }
                else if constexpr (Matrix::Operations::BinaryMatrixOperatable<Operator>) {
                    FunctionObjectFactory::create(
                        _operator, tensor, _op, _op2);
                }
 
                return tensor;
            }


            
            template std::shared_ptr<Tensor> TensorConstructor::create<Matrix::Operations::Unary::ReLU>(
                Matrix::Operations::Unary::ReLU _operator,
                const Matrix::Representation& _m,
                TensorID _op, 
                TensorID _op2,  
                IsTrackable _t, 
                IsLeaf _f,
                IsRecordable _r);

            template std::shared_ptr<Tensor> TensorConstructor::create<Matrix::Operations::Unary::SoftMax>(
                Matrix::Operations::Unary::SoftMax _operator,
                const Matrix::Representation& _m,
                TensorID _op, 
                TensorID _op2,  
                IsTrackable _t, 
                IsLeaf _f,
                IsRecordable _r);

            template std::shared_ptr<Tensor> TensorConstructor::create<Matrix::Operations::Binary::HadamardProduct::Std>(
                Matrix::Operations::Binary::HadamardProduct::Std _operator,
                const Matrix::Representation& _m,
                TensorID _op, 
                TensorID _op2,  
                IsTrackable _t, 
                IsLeaf _f,
                IsRecordable _r);

            template std::shared_ptr<Tensor> TensorConstructor::create<Matrix::Operations::Binary::Multiplication::ParallelDNC>(
                Matrix::Operations::Binary::Multiplication::ParallelDNC _operator,
                const Matrix::Representation& _m,
                TensorID _op, 
                TensorID _op2,  
                IsTrackable _t, 
                IsLeaf _f,
                IsRecordable _r);

            template std::shared_ptr<Tensor> TensorConstructor::create<Matrix::Operations::Binary::Multiplication::Naive>(
                Matrix::Operations::Binary::Multiplication::Naive _operator,
                const Matrix::Representation& _m,
                TensorID _op, 
                TensorID _op2,  
                IsTrackable _t, 
                IsLeaf _f,
                IsRecordable _r);

            template std::shared_ptr<Tensor> TensorConstructor::create<Matrix::Operations::Binary::Multiplication::Square>(
                Matrix::Operations::Binary::Multiplication::Square _operator,
                const Matrix::Representation& _m,
                TensorID _op, 
                TensorID _op2,  
                IsTrackable _t, 
                IsLeaf _f,
                IsRecordable _r);

            template std::shared_ptr<Tensor> TensorConstructor::create<Matrix::Operations::Binary::Addition::Std>(
                Matrix::Operations::Binary::Addition::Std _operator,
                const Matrix::Representation& _m,
                TensorID _op, 
                TensorID _op2,  
                IsTrackable _t, 
                IsLeaf _f,
                IsRecordable _r);

            template std::shared_ptr<Tensor> TensorConstructor::create<Matrix::Operations::Binary::OuterProduct::Naive>(
                Matrix::Operations::Binary::OuterProduct::Naive _operator,
                const Matrix::Representation& _m,
                TensorID _op, 
                TensorID _op2,  
                IsTrackable _t, 
                IsLeaf _f,
                IsRecordable _r);

            template std::shared_ptr<Tensor> TensorConstructor::create<Matrix::Operations::Metric::CrossEntropy>(
                Matrix::Operations::Metric::CrossEntropy _operator,
                const Matrix::Representation& _m,
                TensorID _op, 
                TensorID _op2,  
                IsTrackable _t, 
                IsLeaf _f,
                IsRecordable _r);



        }

    }
}