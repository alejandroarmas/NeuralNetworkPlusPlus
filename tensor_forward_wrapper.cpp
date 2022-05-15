
#include "tensor_forward_wrapper.h"
#include "tensor.h"
#include "tensor_factory.h"

#include "matrix.h"
#include "generator.h"
#include "m_algorithms.h"
#include "m_algorithms_register.h"
#include "m_algorithms_utilities.h"

#include "strong_types.h"

#include <chrono>
#include <concepts>
#include <memory>
#include <variant>




namespace NeuralNetwork {

    namespace Computation {

        namespace Graph {


            template <Matrix::Operations::MatrixOperatable Operator>
            std::shared_ptr<Tensor> TensorOp<Operator>::operator()(
                const std::shared_ptr<Tensor> l, 
                const std::shared_ptr<Tensor> r) {

                    bool recordTensorOperation = l->is_recorded() || r->is_recorded();
                    
                    PerformTensorStrategy implementation;

                    if (recordTensorOperation) {
                        
                        PerformTensorStrategy::RecordTag _;

                        return _.compute_tensor(op_type, l, r, implementation);
                    }
                    
                    PerformTensorStrategy::ComputeTag _;

                    return _.compute_tensor(op_type, l, r, implementation);
                
                }

            /*
                templates explicit instantiation
            */

            template class TensorOp<Matrix::Operations::Unary::ReLU>;
            template class TensorOp<Matrix::Operations::Unary::SoftMax>;
            template class TensorOp<Matrix::Operations::Binary::HadamardProduct::Std>;
            template class TensorOp<Matrix::Operations::Binary::Multiplication::ParallelDNC>;
            template class TensorOp<Matrix::Operations::Binary::Multiplication::Naive>;
            template class TensorOp<Matrix::Operations::Binary::Multiplication::Square>;
            template class TensorOp<Matrix::Operations::Binary::Addition::Std>;
            template class TensorOp<Matrix::Operations::Binary::OuterProduct::Naive>;
            template class TensorOp<Matrix::Operations::Metric::CrossEntropy>;



                template <Matrix::Operations::MatrixOperatable Operator>
                std::shared_ptr<Tensor> PerformTensorStrategy::compute(
                    Operator _op,
                    const std::shared_ptr<Tensor> l, 
                    const std::shared_ptr<Tensor> r, 
                    ComputeTag _ ) {

                        
                    Matrix::Representation out_matrix;
                    std::shared_ptr<Tensor> out_tensor;                    


                    if constexpr (Matrix::Operations::UnaryMatrixOperatable<Operator>) {
                        out_matrix = _op(
                            l->release_matrix());
                    }
                    else if constexpr (Matrix::Operations::BinaryMatrixOperatable<Operator>) {
                        out_matrix = _op(
                            l->release_matrix(),
                            r->release_matrix()
                        );
                    }

                    if constexpr (Matrix::Operations::UnaryMatrixOperatable<Operator>) {

                        out_tensor = TensorConstructor::create(_op,
                            std::move(out_matrix),  
                            l->get_tensor_id(),
                            TensorID(0),
                            IsTrackable(true),
                            IsLeaf(true), 
                            IsRecordable(false));
                        l->become_parent();
                    }
                    else if constexpr (Matrix::Operations::BinaryMatrixOperatable<Operator>) {
                        
                        out_tensor = TensorConstructor::create(_op,
                            std::move(out_matrix),  
                            l->get_tensor_id(),
                            r->get_tensor_id(),
                            IsTrackable(true),
                            IsLeaf(true), 
                            IsRecordable(false));
                        
                        l->become_parent();
                        r->become_parent();
                        
                       }


                    return out_tensor;

                    }

                

                template <Matrix::Operations::MatrixOperatable Operator>
                std::shared_ptr<Tensor> PerformTensorStrategy::compute(
                    Operator _op,
                    const std::shared_ptr<Tensor> l, 
                    const std::shared_ptr<Tensor> r, 
                    RecordTag _) {

                    TensorStatistics _s;
                    _s.set_graph_start(std::chrono::steady_clock::now());
                    
                    Matrix::Operations::Utility::Stringify stringify;
                    _s.set_operation_string(stringify(_op));

                    Matrix::Representation out_matrix;
                    std::shared_ptr<Tensor> out_tensor;
                    

                    _s.set_matrix_start(std::chrono::steady_clock::now());

                    if constexpr (Matrix::Operations::UnaryMatrixOperatable<Operator>) {
                        out_matrix = _op(
                                    l->release_matrix());
                    }
                    else if constexpr (Matrix::Operations::BinaryMatrixOperatable<Operator>) {
                        out_matrix = _op(
                                    l->release_matrix(),
                                    r->release_matrix()
                                );
                    }

                    _s.set_matrix_end(std::chrono::steady_clock::now());
                                            
                        
                    if constexpr (Matrix::Operations::UnaryMatrixOperatable<Operator>) {
                    
                        out_tensor = TensorConstructor::create(_op,
                                std::move(out_matrix),  
                                l->get_tensor_id(),
                                TensorID(0),
                                IsTrackable(true),
                                IsLeaf(true), 
                                IsRecordable(true));
                        l->become_parent();

                    }
                    else if constexpr (Matrix::Operations::BinaryMatrixOperatable<Operator>) {
                        
                        out_tensor = TensorConstructor::create(_op,
                                std::move(out_matrix),  
                                l->get_tensor_id(),
                                r->get_tensor_id(),
                                IsTrackable(true),
                                IsLeaf(true), 
                                IsRecordable(true));
                        l->become_parent();
                        r->become_parent();
                    }

                    _s.set_graph_end(std::chrono::steady_clock::now());
                    out_tensor->stats = _s;

                    
                    return out_tensor;

                    }


        }

    }

}