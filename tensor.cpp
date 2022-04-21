#include "tensor.h"

#include "matrix.h"
#include "generator.h"
#include "m_algorithms.h"
#include "m_algorithms_register.h"
#include "m_algorithms_utilities.h"

#include <chrono>
#include <concepts>
#include <memory>
#include <variant>


namespace NeuralNetwork {

    namespace Computation {


        namespace Graph {


            Tensor::Tensor(Matrix::Rows _l, Matrix::Columns _w, 
                IsTrackable _t, IsLeaf _f, IsRecordable _r): 
                stats({}),
                matrix(Matrix::Representation(_l, _w)), 
                grad({}), graph_node(nullptr), is_leaf(_f.get()),
                requires_grad(_t.get()), record_statistics(_r.get()) {

                    Matrix::Generation::Normal<0, 1> normal_distribution_init;
                    matrix = normal_distribution_init(matrix);

                if (is_leaf) this->register_leaf_op(); 
                        
                    
            }


            Tensor::Tensor(const Matrix::Representation& _m, 
                IsTrackable _t, IsLeaf _f, IsRecordable _r) : 
                stats({}),
                matrix(_m), grad({}), 
                graph_node(nullptr), is_leaf(_f.get()), 
                requires_grad(_t.get()), record_statistics(_r.get()){
                    
                if (is_leaf) this->register_leaf_op(); 
            }


            void Tensor::register_leaf_op(void) {
                        
                auto op = RegisteredOperation::create(
                    Matrix::Operations::Code::NOP, 
                    std::shared_ptr<Tensor>(this)
                    );
                register_operation(op);
            } 



            template <Matrix::Operations::MatrixOperatable Operator>
            std::shared_ptr<Tensor> TensorOp<Operator>::operator()(
                const std::shared_ptr<Tensor> l, 
                const std::shared_ptr<Tensor> r) {

                    bool recordTensorOperation = l->is_recorded() || r->is_recorded();
                    
                    PerformTensorStrategy implementation;

                    if (recordTensorOperation) {
                        
                        RecordTag _;

                        return _.compute_tensor(op_type, l, r, implementation);
                    }
                    
                    ComputeTag _;

                    return _.compute_tensor(op_type, l, nullptr, implementation);
                
                }


            template class TensorOp<Matrix::Operations::Unary::ReLU>;
            template class TensorOp<Matrix::Operations::Binary::HadamardProduct::Std>;
            template class TensorOp<Matrix::Operations::Binary::Multiplication::ParallelDNC>;
            template class TensorOp<Matrix::Operations::Binary::Multiplication::Naive>;
            template class TensorOp<Matrix::Operations::Binary::Multiplication::Square>;
            template class TensorOp<Matrix::Operations::Binary::Addition::Std>;
            template class TensorOp<Matrix::Operations::Binary::OuterProduct::Naive>;



                template <Matrix::Operations::MatrixOperatable Operator>
                std::shared_ptr<Tensor> PerformTensorStrategy::compute(
                    Operator _op,
                    const std::shared_ptr<Tensor> l, 
                    const std::shared_ptr<Tensor> r, 
                    ComputeTag _ ) {

                        
                    Matrix::Representation out_matrix;
                    std::shared_ptr<Tensor> out_tensor;
                    std::shared_ptr<RegisteredOperation> out_op;

                    Matrix::Operations::Utility::Codify codify;
                    auto op_code = codify(_op); 


                    if constexpr (Matrix::Operations::UnaryMatrixOperatable<Operator>) {
                        out_matrix = _op(l->release_matrix());
                    }
                    else if constexpr (Matrix::Operations::BinaryMatrixOperatable<Operator>) {
                        out_matrix = _op(
                                    l->release_matrix(),
                                    r->release_matrix()
                                );
                    }


                    out_tensor = std::make_shared<Tensor>
                        (std::move(out_matrix), IsTrackable(true), 
                        IsLeaf(false), IsRecordable(false));
                    

                    if constexpr (Matrix::Operations::UnaryMatrixOperatable<Operator>) {
                        out_op = RegisteredOperation::create(op_code, 
                                    out_tensor, 
                                    l->get_operation()
                            );
                    }
                    else if constexpr (Matrix::Operations::BinaryMatrixOperatable<Operator>) {
                        out_op = RegisteredOperation::create(op_code, 
                                    out_tensor, 
                                    l->get_operation(),
                                    r->get_operation()
                            );
                    }

                    out_tensor->register_operation(out_op);

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

                    Matrix::Operations::Utility::Codify codify;
                    Matrix::Operations::Code op_code = codify(_op); 

                    Matrix::Representation out_matrix;
                    std::shared_ptr<Tensor> out_tensor;
                    std::shared_ptr<RegisteredOperation> out_op;


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
                        
                    
                    out_tensor = std::make_shared<Tensor>(
                            std::move(out_matrix), IsTrackable(true), 
                            IsLeaf(false), IsRecordable(true));
                    
                        
                    if constexpr (Matrix::Operations::UnaryMatrixOperatable<Operator>) {
                        out_op = RegisteredOperation::create(op_code, 
                                    out_tensor, 
                                    l->get_operation()
                            );
                    }
                    else if constexpr (Matrix::Operations::BinaryMatrixOperatable<Operator>) {
                        out_op = RegisteredOperation::create(op_code, 
                                    out_tensor, 
                                    l->get_operation(),
                                    r->get_operation()
                            );
                    }


                    out_tensor->register_operation(out_op);

                    _s.set_graph_end(std::chrono::steady_clock::now());
                    out_tensor->stats = _s;

                    
                    return out_tensor;

                    }
            

        }

    }

}