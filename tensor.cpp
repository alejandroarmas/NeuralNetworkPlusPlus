#include "tensor.h"

#include "matrix.h"
#include "generator.h"
#include "m_algorithms.h"
#include "m_algorithms_register.h"

#include <memory>


namespace NeuralNetwork {

    namespace Computation {


        namespace Graph {


            Tensor::Tensor(Matrix::Rows _l, Matrix::Columns _w, 
                IsTrackable _t, IsLeaf _f): matrix(std::make_unique<Matrix::Representation>(_l, _w)), 
                grad(nullptr), graph_node(nullptr), is_leaf(_f.get()), requires_grad(_t.get()) {

                    Matrix::Generation::Normal<0, 1> normal_distribution_init;
                    matrix = normal_distribution_init(std::move(matrix));

                if (is_leaf) this->register_leaf_op(); 
                        
                    
            }


            Tensor::Tensor(std::unique_ptr<Matrix::Representation> _m, 
                IsTrackable _t, IsLeaf _f) : 
                matrix(std::move(_m)), grad(nullptr), 
                graph_node(nullptr), is_leaf(_f.get()), 
                requires_grad(_t.get()) {
                    
                if (is_leaf) this->register_leaf_op(); 
            }


            void Tensor::register_leaf_op(void) {
                        
                auto op = RegisteredOperation::create(
                    Matrix::Operations::Code::NOP, 
                    std::shared_ptr<Tensor>(this)
                    );
                register_operation(op);
            } 


            std::shared_ptr<Tensor> TensorOp::operator()(
                const std::shared_ptr<Tensor> l, 
                const std::shared_ptr<Tensor> r) {

                    bool isBinaryOp = r != nullptr; 

                    std::shared_ptr<Tensor> out_tensor;
                    std::shared_ptr<RegisteredOperation> out_op;

                    auto op_code = op_type->get_operation_code(); 

                    if (isBinaryOp) {
                        
                        std::unique_ptr<Matrix::Representation> out_matrix = 
                            op_type->operator()(
                                std::move(l->release_matrix()), 
                                std::move(r->release_matrix())
                            );

                        out_tensor = std::make_shared<Tensor>(
                                std::move(out_matrix), IsTrackable(true), IsLeaf(false));
                        
                        out_op =
                            RegisteredOperation::create(op_code, 
                                    out_tensor, l->get_operation(),
                                    r->get_operation()
                            );

                    }
                    else {
                        auto out_matrix = op_type->operator()(
                                std::move(l->release_matrix())
                            );

                        out_tensor = std::make_shared<Tensor>
                            (std::move(out_matrix), IsTrackable(true), IsLeaf(false));
                        
                        out_op = 
                            RegisteredOperation::create(op_code, out_tensor, 
                                l->get_operation()
                            );

                    }


                    out_tensor->register_operation(out_op);

                    return out_tensor;
                }

                

        }

    }

}