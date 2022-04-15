#ifndef TENSOR_H
#define TENSOR_H


#include "matrix.h"
#include "m_algorithms.h"
#include "m_algorithms_register.h"

#include <memory>

namespace NeuralNetwork {

    namespace Computation {

        namespace Graph {

            class RegisteredOperation;

            class TensorOp;


            using IsTrackable = Matrix::NamedType<bool, struct TrackParameter>;
            using IsLeaf      = Matrix::NamedType<bool, struct LeafParameter>;


            class Tensor : std::enable_shared_from_this<Tensor> {

                using matrix_t = std::unique_ptr<Matrix::Representation>;

                public:
                    Tensor(Matrix::Rows _l, Matrix::Columns _w, 
                        IsTrackable _t = IsTrackable(true), IsLeaf _f = IsLeaf(true));
                                                            
                    Tensor(std::unique_ptr<Matrix::Representation> _m, 
                        IsTrackable _t = IsTrackable(true), IsLeaf _f = IsLeaf(true));

                    bool is_tensor_leaf() { return is_leaf; }
                    bool is_requires_grad() { return requires_grad; }
                    matrix_t release_matrix() { return std::move(matrix); }
                    matrix_t release_grad() { return std::move(grad); }
                    void register_operation(const std::shared_ptr<RegisteredOperation> _node) { graph_node = _node;}
                    std::shared_ptr<RegisteredOperation> get_operation() { return graph_node; }

                protected:
                    void register_leaf_op(void);
                private:
                    matrix_t matrix;
                    matrix_t grad;
                    std::shared_ptr<RegisteredOperation> graph_node;
                    bool is_leaf;
                    bool requires_grad;


            };

            class TensorOp {

                TensorOp(std::unique_ptr<Matrix::Operations::BaseInterface> _op) : 
                    op_type(std::move(_op)) {}
                public:
                    std::shared_ptr<Tensor> operator()(
                        const std::shared_ptr<Tensor>& l, 
                        const std::shared_ptr<Tensor>& r);
                private:
                    std::unique_ptr<Matrix::Operations::BaseInterface> op_type; 


            };

        }

    }

}


#endif // TENSOR_H