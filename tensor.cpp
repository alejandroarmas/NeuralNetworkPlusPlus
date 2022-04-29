#include "tensor.h"
#include "matrix.h"
#include "generator.h"
#include "m_algorithms_register.h"

#include <memory>


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


        }

    }

}