#include "tensor.h"
#include "matrix.h"
#include "generator.h"
#include "m_algorithms_register.h"
#include "tensor_backwards_pass.h"

#include <memory>


namespace NeuralNetwork {

    namespace Computation {


        namespace Graph {


            Tensor::Tensor(Matrix::Rows _l, Matrix::Columns _w, 
                IsTrackable _t, IsLeaf _f, IsRecordable _r): 
                stats({}),
                matrix(Matrix::Representation(_l, _w)), 
                grad({}), 
                graph_node(
                    OperationFactory::create(
                        Matrix::Operations::Code::NOP, *this)), 
                is_leaf(_f.get()),
                requires_grad(_t.get()), record_statistics(_r.get()) {

                    Matrix::Generation::Normal<0, 1> normal_distribution_init;
                    matrix = normal_distribution_init(matrix);
                        
                    
            }


            Tensor::Tensor(const Matrix::Representation& _m, 
                IsTrackable _t, IsLeaf _f, IsRecordable _r) : 
                stats({}),
                matrix(_m), grad({}), 
                graph_node(
                    OperationFactory::create(
                        Matrix::Operations::Code::NOP, *this)), 
                is_leaf(_f.get()), 
                requires_grad(_t.get()), record_statistics(_r.get()) {}


            void Tensor::backwards() {

                ReversePass reverse;

                reverse.backwards(*this, PrintTag{});

            }


        }

    }

}