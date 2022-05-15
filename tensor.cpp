#include "tensor.h"
#include "matrix.h"
#include "generator.h"
#include "m_algorithms_register.h"
#include "tensor_backwards_pass.h"

#include "m_algorithms_utilities.h"
#include <variant>

#include <memory>


namespace NeuralNetwork {

    namespace Computation {


        namespace Graph {

            Tensor::Tensor(Matrix::Rows _l, Matrix::Columns _w, 
                    IsTrackable _t, IsLeaf _f, IsRecordable _r): 
                    stats({}),
                    map(ComputationalGraphMap::get()),
                    matrix(Matrix::Representation(_l, _w)), 
                    grad({}), 
                    my_tensor_id(ComputationalGraphMap::get()._obtain_tensor_id()),  
                    is_leaf(_f.get()),
                    requires_grad(_t.get()), record_statistics(_r.get()) {

                Matrix::Generation::Normal<0, 1> normal_distribution_init;                    
                matrix = normal_distribution_init(matrix);
                                            
            }


            Tensor::Tensor(const Matrix::Representation& _m, 
                    IsTrackable _t, IsLeaf _f, IsRecordable _r) : 
                    stats({}),
                    map(ComputationalGraphMap::get()),
                    matrix(_m), grad({}), 
                    my_tensor_id(ComputationalGraphMap::get()._obtain_tensor_id()),  
                    is_leaf(_f.get()), 
                    requires_grad(_t.get()), record_statistics(_r.get()) {

            }


            void Tensor::detatch_from_computational_graph() {
                // Matrix::Operations::Utility::Stringify stringify;
                // auto fn = Matrix::Operations::Utility::Function::from(get_operation().get_code());

                // std::cout << "Freeing " << this << " Registry: O[" << my_tensor_id.get() << "] = " << std::visit(stringify, fn) << std::endl;
                
                map._recover_tensor_id(my_tensor_id);
            }


            FunctionObject Tensor::get_operation() { 
                return map._get_operation(my_tensor_id);
            }


            bool Tensor::is_tensor_leaf() const {   
                return is_leaf;  
            }

            void Tensor::become_parent() {   
                is_leaf = false; 
            }

            bool Tensor::is_requires_grad() const {   
                return requires_grad; 
            }

            bool Tensor::is_recorded() const {   
                return record_statistics; 
            }


            Tensor::matrix_t Tensor::release_matrix() {   
                return matrix; 
            }

            Tensor::matrix_t Tensor::get_grad(){   
                return grad.value_or(Matrix::Representation()); 
            }
            
            Matrix::Rows Tensor::num_rows(void) const {
                return Matrix::Rows(matrix.num_rows());
            }

            Matrix::Columns Tensor::num_cols(void) const {
                return Matrix::Columns(matrix.num_cols());
            }

            void Tensor::backwards() {

                ReversePass reverse;

                reverse.backwards(*this, PrintTag{});

            }


        }

    }

}