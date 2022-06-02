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
                    IsTrackable _t, IsLeaf _f, IsRecordable _r) noexcept: 
                    stats({}),
                    matrix(Matrix::Representation(_l, _w)), 
                    grad(Matrix::Representation(_l, _w)), 
                    my_tensor_id(ComputationalGraphMap::get()._obtain_tensor_id()),  
                    is_leaf(_f.get()),
                    requires_grad(_t.get()), record_statistics(_r.get()) {

                Matrix::Generation::Normal<0, 1> normal_distribution_init;                    
                matrix = normal_distribution_init(matrix);

                Matrix::Generation::Tester<1> unit_gen;
                grad = unit_gen(grad);

                                            
            }


            Tensor::Tensor(const Matrix::Representation& _m, 
                    IsTrackable _t, IsLeaf _f, IsRecordable _r) noexcept: 
                    stats({}),
                    matrix(_m), 
                    grad(Matrix::Representation(
                        Matrix::Rows(_m.num_rows()), 
                        Matrix::Columns(_m.num_cols()))), 
                    my_tensor_id(ComputationalGraphMap::get()._obtain_tensor_id()),  
                    is_leaf(_f.get()), 
                    requires_grad(_t.get()), record_statistics(_r.get()) {

                Matrix::Generation::Tester<1> unit_gen;
                grad = unit_gen(grad);
            }

            Tensor::Tensor(const Tensor& other) noexcept: 
                    // stats(other.stats),
                    matrix(other.matrix), 
                    grad(other.grad), 
                    my_tensor_id(other.my_tensor_id),  
                    is_leaf(other.is_leaf),
                    requires_grad(other.requires_grad), 
                    record_statistics(other.record_statistics) {}


            Tensor& Tensor::operator=(const Tensor& other) noexcept {
                my_tensor_id  = other.my_tensor_id;  
                is_leaf       = other.is_leaf; 
                requires_grad = other.requires_grad;
                // stats = other.stats;
                matrix        = other.matrix; 
                grad          = other.grad; 
                return *this;
            }


            void Tensor::detatch_from_computational_graph() noexcept {
                ComputationalGraphMap::get()._recover_tensor_id(my_tensor_id);
            }


            FunctionObject Tensor::get_operation() noexcept { 
                return ComputationalGraphMap::get()._get_operation(my_tensor_id);
            }


            bool Tensor::is_tensor_leaf() const noexcept {   
                return is_leaf;  
            }

            void Tensor::become_parent() noexcept {   
                is_leaf = false; 
            }

            bool Tensor::is_requires_grad() const noexcept {   
                return requires_grad; 
            }

            bool Tensor::is_recorded() const noexcept {   
                return record_statistics; 
            }


            Tensor::matrix_t& Tensor::release_matrix() noexcept {   
                return matrix; 
            }

            Tensor::matrix_t& Tensor::get_grad() noexcept {   
                return grad; 
            }
            
            Matrix::Rows Tensor::num_rows(void) const noexcept {
                return Matrix::Rows(matrix.num_rows());
            }

            Matrix::Columns Tensor::num_cols(void) const noexcept {
                return Matrix::Columns(matrix.num_cols());
            }

            void Tensor::backwards() noexcept {

                ReversePass reverse;

                reverse.backwards(*this, PrintTag{});

            }


        }

    }

}