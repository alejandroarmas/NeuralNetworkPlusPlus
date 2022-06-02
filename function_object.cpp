#include "computational_graph_map.h"
#include "function_object.h"
#include "tensor.h"
#include "matrix.h"
#include "m_algorithms.h"

namespace NeuralNetwork {

    namespace Computation {

        namespace Graph {

            /*
            
                DESCRIPTION:
                    Cross-entropy loss function for the 
                    softmax function.

                    Suppose y is a row vector in R^n*1

                    then dJ/do = yˆ - y 
                    
                    Suppose y is a column vector in R^1*n
                        dJ/do = (yˆ − y)T

            */
            OperationTransitioner::State OperationTransitioner::operator()(
                States::CrossEntropy ce, Events::Differentiate& df) noexcept {
                
                auto ltid = ce.left_op_id();
                auto rtid = ce.right_op_id();
                

                ComputationalGraphMap& map = ComputationalGraphMap::get();
                
                auto right_op = map._get_tensor(rtid);

                auto left_matrix  = Matrix::Representation{map._get_tensor(ltid)->release_matrix()};
                auto right_matrix = Matrix::Representation{right_op->release_matrix()};

                Matrix::Operations::Binary::Subtraction::Std subtract;
                Matrix::Operations::Unary::SoftMax softmax;
                Matrix::Operations::Unary::Transpose transpose;

                df.gradient = subtract(softmax(right_matrix), left_matrix);
                
                if (right_matrix.get_type() == Matrix::Representation::Type::COLUMN_VECTOR) {
                    df.gradient = transpose(df.gradient);
                }

                right_op->get_grad() = df.gradient;
                
                return States::Invalidated{};
            }


            /*
                DESCRIPTION:

                    Compute the gradients of each parent 
                    operand for each respective tensor. 

                    Recall the chain rule applied to 
                    vectorized gradients.
                        dJ/dx = dj/dz * dz/dx


            Three cases:

                Suppose incoming gradient dj/dz.

                1)
                    z = Wx, where z and x are vectors in R^n,
                    and W is a matrix in R^m*n.
                    
                    dj/dx = dj/dz * W
                    dj/dW = outerproduct{(dj/dz)^T, x^T}    

                
                2)    
                    z = xW, where z and x are vectors in R^m 
                    and R^n respectively, and W is a matrix 
                    in R^m*n.

                    dj/dx = dj/dz * W^T
                    dj/dW = outerproduct{x^T, dj/dz}    



                FIXME: cyclical computation graph could result in overwriting gradient
            */
            OperationTransitioner::State OperationTransitioner::operator()(States::MatrixMultiply mm, Events::Differentiate& df) noexcept {
                
                // auto tid = mm.get_tensor_id();
                auto ltid = mm.left_op_id();
                auto rtid = mm.right_op_id();
                
                ComputationalGraphMap& map = ComputationalGraphMap::get();
                
                auto left_op = map._get_tensor(ltid);
                auto right_op = map._get_tensor(rtid);


                auto left_matrix  = Matrix::Representation{left_op->release_matrix()};
                auto right_matrix = Matrix::Representation{right_op->release_matrix()};

                bool row_times_matrix = left_matrix.get_type() == Matrix::Representation::Type::ROW_VECTOR && 
                    right_matrix.get_type() == Matrix::Representation::Type::MATRIX;

                bool matrix_times_col = left_matrix.get_type() == Matrix::Representation::Type::MATRIX && 
                    right_matrix.get_type() == Matrix::Representation::Type::COLUMN_VECTOR;

                assert(row_times_matrix || matrix_times_col && "Matrix Multiply was invalid.");

                Matrix::Operations::Unary::Transpose transpose;
                Matrix::Operations::Binary::Multiplication::ParallelDNC mult;
                Matrix::Operations::Binary::OuterProduct::Naive outer;

                std::cout << "Matrix Multiply Derivatives." << std::endl;

                if (row_times_matrix) {

                    std::cout << "row_times_matrix" << std::endl;
                    auto xT   = transpose(left_matrix);                   
                    auto djdW = outer(xT, df.gradient);

                    auto wT   = transpose(right_matrix);
                    auto djdx = mult(df.gradient, wT);
                    
                    left_op->get_grad()  = djdx;
                    right_op->get_grad() = djdW;
                }
                else if (matrix_times_col) {
                    std::cout << "matrix_times_col" << std::endl;

                    auto djdx = mult(df.gradient, left_matrix);

                    auto xT   = transpose(right_matrix);
                    auto dfT  = transpose(df.gradient);
                    auto djdW = outer(dfT, xT);
                    
                    
                    left_op->get_grad()  = djdW;
                    right_op->get_grad() = djdx;

                }
                else assert (row_times_matrix || matrix_times_col && "Invalid Operands.");

                return States::Invalidated{};
            }

            OperationTransitioner::State OperationTransitioner::operator()(States::Plus add, Events::Differentiate& df) noexcept {
                
                auto ltid = add.left_op_id();
                auto rtid = add.right_op_id();

                ComputationalGraphMap& map = ComputationalGraphMap::get();

                auto left_op = map._get_tensor(ltid);
                auto right_op = map._get_tensor(rtid);


                auto left_matrix  = Matrix::Representation{left_op->release_matrix()};
                auto right_matrix = Matrix::Representation{right_op->release_matrix()};


                left_op->get_grad() = left_matrix;
                right_op->get_grad() = right_matrix;
                
                return States::Invalidated{};

            }

            OperationTransitioner::State OperationTransitioner::operator()(States::ReLU relu, Events::Differentiate& df) noexcept {
                
                std::cout << "Relu backpropigate" << std::endl;

                auto ltid = relu.left_op_id();

                ComputationalGraphMap& map = ComputationalGraphMap::get();

                auto left_op = map._get_tensor(ltid);


                auto left_matrix  = Matrix::Representation{left_op->release_matrix()};

                Matrix::Operations::Unary::Sign sign;
                Matrix::Operations::Binary::HadamardProduct::Std hadamard;

                left_op->get_grad() = hadamard(sign(left_matrix), df.gradient);
                
                return States::Invalidated{};

            }

            OperationTransitioner::State OperationTransitioner::operator()(const States::NoOperation& nop, Events::Differentiate&) noexcept {
                return nop;
            }

        }
    }
}