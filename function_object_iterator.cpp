#include "function_object_iterator.h"
#include "tensor.h"
#include "generator.h"




namespace NeuralNetwork {

    namespace Computation {

        namespace Graph {


                void ReadParameterPolicy::process_head(TensorID tid) {}


                void ReadParameterPolicy::apply_to_children(std::stack<TensorID>& nodeStack, TensorID tid) {
                    nodeStack.emplace(tid);
                }


                ReadParameterPolicy::ReturnType ReadParameterPolicy::dereference(TensorID tid) {

                    ComputationalGraphMap& map = ComputationalGraphMap::get();
                    return map._get_tensor(tid)->release_matrix();
                }


                ReadParameterPolicy::Matrix_t ReadParameterPolicy::grad(TensorID tid) {

                    ComputationalGraphMap& map = ComputationalGraphMap::get();
                    auto gradient = Matrix_t{map._get_tensor(tid)->get_grad()};
                    return Matrix_t{gradient};
                }


                void ComputeGradientPolicy::process_head(TensorID tid) {

                    // std::cout << "Backpropigating TID: " << _t.get() << std::endl;

                    ComputationalGraphMap& map = ComputationalGraphMap::get();
                    auto operation = map._get_operation(tid);

                    // auto _r = map._get_tensor(current)->release_matrix().num_rows();
                    // auto _w = map._get_tensor(current)->release_matrix().num_cols();

                    // auto matrix = Matrix_t{
                    //     Matrix::Rows(_r),
                    //     Matrix::Columns(_w)
                    // };
                    // Matrix::Generation::Tester<1> unit_gen;

                    // matrix = unit_gen(matrix);

                    // Events::Differentiate backpropigate_grad(matrix);
                    Events::Differentiate backpropigate_grad(Matrix_t{});
                    
                    operation.stringify_type();
                    std::cout << "Computing Leaf Derivative" << std::endl;
                    operation.process_event(backpropigate_grad);
                    operation.stringify_type();
                            

                 }


                void ComputeGradientPolicy::apply_to_children(std::stack<TensorID>& nodeStack, TensorID tid) {
                    
                    ComputationalGraphMap& map = ComputationalGraphMap::get();
                    auto df = Matrix_t{map._get_tensor(tid)->get_grad()};
                    assert(df.num_rows() && df.num_cols() && "Invalid Derivative.");
                    std::cout << "Gradient DIM: [" << df.num_rows() << "," << df.num_cols() << "]" << std::endl;
                    
                    Events::Differentiate backpropigate_grad(df);
                    auto operation = map._get_operation(tid); 
                    
                    operation.stringify_type();
                    std::cout << "Processing event:" << std::endl;
                    operation.process_event(backpropigate_grad);
                    operation.stringify_type();
                    nodeStack.emplace(tid);
                }


                ComputeGradientPolicy::ReturnType ComputeGradientPolicy::dereference(TensorID tid) {

                    ComputationalGraphMap& map = ComputationalGraphMap::get();
                    FunctionObject fn_obj = map._get_operation(tid);
                    return fn_obj;
                }

                template <TraversalPolicy TP>  
                LevelOrderIterator<TP>::LevelOrderIterator(const TensorID _t) noexcept : current(_t) {
                        
                        if (_t.get()) {

                            TP::process_head(_t);
                            this->_stack_children();   
                        }

                    }


                template <TraversalPolicy TP>  
                LevelOrderIterator<TP>& LevelOrderIterator<TP>::operator++(void) noexcept {

                    if (!nodeStack.empty()) {
                        
                        current = nodeStack.top();
                        nodeStack.pop();

                        this->_stack_children();
                    }
                    else current = TensorID(0);

            
                    return *this;
                }


                template <TraversalPolicy TP>  
                typename LevelOrderIterator<TP>::IterReturnType LevelOrderIterator<TP>::operator*() const noexcept {
                        
                    return TP::dereference(current);
                }


                template <TraversalPolicy TP>  
                void LevelOrderIterator<TP>::_stack_children(void) noexcept {

                        ComputationalGraphMap& map = ComputationalGraphMap::get();
                        FunctionObject fn_obj = map._get_operation(current);

                        fn_obj.stringify_type();

                        for (std::size_t i = 0; const auto tid: fn_obj.serialize()) {

                            if (tid) {
                                std::cout << "Backpropigating TID " << i << ": " << tid->get() << std::endl;
                                

                                if (i++) {
                                    
                                    TP::apply_to_children(nodeStack, TensorID(tid->get()));

                                }
                            }

                        } 

                        
                    return;
                }

                template class LevelOrderIterator<ReadParameterPolicy>;
                template class LevelOrderIterator<ComputeGradientPolicy>;
                
                
                
            } // Graph

        } // Computation

    } // NN