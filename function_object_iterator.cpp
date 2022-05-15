
#include "function_object_iterator.h"




namespace NeuralNetwork {

    namespace Computation {

        namespace Graph {

                LevelOrderIterator& LevelOrderIterator::operator++(int) noexcept {

                    if (!nodeStack.empty()) {
                        

                        current = nodeStack.top();
                        nodeStack.pop();

                        this->_stack_children();
                    }
                    else current = TensorID(0);

            
                    return *this;
                }


                FunctionObject LevelOrderIterator::operator*() const noexcept {
                        
                    ComputationalGraphMap& map = ComputationalGraphMap::get();
                    FunctionObject fn_obj = map._get_operation(current);
                    return fn_obj;
                }


                void LevelOrderIterator::_stack_children(void) noexcept {

                        ComputationalGraphMap& map = ComputationalGraphMap::get();
                        FunctionObject fn_obj = map._get_operation(current);

                        fn_obj.stringify_type();

                        for (std::size_t i = 0; const auto tid: fn_obj.serialize()) {

                            if (tid) {
                                std::cout << "tid" << i << ": " << tid->get() << std::endl;
                                
                                if (i++) {
                                    nodeStack.emplace(tid->get());
                                }
                            }

                        } 

                        
                    return;
                }
                
                
            } // Graph

        } // Computation

    } // NN