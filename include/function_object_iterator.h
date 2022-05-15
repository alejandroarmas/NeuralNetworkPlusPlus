#ifndef FUNCTION_OBJECT_ITERATOR_H
#define FUNCTION_OBJECT_ITERATOR_H


#include "computational_graph_map.h"
#include "function_object.h"
#include "strong_types.h"

#include <stack>
#include <optional>



namespace NeuralNetwork {

    namespace Computation {

        namespace Graph {

            /*
                Breadth First Search through computational graph.
            */
            class LevelOrderIterator {

                public:
                    LevelOrderIterator(TensorID _t) noexcept : current(_t) {
                        if (_t.get()) this->_stack_children();
                    }
                    LevelOrderIterator(LevelOrderIterator&) = default; 
                    LevelOrderIterator(LevelOrderIterator&&) = default; 
                    LevelOrderIterator& operator=(const LevelOrderIterator&) = default; 
                    LevelOrderIterator& operator=(LevelOrderIterator&&) = default; 
                    LevelOrderIterator& operator++(int) noexcept;
                    
                    FunctionObject operator*() const noexcept;

                    bool operator!=(const LevelOrderIterator& other) const noexcept{
                        return current != other.current; 
                    }
                
                private:
                    void _stack_children() noexcept;

                    TensorID current;
                    std::stack<TensorID> nodeStack;

                


            };

        }

    }

}


#endif // FUNCTION_OBJECT_ITERATOR_H