#ifndef FUNCTION_OBJECT_ITERATOR_H
#define FUNCTION_OBJECT_ITERATOR_H


#include "computational_graph_map.h"
#include "function_object.h"
#include "strong_types.h"
#include "m_algorithms_concepts.h"

#include <stack>
#include <optional>



namespace NeuralNetwork {

    namespace Computation {

        namespace Graph {

            struct BackPropigationMatrixTrait {
                using Type = Matrix::Representation;
            };


            class ComputeGradientPolicy {

                public:

                    using Matrix_t   = BackPropigationMatrixTrait::Type;
                    using ReturnType = FunctionObject;

                    static void process_head(TensorID tid);
                    static void apply_to_children(std::stack<TensorID>& tid_stack, TensorID tid);
                    static ReturnType dereference(TensorID tid);
           };

            class ReadParameterPolicy {

                public:

                    using Matrix_t   = BackPropigationMatrixTrait::Type;
                    using ReturnType = BackPropigationMatrixTrait::Type&;

                    static void process_head(TensorID tid);
                    static void apply_to_children(std::stack<TensorID>& tid_stack, TensorID tid);
                    static ReturnType dereference(TensorID tid);
                    static Matrix_t grad(TensorID tid);
           };

        
            /*
                Breadth First Search through computational graph.

                Policy:
                    Compute Gradient
                    Read Parameter

            */
            template <TraversalPolicy TP = ComputeGradientPolicy>  
            class LevelOrderIterator {

                using Matrix_t = BackPropigationMatrixTrait::Type;
                using IterReturnType =  TP::ReturnType;

                constexpr static size_t NoOpIdx = 0;

                public:
                    explicit LevelOrderIterator(const TensorID _t) noexcept;

                    // LevelOrderIterator(LevelOrderIterator&) = default; 
                    // LevelOrderIterator(LevelOrderIterator&&) = default; 
                    // LevelOrderIterator& operator=(const LevelOrderIterator&) = default; 
                    // LevelOrderIterator& operator=(LevelOrderIterator&&) = default; 
                    LevelOrderIterator& operator++(void) noexcept;
                    
                    
                    Matrix_t gradient() const noexcept 
                    requires Same_as<TP, ReadParameterPolicy> {
                        return ReadParameterPolicy::grad(current);
                    }

                    IterReturnType operator*() const noexcept;

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