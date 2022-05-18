#ifndef COMPUTATIONAL_GRAPH_MAP_H
#define COMPUTATIONAL_GRAPH_MAP_H

#include <memory>
#include <stack>

#include "strong_types.h"
#include "m_algorithms_register.h"
#include "function_object.h"


namespace NeuralNetwork {

    namespace Computation {


        namespace Graph {

            class Tensor;
 
            /*
                DESCRIPTION:
                    Singleton Mediator that holds the edges between Tensors 
                    and their registered operations.

                    Organised as contiguous data structure, to avoid pointer 
                    chasing during runtime and help CPU's memory prefetcher 
                    load data before it's used.

            */
            class ComputationalGraphMap {


                public:
                    static ComputationalGraphMap& get(){
                        static ComputationalGraphMap map;
                        return map;
                    }
                    ComputationalGraphMap(ComputationalGraphMap const&) = delete;
                    ComputationalGraphMap(ComputationalGraphMap&&) = delete;
                    ComputationalGraphMap& operator=(ComputationalGraphMap const&) = delete;
                    ComputationalGraphMap& operator=(ComputationalGraphMap &&) = delete;

                    void _recover_tensor_id(TensorID my_tensor_id) noexcept;
                    std::shared_ptr<Tensor> _get_tensor(TensorID my_tensor_id) noexcept; 
                    FunctionObject _get_operation(TensorID my_tensor_id) noexcept;
                    TensorID _obtain_tensor_id() noexcept;
                    TensorID _register_operation(std::shared_ptr<Tensor> _t, FunctionObject& _node) noexcept;


                protected:
                    constexpr static uint16_t ENTRIES = 2000;
                    ComputationalGraphMap() :
                        op_registry(ENTRIES),
                        tensor_registry(ENTRIES),
                        recovered_tensor_id() {}


                private:
                    std::vector<FunctionObject> op_registry;
                    std::vector<std::shared_ptr<Tensor>> tensor_registry;
                    std::stack<TensorID> recovered_tensor_id;
                    static TensorID tensor_id;

                
            };

 
 
        }

    }

}


#endif // COMPUTATIONAL_GRAPH_MAP_H