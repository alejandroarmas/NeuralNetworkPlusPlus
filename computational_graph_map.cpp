#include "tensor.h"
#include "computational_graph_map.h"
#include "m_algorithms_register.h"

#include <assert.h>

namespace NeuralNetwork {

    namespace Computation {


        namespace Graph {


            FunctionObject ComputationalGraphMap::_get_operation(TensorID my_tensor_id) noexcept { 
                // u_int16_t my_tensor_id = _t->get_tensor_id();

                assert(my_tensor_id > TensorID(0) && "Must be an op_id greater than 0.");
                assert(my_tensor_id <= tensor_id && "OP registry not this large");
                // if (my_tensor_id >= tensor_id) throw std::invalid_argument("OP registry not this large.");
                return op_registry.at(my_tensor_id.get()); 
            }

            std::shared_ptr<Tensor> ComputationalGraphMap::_get_tensor(TensorID my_tensor_id) noexcept { 

                std::cout << "Get Tensor ID: " << my_tensor_id.get() << std::endl;
                
                assert(my_tensor_id > TensorID(0) && "Must be an op_id greater than 0.");
                assert(my_tensor_id <= tensor_id && "OP registry not this large");

                return tensor_registry.at(my_tensor_id.get()); 
            }
        


        

            void ComputationalGraphMap::_recover_tensor_id(TensorID my_tensor_id) noexcept {
                
                recovered_tensor_id.push(my_tensor_id);
                tensor_registry.at(my_tensor_id.get()) = nullptr;
            }



            TensorID ComputationalGraphMap::_obtain_tensor_id() noexcept {
                // Matrix::Operations::Utility::Stringify stringify;

                TensorID next_tensor_id = TensorID(0);

                if (!recovered_tensor_id.empty()){
                    next_tensor_id = recovered_tensor_id.top();
                    recovered_tensor_id.pop();
                    // auto fn = Matrix::Operations::Utility::Function::from(_get_operation(next_tensor_id).get_code());
                    // std::cout << "Recovered Registry: O[" << next_tensor_id << "]" << std::visit(stringify, fn) << std::endl;
                    std::cout << "Recovered Registry: OP[" << next_tensor_id.get() << "]" << std::endl;
                } else {
                    next_tensor_id = ++tensor_id;
                }
                return next_tensor_id;
            }


            TensorID ComputationalGraphMap::_register_operation(std::shared_ptr<Tensor> _t, FunctionObject& _node) noexcept {

                TensorID my_tensor_id = _t->get_tensor_id();

                assert(my_tensor_id <= tensor_id && "OP registry not this large");

                op_registry.at(my_tensor_id.get()) = _node;
                tensor_registry.at(my_tensor_id.get()) = _t;


                std::cout << "Updated Operation: OP[" << my_tensor_id.get() << "]" << std::endl;
                return my_tensor_id;
            }




        }
        
    }

}