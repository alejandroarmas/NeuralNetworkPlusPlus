#include "tensor_backwards_pass.h"
#include "m_algorithms_utilities.h"

#include <iostream>
#include <iomanip>

#include <variant>
#include <utility>
#include <stack>


namespace NeuralNetwork {

    namespace Computation {

        namespace Graph {


            void ReversePass::backwards(Tensor& _t, 
                PrintTag _ ) {

                    for (auto it = _t.begin(); it != _t.end(); ++it) {
                        std::cout << "Reverse Pass Iteration." << std::endl;
                    }


                }


         
                
 
        }

    }

}