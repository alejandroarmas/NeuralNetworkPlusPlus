
#include "m_algorithms_register.h"

#include "m_algorithms.h"
#include <iostream>

#include <memory>

namespace NeuralNetwork {

    namespace Computation {

        namespace Graph {

            RegisteredOperation::NodePair RegisteredOperation::get_operands() const {

                        if (operand && bin_operand) {
                            return 
                            {
                                    this->operand, 
                                    this->bin_operand
                            };
                        }
                        else if (operand) {
                            return 
                            {
                                    this->operand, 
                                    nullptr
                            };
                        }
                        else if (bin_operand) {
                            return 
                            {
                                    nullptr,
                                    this->bin_operand 
                            };
                        }

                        return 
                        {
                                nullptr,
                                nullptr 
                        };

                     }

        }

    }

}
