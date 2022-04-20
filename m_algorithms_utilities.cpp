
#include <iostream>

#include "m_algorithms_utilities.h"

namespace Matrix {

    namespace Operations {

        namespace Utility {


            std::string debug_message(const std::unique_ptr<Matrix::Representation>& l, 
                    const std::unique_ptr<Matrix::Representation>& r) {

                std::string error_msg = "Matrix A Columns not equal to Matrix B Rows: [" + 
                    std::to_string(l->num_rows()) + "," + 
                    std::to_string(l->num_cols()) + "] X [" + 
                    std::to_string(r->num_rows()) + "," + 
                    std::to_string(r->num_cols()) + "]";


                return error_msg;
                }


            std::string debug_message_2(const std::unique_ptr<Matrix::Representation>& l, 
                    const std::unique_ptr<Matrix::Representation>& r) {
                
                std::string error_msg = "Matrix A size not equal to Matrix B: [" + 
                    std::to_string(l->num_rows()) + "," + 
                    std::to_string(l->num_cols()) + "] X [" + 
                    std::to_string(r->num_rows()) + "," + 
                    std::to_string(r->num_cols()) + "]";


                return error_msg;
                }

        }

    }
}