
#include <iostream>

#include "m_algorithms_utilities.h"

namespace Matrix {

    namespace Operations {

        namespace Utility {


            std::string_view debug_message(const Matrix::Representation& l, 
                    const Matrix::Representation& r) {

                std::string error_msg = "Matrix A Columns not equal to Matrix B Rows: [" + 
                    std::to_string(l.num_rows()) + "," + 
                    std::to_string(r.num_rows()) + "," + 
                    std::to_string(l.num_cols()) + "] X [" + 
                    std::to_string(r.num_cols()) + "]";


                return std::string_view{error_msg};
                }


            std::string_view debug_message_2(const Matrix::Representation& l, 
                    const Matrix::Representation& r) {
                
                std::string error_msg = "Matrix A size not equal to Matrix B: [" + 
                    std::to_string(l.num_rows()) + "," + 
                    std::to_string(l.num_cols()) + "] X [" + 
                    std::to_string(r.num_rows()) + "," + 
                    std::to_string(r.num_cols()) + "]";


                return std::string_view{error_msg};
                }

        }

    }
}