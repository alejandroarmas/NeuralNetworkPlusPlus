#include <iostream>
#include <iomanip>

#include "matrix.h"
#include "functions.h"


bool Matrix::Representation::operator==(const Matrix::Representation& _other) {

    bool isEqual = this->data.size() == _other.data.size();
    
    for (size_t i = 0; isEqual && i < this->data.size(); i++) {
        isEqual = Functions::Utility::compare_float(this->data.at(i), _other.data.at(i));
    }

    return isEqual;
}


bool Matrix::Representation::operator!=(const Matrix::Representation& _other) {
    
    bool isEqual = this->data.size() == _other.data.size();    

    for (size_t i = 0; isEqual && i < this->data.size(); i++) {
        isEqual = Functions::Utility::compare_float(this->data.at(i), _other.data.at(i));
    }

    return !isEqual;
}



float Matrix::Representation::get(u_int64_t r, u_int64_t c) {

                uint64_t calculated_index = c + r * columns; 

                if (r <= rows && c <= columns) {
                    return data.at(calculated_index);
                }
                else throw std::range_error("Index not accepted for this Matrix.");

            }


void Matrix::Representation::put(u_int64_t r, u_int64_t c, float val) {

                uint64_t calculated_index = c + r * columns; 

                if (r <= rows && c <= columns) {
                    data.at(calculated_index) = val;
                }
                else throw std::range_error("Index not accepted for this Matrix.");

            }






