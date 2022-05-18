#include <iostream>
#include <iomanip>
#include <assert.h>

#include "matrix.h"
#include "functions.h"


bool Matrix::Representation::operator==(const Matrix::Representation _other) noexcept {

    bool isEqual = this->data.size() == _other.data.size();
    
    for (size_t i = 0; isEqual && i < this->data.size(); i++) {
        isEqual = Functions::Utility::compare_float(this->data.at(i), _other.data.at(i));
    }

    return isEqual;
}


bool Matrix::Representation::operator!=(const Matrix::Representation _other) noexcept {
    
    bool isEqual = this->data.size() == _other.data.size();    

    for (size_t i = 0; isEqual && i < this->data.size(); i++) {
        isEqual = Functions::Utility::compare_float(this->data.at(i), _other.data.at(i));
    }

    return !isEqual;
}

float Matrix::Representation::get(u_int64_t r, u_int64_t c) const noexcept {

    assert(r <= rows && c <= columns && "Invalid Matrix Index.");

    uint64_t calculated_index = c + r * columns; 

    return data.at(calculated_index);

}


void Matrix::Representation::put(u_int64_t r, u_int64_t c, float val) noexcept {

    assert(r <= rows && c <= columns && "Invalid Matrix Index.");

    uint64_t calculated_index = c + r * columns; 

    data.at(calculated_index) = val;

}







