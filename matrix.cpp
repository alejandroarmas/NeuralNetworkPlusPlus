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


Matrix::Representation::Type Matrix::Representation::get_type(void) const noexcept {
    bool is_row_vector    = rows    == 1; 
    bool is_column_vector = columns == 1;
    bool is_scalar        = is_row_vector && is_column_vector;

    
    if (is_scalar) 
        return Type::SCALAR;
    else if (is_column_vector) 
        return Type::COLUMN_VECTOR;
    else if (is_row_vector) 
        return Type::ROW_VECTOR;
    else
        return Type::MATRIX;
}


std::string_view Matrix::Representation::get_type_string(void) const noexcept {
    bool is_row_vector    = rows    == 1; 
    bool is_column_vector = columns == 1;
    bool is_scalar        = is_row_vector && is_column_vector;

    
    if (is_scalar) 
        return "SCALAR";
    else if (is_column_vector) 
        return "COLUMN_VECTOR";
    else if (is_row_vector) 
        return "ROW_VECTOR";
    else
        return "MATRIX";
}





