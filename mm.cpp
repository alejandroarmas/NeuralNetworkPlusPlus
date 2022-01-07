#include <iostream>
#include <iomanip>

#include "mm.h"


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






