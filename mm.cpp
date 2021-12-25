#include <iostream>
#include <iomanip>

#include "mm.h"

#define MAX_PRINT_WIDTH 5
#define MAX_PRINT_LENGTH 5



template <class T>
T Matrix::Representation<T>::get(u_int64_t r, u_int64_t c) {

                uint64_t calculated_index = c + r * columns; 

                if (r <= rows && c <= columns) {
                    return data.at(calculated_index);
                }
                else throw std::range_error("Index not accepted for this Matrix.");

            }


template <class T>
void Matrix::Representation<T>::put(u_int64_t r, u_int64_t c, T val) {

                uint64_t calculated_index = c + r * columns; 

                if (r <= rows && c <= columns) {
                    data.at(calculated_index) = val;
                }
                else throw std::range_error("Index not accepted for this Matrix.");

            }






