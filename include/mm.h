#ifndef MATRIX_MULTIPLY_H
#define MATRIX_MULTIPLY_H

#include <vector>
#include <cstdint>
#include <stdexcept>
#include <memory>




namespace Matrix {



    template <class T>
    class Representation {
            typedef typename std::vector<T>::iterator matrix_iter;
        public:
            Representation(u_int64_t _l, u_int64_t _w) : rows(_l), columns(_w), data(std::move(std::vector<T>(_l * _w, 0))) {}
            Representation(const Matrix::Representation<T>& _other) : rows(_other.rows), columns(_other.columns), data(_other.data) {}
            u_int64_t num_rows() const { return rows; }
            u_int64_t num_cols() const { return columns; }
            T get(u_int64_t r, u_int64_t c);
            void put(u_int64_t r, u_int64_t c, T val);
            matrix_iter scanStart() { return this->data.begin(); }
            matrix_iter scanEnd() { return this->data.end(); }
        private:
            u_int64_t rows;
            u_int64_t columns;
            std::vector<T> data;
    };


    template class Representation<float>;
    template class Representation<double>;
    template class Representation<int8_t>;
    template class Representation<int16_t>;
    template class Representation<int32_t>;
    template class Representation<int64_t>;
    template class Representation<uint8_t>;
    template class Representation<uint16_t>;
    template class Representation<uint32_t>;
    template class Representation<uint64_t>;
    
}



#endif