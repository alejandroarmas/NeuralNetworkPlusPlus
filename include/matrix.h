#ifndef MATRIX_MULTIPLY_H
#define MATRIX_MULTIPLY_H

#include <vector>
#include <cstdint>
#include <stdexcept>
#include <memory>




namespace Matrix {


    class Representation {
        public:
            typedef typename std::vector<float>::iterator matrix_iter;
            
            Representation(u_int64_t _l, u_int64_t _w) : rows(_l), columns(_w), data(std::vector<float>(_l * _w, 0)) {}
            Representation(const Matrix::Representation& _other) : rows(_other.rows), columns(_other.columns), data(_other.data) {}
            
            bool operator==(const Matrix::Representation& _other);
            bool operator!=(const Matrix::Representation& _other);

            u_int64_t num_rows() const { return rows; }
            u_int64_t num_cols() const { return columns; }
            
            float get(u_int64_t r, u_int64_t c);
            void put(u_int64_t r, u_int64_t c, float val);
            
            matrix_iter scanStart() { return this->data.begin(); }
            matrix_iter scanEnd() { return this->data.end(); }
        private:
            u_int64_t rows;
            u_int64_t columns;
            std::vector<float> data;
    };

}



#endif