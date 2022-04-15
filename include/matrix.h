#ifndef MATRIX_REPRESENTATION_H
#define MATRIX_REPRESENTATION_H

#include <vector>
#include <cstdint>
#include <stdexcept>
#include <memory>




namespace Matrix {


    template <typename T, typename Parameter>
    class NamedType {
    public:
        constexpr explicit NamedType(T const& value) : value_(value) {}
        constexpr explicit NamedType(T&& value) : value_(std::move(value)) {}
        constexpr T& get() { return value_; }
        constexpr T const& get() const {return value_; }
    private:
        T value_;
    };

    using Rows    = NamedType<u_int64_t, struct RowParameter>;
    using Columns = NamedType<u_int64_t, struct ColumnParameter>;


    class Representation {


        public:
            typedef typename std::vector<float>::iterator matrix_iter;
            
            Representation(Rows _l, Columns _w) : rows(_l.get()), columns(_w.get()), data(std::vector<float>(_l.get() * _w.get(), 0)) {}
            Representation(const Matrix::Representation& _other) : rows(_other.num_rows()), columns(_other.num_cols()), data(_other.data) {}
            Representation(const Matrix::Representation&& _other) : rows(_other.num_rows()), columns(_other.num_cols()), data(std::move(_other.data)) {}
            
            bool operator==(const Matrix::Representation _other);
            bool operator!=(const Matrix::Representation _other);

            constexpr u_int64_t num_rows() const { return rows; }
            constexpr u_int64_t num_cols() const { return columns; }
            
            // constexpr float get(u_int64_t r, u_int64_t c);
            // constexpr void put(u_int64_t r, u_int64_t c, float val);
            
            constexpr float get(u_int64_t r, u_int64_t c) {

                uint64_t calculated_index = c + r * columns; 

                if (r <= rows && c <= columns) {
                    return data.at(calculated_index);
                }
                else throw std::range_error("Index not accepted for this Matrix.");

            }


            constexpr void put(u_int64_t r, u_int64_t c, float val) {

                uint64_t calculated_index = c + r * columns; 

                if (r <= rows && c <= columns) {
                    data.at(calculated_index) = val;
                }
                else throw std::range_error("Index not accepted for this Matrix.");

            }


            constexpr matrix_iter scanStart() { return this->data.begin(); }
            constexpr matrix_iter scanEnd() { return this->data.end(); }
        private:
            u_int64_t rows;
            u_int64_t columns;
            std::vector<float> data;
    };

}



#endif