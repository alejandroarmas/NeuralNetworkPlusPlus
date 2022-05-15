#ifndef STRONG_TYPE_UTILITY_H
#define STRONG_TYPE_UTILITY_H

#include <cstdint>
#include <concepts>


namespace Matrix {

    template < class T >
    concept Integral = std::is_integral_v<T>;


    template <typename T, typename Parameter>
    class NamedType {
    public:
        constexpr explicit NamedType(T const& value) : value_(value) {}
        constexpr explicit NamedType(T&& value) : value_(std::move(value)) {}
        constexpr T& get() { return value_; }
        constexpr T const& get() const {return value_; }
        
        template <Integral U = T>
        constexpr NamedType operator++(int) {
            value_++;
            return *this;
        }
        template <Integral U = T>
        constexpr NamedType operator++() {
            ++value_;
            return *this;
        }
        template <Integral U = T>
        constexpr bool operator<(const NamedType& rhs) const {
            return value_ < rhs.value_;
        }
        template <Integral U = T>
        constexpr bool operator>(const NamedType& rhs) const {
            return value_ > rhs.value_;
        }
        template <Integral U = T>
        constexpr bool operator>=(const NamedType& rhs) const {
            return value_ >= rhs.value_;
        }
        template <Integral U = T>
        constexpr bool operator<=(const NamedType& rhs) const {
            return value_ <= rhs.value_;
        }
        template <Integral U = T>
        constexpr bool operator==(const NamedType& rhs) const {
            return value_ == rhs.value_;
        }
    private:
        T value_;
    };


    using Rows    = NamedType<u_int64_t, struct RowParameter>;
    using Columns = NamedType<u_int64_t, struct ColumnParameter>;

}





namespace NeuralNetwork {

    namespace Computation {


        namespace Graph {


            using IsTrackable  = 
                Matrix::NamedType<bool, struct TrackParameter>;


            using IsLeaf       = 
                Matrix::NamedType<bool, struct LeafParameter>;


            using IsRecordable = 
                Matrix::NamedType<bool, struct RecordParameter>;


            using IsIdentity = 
                Matrix::NamedType<bool, struct IdentityParameter>;

                
            using TensorID = Matrix::NamedType<u_int16_t, struct TensorIDParameter>;

        }

    }

}


#endif // STRONG_TYPE_UTILITY_H