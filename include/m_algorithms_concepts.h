#ifndef MATRIX_ALGORITHMS_CONCEPTS_H
#define MATRIX_ALGORITHMS_CONCEPTS_H

#include <concepts>
#include <memory>

#include "matrix.h"

namespace Matrix {

    namespace Operations {


        template< class T, class U >
        concept SameHelper = std::is_same_v<T, U>;
        
    
        template< class T, class U >
        concept same_as = SameHelper<T, U> && SameHelper<U, T>;


        template <typename T>
        concept UnaryMatrixOperatable = requires(T _op, std::unique_ptr<Matrix::Representation> mtx) {
            _op.operate(mtx);
            { _op.operate(mtx) } -> same_as<decltype(mtx)>;
        };

        template <typename T>
        concept BinaryMatrixOperatable = requires(T _op, std::unique_ptr<Matrix::Representation> mtx) {
            _op.operate(mtx, mtx);
            { _op.operate(mtx, mtx) } -> same_as<decltype(mtx)>;
        };

        template <typename T>
        concept MatrixOperatable = BinaryMatrixOperatable<T> || UnaryMatrixOperatable<T>;
        

    }
}

#endif // MATRIX_ALGORITHMS_CONCEPTS_H