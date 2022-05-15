#ifndef MATRIX_ALGORITHMS_CONCEPTS_H
#define MATRIX_ALGORITHMS_CONCEPTS_H

#include <concepts>
#include <memory>

#include "matrix.h"

template< class Derived, class Base >
concept Derived_from =
  std::is_base_of_v<Base, Derived> &&
  std::is_convertible_v<const volatile Derived*, const volatile Base*>;

template< class T, class U >
concept SameHelper = std::is_same_v<T, U>;


template< class T, class U >
concept Same_as = SameHelper<T, U> && SameHelper<U, T>;


namespace Matrix {

    namespace Operations {




        template <typename T>
        concept UnaryMatrixOperatable = requires(T _op, Matrix::Representation mtx) {
            _op.operate(mtx);
            { _op.operate(mtx) } -> Same_as<decltype(mtx)>;
        };

        template <typename T>
        concept BinaryMatrixOperatable = requires(T _op, Matrix::Representation mtx) {
            _op.operate(mtx, mtx);
            { _op.operate(mtx, mtx) } -> Same_as<decltype(mtx)>;
        };

        // template <typename T>
        // concept MetricOperatable = BinaryMatrixOperatable<T> || requires(T _op, Matrix::Representation mtx) {
        //     {mtx.num_rows()} == 1;
        // };

        template <typename T>
        concept MatrixOperatable = BinaryMatrixOperatable<T> || UnaryMatrixOperatable<T>;
        



    }
}

namespace NeuralNetwork {

    namespace Computation {

        namespace Graph {


            template <typename State>
            concept IsStateFull = requires(State state) {
                state.is_state();
                { state.is_state() } -> Same_as<void>;
            };

            template <typename Registry>
            concept NoOperandRegistry = IsStateFull<Registry> && requires(Registry registry) {

                registry.get_tensor_id();
                { registry.get_tensor_id() } -> Same_as<TensorID>;
            };


            template <typename Registry>
            concept UnaryRegistry = NoOperandRegistry<Registry> && requires(Registry registry) {
                
                registry.left_op_id();                
                { registry.left_op_id()    } -> Same_as<TensorID>;
            };

            template <typename Registry>
            concept BinaryRegistry = UnaryRegistry<Registry> && requires(Registry registry) {
                
                registry.right_op_id();                
                { registry.right_op_id()   } -> Same_as<TensorID>;
            };



        }
    }
}

#endif // MATRIX_ALGORITHMS_CONCEPTS_H