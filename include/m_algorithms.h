#ifndef MATRIX_ALGORITHMS_H
#define MATRIX_ALGORITHMS_H

#include <algorithm>
#include <functional>
#include <exception>
#include <iostream>
#include <variant>

#include <assert.h>

#include "matrix.h"
#include "m_algorithms_concepts.h"


namespace Matrix {


    namespace Operations {


        enum class Code {
            NOP, MULTIPLY, PLUS, ReLU, SoftMax, OUTER_PRODUCT, HADAMARD, CROSS_ENTROPY,        
        };
       

      
        namespace Unary {


            template <class Implementation>
            class UnaryAdapter {

                public:
                    Matrix::Representation operator()(
                        const Matrix::Representation& l) const noexcept {

                        return Impl().operate(l); 
                        };
                    
                ~UnaryAdapter() = default;
                private:
                    Implementation& Impl() const noexcept { return *static_cast<Implementation*>(const_cast<UnaryAdapter<Implementation>*>(this)); }
                    friend Implementation;
                    
            };

            


            class ReLU : public UnaryAdapter<ReLU> {

                public:
                    Matrix::Representation operate(
                        const Matrix::Representation& m) const noexcept;
            };

            static_assert(MatrixOperatable<ReLU>);


            class SoftMax : public UnaryAdapter<ReLU> {

                public:
                    Matrix::Representation operate(
                        const Matrix::Representation& m) const noexcept;
            };

            static_assert(MatrixOperatable<SoftMax>);


            // class Transpose : public UnaryAdapter<ReLU> {

            //     public:
            //         Matrix::Representation operate(
            //             const Matrix::Representation& m) const noexcept;
            // };

        }

        namespace Metric {



            template <class Implementation>
            class BaseOp {

                public:
                    BaseOp() = default;
                    ~BaseOp() = default;
                    Matrix::Representation operator()(
                        const Matrix::Representation& l, 
                        const Matrix::Representation& r) const noexcept { 
                            
                            bool rows_compatable = l.num_rows() == r.num_rows();
                            bool cols_compatable = l.num_cols() == r.num_cols();
                            bool is_vector = l.num_rows() == 1 || l.num_cols() == 1;

                            assert(rows_compatable && cols_compatable && is_vector);
                            
                            auto result = Impl().operate(l, r);

                            assert(result.get_type() == Matrix::Representation::Type::SCALAR && "Metric Operation must return scalar.");

                            return Matrix::Representation{result};
                        }
                private:
                    Implementation& Impl() const noexcept { return *static_cast<Implementation*>(const_cast<BaseOp<Implementation>*>(this)); }
                    friend Implementation;

            };


            class CrossEntropy : public BaseOp<CrossEntropy> {
                public:
                    Matrix::Representation operate(
                        const Matrix::Representation& p, 
                        const Matrix::Representation& q) const noexcept;
            };

        
            static_assert(MatrixOperatable<CrossEntropy>);


        } // Metric


        namespace Binary {


            template <class Implementation>
            class BaseOp {

                public:
                    BaseOp() = default;
                    ~BaseOp() = default;
                    Matrix::Representation operator()(
                        const Matrix::Representation& l, 
                        const Matrix::Representation& r) const noexcept { 
                                                    
                            return Impl().operate(l, r);
                        };
                private:
                    Implementation& Impl() const noexcept { return *static_cast<Implementation*>(const_cast<BaseOp<Implementation>*>(this)); }
                    friend Implementation;

            };






            namespace Addition {


                class Std : public BaseOp<Std> {
                    public:
                        Matrix::Representation operate(
                            const Matrix::Representation& l, 
                            const Matrix::Representation& r) const noexcept;
                };

            }

            static_assert(MatrixOperatable<Addition::Std>);

            namespace Subtraction {


                class Std : public BaseOp<Std> {
                    public:
                        Matrix::Representation operate(
                            const Matrix::Representation& l, 
                            const Matrix::Representation& r) const noexcept;
                };

            }

            static_assert(MatrixOperatable<Subtraction::Std>);


            namespace OuterProduct {



                class Naive : public BaseOp<Naive> {
                    public:
                        Matrix::Representation operate(
                            const Matrix::Representation& l, 
                            const Matrix::Representation& r) const noexcept;
                };



            }

            static_assert(MatrixOperatable<OuterProduct::Naive>);





            namespace HadamardProduct {

                
                class Naive : public BaseOp<Naive> {

                    public:
                        Matrix::Representation operate(
                            const Matrix::Representation& l, 
                            const Matrix::Representation& r) const noexcept;
                };


                class Std : public BaseOp<Naive> {

                    public:
                        Matrix::Representation operate(
                            const Matrix::Representation& l, 
                            const Matrix::Representation& r) const noexcept;
                };


            }

            static_assert(MatrixOperatable<HadamardProduct::Naive>);
            static_assert(MatrixOperatable<HadamardProduct::Std>);



            /*
            Matrix Multiplication Usage:

                Matrix::Representation ma = Matrix::Representation(2000, 100);
                Matrix::Representation mb = Matrix::Representation(100, 3000);

                Matrix::Operations::Multiplication::Naive mul;

                Matrix::Representation mc = mul(ma, mb);
            */
            namespace Multiplication {


                class Naive : public BaseOp<Naive> {

                    public:
                        Matrix::Representation operate(
                            const Matrix::Representation& l, 
                            const Matrix::Representation& r) const noexcept;

                };


                class Square : public BaseOp<Square> {

                        public:
                            Matrix::Representation operate(
                                const Matrix::Representation& l, 
                                const Matrix::Representation& r) const noexcept;
                };


                class ParallelDNC : public BaseOp<ParallelDNC> {

                                public:
                                    Matrix::Representation operate(
                                        const Matrix::Representation& l, 
                                        const Matrix::Representation& r) const noexcept;
                };


                void add_matmul_rec(std::vector<float>::iterator c, std::vector<float>::iterator a, std::vector<float>::iterator b, 
                        int m, int n, int p, int fdA, int fdB, int fdC) noexcept;
                    

            } // namespace Multiplication

            static_assert(MatrixOperatable<Multiplication::Naive>);
            static_assert(MatrixOperatable<Multiplication::Square>);
            static_assert(MatrixOperatable<Multiplication::ParallelDNC>);
            
        }  // namespace Binary

    } // namespace Operations

} // namespace Matrix

#endif // MATRIX_ALGORITHMS_H