#ifndef MATRIX_ALGORITHMS_H
#define MATRIX_ALGORITHMS_H

#include <algorithm>
#include <functional>
#include <exception>
#include <iostream>
#include <variant>

#include "matrix.h"
#include "m_algorithms_concepts.h"


namespace Matrix {


    namespace Operations {


        enum struct Code {
            NOP, MULTIPLY, PLUS, ReLU, OUTER_PRODUCT, HADAMARD,
        
        };
       

        class BaseInterface {

            public:
                virtual ~BaseInterface() = default;
                virtual std::unique_ptr<Matrix::Representation> operator()(
                    const std::unique_ptr<Matrix::Representation>& l, 
                    const std::unique_ptr<Matrix::Representation>& r = nullptr) = 0;
                virtual Code get_operation_code() = 0;

        };

        namespace Unary {


            template <class Implementation>
            class UnaryAdapter : public BaseInterface {

                public:
                    std::unique_ptr<Matrix::Representation> operator()(
                        const std::unique_ptr<Matrix::Representation>& l, 
                        const std::unique_ptr<Matrix::Representation>& r = nullptr) override {

                        if (!l) {
                            throw std::invalid_argument("Left operand not referencing a matrix.");
                        }


                        if (r != nullptr) {
                            throw std::invalid_argument("Unary Operation needs one operand.");
                        }
                        return Impl().operate(l); 
                        };
                    
                ~UnaryAdapter() = default;
                Code get_operation_code() override { return Impl().get_code(); };

                private:
                    Implementation& Impl() { return *static_cast<Implementation*>(this); }
                    friend Implementation;
                    

            };

            


            class ReLU : public UnaryAdapter<ReLU> {

                public:
                    std::unique_ptr<Matrix::Representation> operate(
                        const std::unique_ptr<Matrix::Representation>& m);
                Code get_code() { return Code::ReLU; };

            };

            static_assert(MatrixOperatable<ReLU>);


        }

        namespace Binary {


            template <class Implementation>
            class BaseOp : public BaseInterface {

                public:
                    BaseOp() = default;
                    virtual std::unique_ptr<Matrix::Representation> operator()(
                        const std::unique_ptr<Matrix::Representation>& l, 
                        const std::unique_ptr<Matrix::Representation>& r) override { 
                            
                        if (!l) {
                            throw std::invalid_argument("Left operand not referencing a matrix.");
                        }
                        if (!r) {
                            throw std::invalid_argument("Right operand not referencing a matrix.");
                        }
                        
                            return Impl().operate(l, r);
                        };
                    virtual ~BaseOp() = default;
                    Code get_operation_code() override { return Impl().get_code(); };
                private:
                    Implementation& Impl() { return *static_cast<Implementation*>(this); }
                    friend Implementation;

            };






            namespace Addition {


                class Std : public BaseOp<Std> {
                    public:
                        std::unique_ptr<Matrix::Representation> operate(
                            const std::unique_ptr<Matrix::Representation>& l, 
                            const std::unique_ptr<Matrix::Representation>& r);
                Code get_code() { return Code::PLUS; };

                };

            }

            static_assert(MatrixOperatable<Addition::Std>);


            namespace OuterProduct {



                class Naive : public BaseOp<Naive> {
                    public:
                        std::unique_ptr<Matrix::Representation> operate(
                            const std::unique_ptr<Matrix::Representation>& l, 
                            const std::unique_ptr<Matrix::Representation>& r);
                        Code get_code() { return Code::OUTER_PRODUCT; };

                };



            }

            static_assert(MatrixOperatable<OuterProduct::Naive>);





            namespace HadamardProduct {

                
                class Naive : public BaseOp<Naive> {

                    public:
                        std::unique_ptr<Matrix::Representation> operate(
                            const std::unique_ptr<Matrix::Representation>& l, 
                            const std::unique_ptr<Matrix::Representation>& r);
                        Code get_code() { return Code::HADAMARD; };

                };


                class Std : public BaseOp<Naive> {

                    public:
                        std::unique_ptr<Matrix::Representation> operate(
                            const std::unique_ptr<Matrix::Representation>& l, 
                            const std::unique_ptr<Matrix::Representation>& r);
                        Code get_code() { return Code::HADAMARD; };

                };


            }

            static_assert(MatrixOperatable<HadamardProduct::Naive>);
            static_assert(MatrixOperatable<HadamardProduct::Std>);



            /*
            Matrix Multiplication Usage:

                std::unique_ptr<Matrix::Representation> ma = std::make_unique<Matrix::Representation>(2000, 100);
                std::unique_ptr<Matrix::Representation> mb = std::make_unique<Matrix::Representation>(100, 3000);

                Matrix::Operations::Multiplication::Naive mul;

                std::unique_ptr<Matrix::Representation> mc = mul(ma, mb);
            */
            namespace Multiplication {


                class Naive : public BaseOp<Naive> {

                    public:
                        std::unique_ptr<Matrix::Representation> operate(
                            const std::unique_ptr<Matrix::Representation>& l, 
                            const std::unique_ptr<Matrix::Representation>& r);
                        Code get_code() { return Code::MULTIPLY; };


                };


                class Square : public BaseOp<Square> {

                        public:
                            std::unique_ptr<Matrix::Representation> operate(
                                const std::unique_ptr<Matrix::Representation>& l, 
                                const std::unique_ptr<Matrix::Representation>& r) ;
                        Code get_code() { return Code::MULTIPLY; };

                };


                class ParallelDNC : public BaseOp<ParallelDNC> {

                                public:
                                    std::unique_ptr<Matrix::Representation> operate(
                                        const std::unique_ptr<Matrix::Representation>& l, 
                                        const std::unique_ptr<Matrix::Representation>& r);
                        Code get_code() { return Code::MULTIPLY; };

                };


                void add_matmul_rec(std::vector<float>::iterator c, std::vector<float>::iterator a, std::vector<float>::iterator b, 
                        int m, int n, int p, int fdA, int fdB, int fdC);
                    

            } // namespace Multiplication

            static_assert(MatrixOperatable<Multiplication::Naive>);
            static_assert(MatrixOperatable<Multiplication::Square>);
            static_assert(MatrixOperatable<Multiplication::ParallelDNC>);
            
        }  // namespace Binary

    } // namespace Operations

} // namespace Matrix

#endif // MATRIX_ALGORITHMS_H