#ifndef MATRIX_ALGORITHMS_H
#define MATRIX_ALGORITHMS_H

#include <algorithm>
#include <functional>
#include <exception>

#include "matrix.h"


namespace Matrix {


    namespace Operations {


        class BaseInterface {

            public:
                virtual ~BaseInterface() = default;
                virtual std::unique_ptr<Matrix::Representation> operator()(
                    const std::unique_ptr<Matrix::Representation>& l, 
                    const std::unique_ptr<Matrix::Representation>& r = nullptr) = 0;
                

        };

        namespace Unary {


            template <class Implementation>
            class UnaryAdapter : public BaseInterface {

                std::unique_ptr<Matrix::Representation> operator()(
                    const std::unique_ptr<Matrix::Representation>& l, 
                    const std::unique_ptr<Matrix::Representation>& r = nullptr) {
                            
                    if (r != nullptr) {
                        throw std::invalid_argument("Unary Operation needs one operand.");
                    }
                    return Impl().operator()(l); 
                    };
                    
                ~UnaryAdapter() = default;
                private:
                    Implementation& Impl() { return *static_cast<Implementation*>(this); }
                    friend Implementation;
                    

            };


            class ReLU : public UnaryAdapter<ReLU> {

                public:
                    std::unique_ptr<Matrix::Representation> operator()(
                        const std::unique_ptr<Matrix::Representation>& m);

            };



        }

        namespace Binary {


            template <class Implementation>
            class BaseOp : public BaseInterface {

                public:
                    BaseOp() = default;
                    virtual std::unique_ptr<Matrix::Representation> operator()(
                        const std::unique_ptr<Matrix::Representation>& l, 
                        const std::unique_ptr<Matrix::Representation>& r) { return Impl().operator()(l, r); };
                    virtual ~BaseOp() = default;
                private:
                    Implementation& Impl() { return *static_cast<Implementation*>(this); }
                    friend Implementation;

            };



            std::string debug_message(const std::unique_ptr<Matrix::Representation>& l, 
                                    const std::unique_ptr<Matrix::Representation>& r);
            
            std::string debug_message_2(const std::unique_ptr<Matrix::Representation>& l, 
                                    const std::unique_ptr<Matrix::Representation>& r);



            namespace Addition {


                class Std : public BaseOp<Std> {
                    public:
                        std::unique_ptr<Matrix::Representation> operator()(
                            const std::unique_ptr<Matrix::Representation>& l, 
                            const std::unique_ptr<Matrix::Representation>& r);

                };

            }

            namespace OuterProduct {



                class Naive : public BaseOp<Naive> {
                    public:
                        std::unique_ptr<Matrix::Representation> operator()(
                            const std::unique_ptr<Matrix::Representation>& l, 
                            const std::unique_ptr<Matrix::Representation>& r);

                };



            }






            namespace HadamardProduct {

                
                class Naive : public BaseOp<Naive> {

                    public:
                        std::unique_ptr<Matrix::Representation> operator()(
                            const std::unique_ptr<Matrix::Representation>& l, 
                            const std::unique_ptr<Matrix::Representation>& r);

                };


                class Std : public BaseOp<Naive> {

                    public:
                        std::unique_ptr<Matrix::Representation> operator()(
                            const std::unique_ptr<Matrix::Representation>& l, 
                            const std::unique_ptr<Matrix::Representation>& r);

                };


            }




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
                        std::unique_ptr<Matrix::Representation> operator()(
                            const std::unique_ptr<Matrix::Representation>& l, 
                            const std::unique_ptr<Matrix::Representation>& r);

                };


                class Square : public BaseOp<Square> {

                                public:
                                    std::unique_ptr<Matrix::Representation> operator()(
                                        const std::unique_ptr<Matrix::Representation>& l, 
                                        const std::unique_ptr<Matrix::Representation>& r) ;

                };


                class ParallelDNC : public BaseOp<ParallelDNC> {

                                public:
                                    std::unique_ptr<Matrix::Representation> operator()(
                                        const std::unique_ptr<Matrix::Representation>& l, 
                                        const std::unique_ptr<Matrix::Representation>& r);

                };


                void add_matmul_rec(std::vector<float>::iterator c, std::vector<float>::iterator a, std::vector<float>::iterator b, 
                        int m, int n, int p, int fdA, int fdB, int fdC);
                    

            } // namespace Multiplication
            
        }  // namespace Binary

    } // namespace Operations

} // namespace Matrix

#endif // MATRIX_ALGORITHMS_H