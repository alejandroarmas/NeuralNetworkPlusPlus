#ifndef MATRIX_ALGORITHMS_H
#define MATRIX_ALGORITHMS_H

#include <algorithm>
#include <functional>

#include "matrix.h"


namespace Matrix {


    namespace Operations {

        class BaseBinaryOpInterface {

            public:
                virtual ~BaseBinaryOpInterface() = default;
                virtual std::unique_ptr<Matrix::Representation> operator()(
                    std::unique_ptr<Matrix::Representation>& l, 
                    std::unique_ptr<Matrix::Representation>& r) = 0;
                

        };

        template <class Implementation>
        class BaseOp : public BaseBinaryOpInterface{

            public:
                virtual std::unique_ptr<Matrix::Representation> operator()(
                    std::unique_ptr<Matrix::Representation>& l, 
                    std::unique_ptr<Matrix::Representation>& r) { return Impl().operator()(l, r); };
                virtual ~BaseOp() = default;
            private:
                BaseOp& Impl() { return *static_cast<Implementation*>(this); }
                BaseOp() = default;
                friend Implementation;

        };



        std::string debug_message(std::unique_ptr<Matrix::Representation>& l, 
                                std::unique_ptr<Matrix::Representation>& r);
        
        std::string debug_message_2(std::unique_ptr<Matrix::Representation>& l, 
                                std::unique_ptr<Matrix::Representation>& r);



        namespace Addition {


            class Std : public BaseOp<Std> {
                public:
                    std::unique_ptr<Matrix::Representation> operator()(
                        std::unique_ptr<Matrix::Representation>& l, 
                        std::unique_ptr<Matrix::Representation>& r) override;

            };

        }

        namespace OuterProduct {



            class Naive : public BaseOp<Naive> {
                public:
                    std::unique_ptr<Matrix::Representation> operator()(
                        std::unique_ptr<Matrix::Representation>& l, 
                        std::unique_ptr<Matrix::Representation>& r) override;

            };



        }


        namespace HadamardProduct {

            
            class Naive : public BaseOp<Naive> {

                public:
                    std::unique_ptr<Matrix::Representation> operator()(
                        std::unique_ptr<Matrix::Representation>& l, 
                        std::unique_ptr<Matrix::Representation>& r) override;

            };


            class Std : public BaseOp<Naive> {

                public:
                    std::unique_ptr<Matrix::Representation> operator()(
                        std::unique_ptr<Matrix::Representation>& l, 
                        std::unique_ptr<Matrix::Representation>& r) override;

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
                        std::unique_ptr<Matrix::Representation>& l, 
                        std::unique_ptr<Matrix::Representation>& r) override;

            };


            class Square : public BaseOp<Square> {

                            public:
                                std::unique_ptr<Matrix::Representation> operator()(
                                    std::unique_ptr<Matrix::Representation>& l, 
                                    std::unique_ptr<Matrix::Representation>& r) override;

            };


            class ParallelDNC : public BaseOp<ParallelDNC> {

                            public:
                                std::unique_ptr<Matrix::Representation> operator()(
                                    std::unique_ptr<Matrix::Representation>& l, 
                                    std::unique_ptr<Matrix::Representation>& r) override;

            };


            void add_matmul_rec(std::vector<float>::iterator c, std::vector<float>::iterator a, std::vector<float>::iterator b, 
                    int m, int n, int p, int fdA, int fdB, int fdC);
                

        }

    }

}

#endif // MATRIX_ALGORITHMS_H