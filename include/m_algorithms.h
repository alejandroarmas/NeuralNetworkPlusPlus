#ifndef MATRIX_ALGORITHMS_H
#define MATRIX_ALGORITHMS_H

#include <algorithm>
#include <functional>

#include "matrix.h"


namespace Matrix {


    namespace Operations {

        class BaseOp {

            public:
                virtual std::unique_ptr<Matrix::Representation> operator()(
                    std::unique_ptr<Matrix::Representation>& l, 
                    std::unique_ptr<Matrix::Representation>& r) = 0;
                virtual ~BaseOp() = default;

        };


        namespace Addition {


            class Std : public BaseOp {
                public:
                    std::unique_ptr<Matrix::Representation> operator()(
                        std::unique_ptr<Matrix::Representation>& l, 
                        std::unique_ptr<Matrix::Representation>& r) override;

            };

        }


        namespace HadamardProduct {

            
            class Naive : public BaseOp {

                public:
                    std::unique_ptr<Matrix::Representation> operator()(
                        std::unique_ptr<Matrix::Representation>& l, 
                        std::unique_ptr<Matrix::Representation>& r) override;

            };


            class Std : public BaseOp {

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


            class Naive : public BaseOp {

                public:
                    std::unique_ptr<Matrix::Representation> operator()(
                        std::unique_ptr<Matrix::Representation>& l, 
                        std::unique_ptr<Matrix::Representation>& r) override;

            };


            class Square : public BaseOp {

                            public:
                                std::unique_ptr<Matrix::Representation> operator()(
                                    std::unique_ptr<Matrix::Representation>& l, 
                                    std::unique_ptr<Matrix::Representation>& r) override;

            };


            class RecursiveParallel : public BaseOp {

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