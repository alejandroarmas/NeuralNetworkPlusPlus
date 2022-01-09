#ifndef MATRIX_ALGORITHMS_H
#define MATRIX_ALGORITHMS_H

#include <algorithm>
#include <functional>

#include "mm.h"


namespace Matrix {


    namespace Operations {

        class BaseOp {

            public:
                virtual std::unique_ptr<Matrix::Representation> operator()(
                    Matrix::Representation l, 
                    Matrix::Representation r) = 0;

        };

        namespace Add {


            class BaseAdd : BaseOp {

            };


            class Std : public BaseAdd {
                public:
                    std::unique_ptr<Matrix::Representation> operator()(
                        Matrix::Representation l, 
                        Matrix::Representation r) override;

            };

        }


        namespace HadamardProduct {

            class BaseHP : BaseOp {

            };

            class Naive : public BaseHP {

                public:
                    std::unique_ptr<Matrix::Representation> operator()(
                        Matrix::Representation l, 
                        Matrix::Representation r) override;

            };


            class Std : public BaseHP {

                public:
                    std::unique_ptr<Matrix::Representation> operator()(
                        Matrix::Representation l, 
                        Matrix::Representation r) override;

            };


        }




        /*
        Matrix Multiplication Usage:

            std::unique_ptr<Matrix::Representation> ma = std::make_unique<Matrix::Representation>(2000, 100);
            std::unique_ptr<Matrix::Representation> mb = std::make_unique<Matrix::Representation>(100, 3000);

            Matrix::Operations::Multiplication::Naive mul;

            std::unique_ptr<Matrix::Representation> mc = mul(*ma, *mb);
        */
        namespace Multiplication {

            class BaseMul : BaseOp {

            };

            class Naive : public BaseMul {

                public:
                    std::unique_ptr<Matrix::Representation> operator()(
                        Matrix::Representation l, 
                        Matrix::Representation r) override;

            };

            class Square : public BaseMul {

                            public:
                                std::unique_ptr<Matrix::Representation> operator()(
                                    Matrix::Representation l, 
                                    Matrix::Representation r) override;

                        };



        }

    }

}

#endif // MATRIX_ALGORITHMS_H