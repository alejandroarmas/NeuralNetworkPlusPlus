#ifndef MATRIX_GENERATOR_H
#define MATRIX_GENERATOR_H

#include <memory>

#include "mm.h"


namespace Matrix {

    namespace Generation {

        class Base {

            public:
                virtual std::unique_ptr<Matrix::Representation> operator() (std::unique_ptr<Matrix::Representation> m) = 0;
        };


        template <int Mean = 0, int Variance = 1> 
        class Normal : public Base {

            public:
                std::unique_ptr<Matrix::Representation> operator() (std::unique_ptr<Matrix::Representation> m) override;
        };


        template <int Val = 1>
        class Tester : public Base {

            public:
                std::unique_ptr<Matrix::Representation> operator() (std::unique_ptr<Matrix::Representation> m) override;
        };

      
    }

}

#include "t_generator.cpp"

#endif