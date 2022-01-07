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


        class Normal : public Base {

            public:
                Normal(double _m, double _v) : mean(_m), variance(_v) {}
                std::unique_ptr<Matrix::Representation> operator() (std::unique_ptr<Matrix::Representation> m) override;
            private:
                double  mean; 
                double variance;
        };



        class Tester : public Base {

            public:
                std::unique_ptr<Matrix::Representation> operator() (std::unique_ptr<Matrix::Representation> m) override;
            private:
        };

      
    }

}

#endif