#ifndef MATRIX_PRINTER_H
#define MATRIX_PRINTER_H


#include <memory>

namespace Matrix {
    
    class Printer {

        public:
            void operator()(const Matrix::Representation& m) noexcept;
    };


}


#endif