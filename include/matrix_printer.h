#ifndef MATRIX_PRINTER_H
#define MATRIX_PRINTER_H


#include <memory>

namespace Matrix {
    
    class Printer {

        public:
            std::unique_ptr<Matrix::Representation> operator()(std::unique_ptr<Matrix::Representation> m);
    };


}


#endif