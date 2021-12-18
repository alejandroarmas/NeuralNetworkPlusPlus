#ifndef MATRIX_PRINTER_H
#define MATRIX_PRINTER_H

namespace Matrix {
    template <class T>
    class Printer {

        public:
            void operator()(Matrix::Representation<T> m);
    };


    template class Printer<float>;
    template class Printer<double>;
    template class Printer<int8_t>;
    template class Printer<int16_t>;
    template class Printer<int32_t>;
    template class Printer<int64_t>;
    template class Printer<uint8_t>;
    template class Printer<uint16_t>;
    template class Printer<uint32_t>;
    template class Printer<uint64_t>;


}


#endif