#include <iostream>
#include <iomanip>


#include "mm.h"
#include "matrix_printer.h"

#define MAX_PRINT_WIDTH 5
#define MAX_PRINT_LENGTH 5


void Matrix::Printer::operator()(Matrix::Representation m) {

    bool valid_column_print = true;
    bool reached_end_row = true;
    bool before_max_width = true;
    bool last_column_val = true;
    
    u_int64_t n_cols = m.num_cols();
    u_int64_t n_rows = m.num_rows();
    uint64_t total_iter = n_rows * n_cols; 

    std::cout << "Matrix[R=" << n_rows << "][C=" << n_cols << "]:"; 

    std::cout << std::setprecision(2);

    for (u_int64_t i = 0; i < total_iter; i++) {

        reached_end_row = i % n_cols == 0;
        before_max_width = i % n_cols < MAX_PRINT_WIDTH;
        last_column_val = i % n_cols == n_cols - 1;
        

        if (reached_end_row) std::cout << '\n';
        std::cout << std::setw(6) << std::right;
        
        if (before_max_width || last_column_val) {
            
            std::cout << m.get(i / n_cols, i % n_cols) << " ";
            valid_column_print = true;
        }
        else if (!valid_column_print) {}
        else {
            std::cout << "... ";
            valid_column_print = false;
        }

    }

    std::cout << std::endl << std::endl;
}