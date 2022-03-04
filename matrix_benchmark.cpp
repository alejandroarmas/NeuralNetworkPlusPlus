#include <iostream>
#include <time.h>

#include <cilk/cilk.h>
#include <cilk/cilkscale.h>

#include "matrix_benchmark.h"


namespace Matrix {



    std::unique_ptr<Representation> Timer::operator()(std::unique_ptr<Matrix::Representation>& l, 
            std::unique_ptr<Matrix::Representation>& r) {


                struct timespec start, end;

                clock_gettime(CLOCK_MONOTONIC, &start);
                std::unique_ptr<Matrix::Representation> mc = matrix_operation->operator()(l, r);
                clock_gettime(CLOCK_MONOTONIC, &end);

                double tdiff = (end.tv_sec - start.tv_sec) + 1e-9*(end.tv_nsec - start.tv_nsec);

                std::cout << "Performed in: " << tdiff << " Seconds." << std::endl;

                return mc;
            }

// #ifdef CILKSCALE
    /*
    Cilkscale's command-line output includes work and span measurements for the Cilk program in terms of empirically measured times.
    Parallelism measurements are derived from these times.
    A simple struct wsp_t contains the number of nanoseconds for work and span. This data is collected immediately before and after 
    the wrapped function's execution. Then these two measurements are subtracted and dumped to stdout in CSV format, with the first 
    column being the label of the measurement. At the end, the same measurements are output for the program as a whole with an 
    empty label. The final measurement includes all the setup and teardown code, which pollutes the measurement we are interested in.
    Because the dump to stdout can interleave with other program output, you might want to set the environment variable 
    CILKSCALE_OUT="filename.csv" to redirect Cilkscale output to a specific file (you will only be able to access that file when 
    running Cilkscale instrumented programs locally --- awsrun.py currently doesn't return output files).

    In addition to a span column, you are also seeing a "burdened span" column. Burdened span accounts for the worst possible 
    migration overhead, which can come from work-stealing and other factors.
    */
    std::unique_ptr<Representation> ParallelMeasurer::operator()(std::unique_ptr<Matrix::Representation>& l, 
            std::unique_ptr<Matrix::Representation>& r) {

                wsp_t start_wsp, stop_wsp;
                
                start_wsp = wsp_getworkspan();
                std::unique_ptr<Matrix::Representation> mc = matrix_operation->operator()(l, r);
                stop_wsp = wsp_getworkspan();

                wsp_dump(wsp_sub(stop_wsp, start_wsp), "Cilkscale Parallel Measurement:");
                
                return mc;
            }
// #endif

}
