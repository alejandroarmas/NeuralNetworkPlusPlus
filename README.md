# NeuralNetworkPlusPlus
Implementation of a Multilayer Perceptron from Scratch. 


# Requirements

This project utilizes efficient parallel code via **OpenCilk** for shared-multicore machines. This means you need an extention to the LLVM compiler via Tapir. This means having llvm intermediate representations of logically parallel tasks for effective compiler optimizations [1]. The Cilk scheduler then decides at runtime how to schedule and execute logically parallel tasks onto parallel processors in a provably efficient schedule.


```
    wget https://github.com/OpenCilk/opencilk-project/releases/download/opencilk%2Fv1.0/OpenCilk-1.0-LLVM-10.0.1-Ubuntu-20.04-x86_64.tar.gz
    tar -zvxf ./OpenCilk-1.0-LLVM-10.0.1-Ubuntu-20.04-x86_64.tar.gz
    mv ./OpenCilk-10.0.1-Linux ./OpenCilk
    rm -rf ./OpenCilk-1.0-LLVM-10.0.1-Ubuntu-20.04-x86_64.tar.gz
    LLVM_BIN_PATH=$PWD/OpenCilk/bin
    LLVM_LIB_PATH=$PWD/OpenCilk/lib
    echo "export PATH=$PATH:$LLVM_BIN_PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LLVM_LIB_PATH" >> ~/.bashrc
```


### Highlights

Achieves a 5000x5000 dense Matrix Multiplication in 6.33143 Seconds using `Matrix::Operations::Multiplication::ParallelDNC`, compared the naive implementation `Matrix::Operations::Multiplication::Naive` of 898.936 Seconds on an M1 Max. That's a 141.9799x speedup! 

M1 max has theoretical maximum of 239.616 GFLOPS.

(3.228 GHz x 8 cores + 2 x 2.064 GHz) x 8 instructions/cycle = 239.616 GFLOPS




[1]: [Schardl, T. B., Moses, W. S., & Leiserson, C. E. (2017). Tapir: Embedding Fork-Join Parallelism into LLVM’s Intermediate Representation. PPoPP, 249–265. https://doi.org/10.1145/3018743.3018758](https://doi.org/10.1145/3018743.3018758)