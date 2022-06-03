# Wirikuta

### What is Wirikuta?

Wirikuta is a high-performance Automatic Differentiation Engine used to build Deep neural networks. 

According to Huichol symbology, Wirikuta is the land where life began and the religous site embarked for pilgrimage, where the Huichol peoples gather Peyote. Peyote is a psychoactive cactus providing acquisition of shamanic powers, which provide insights to the metaphysical world. In that same vein, information processing through deep learning models and data, elucidate the physical one we reside. I hope you find this code useful.   

```c++
    constexpr double LEARNING_RATE = 0.001; 
    constexpr double TRAINING_EPOCS = 300; 

    auto ma = NeuralNetwork::Computation::Graph::TensorConstructor::create(
            Matrix::Rows(1), 
            Matrix::Columns(2000));
    
    auto ground_truth = NeuralNetwork::Computation::Graph::TensorConstructor::create(
            Matrix::Rows(1), 
            Matrix::Columns(10));
    

    NeuralNetwork::Sequential model;

    model.add(std::make_unique<NeuralNetwork::Layer>(
            std::make_unique<NeuralNetwork::MatrixMultiplyStep>(Matrix::Rows(2000), Matrix::Columns(1000)),
            std::make_unique<NeuralNetwork::AddStep>(Matrix::Columns(1000))    
    ));
    model.add(std::make_unique<NeuralNetwork::ActivationFunctions::ReLU>());
    model.add(std::make_unique<NeuralNetwork::Layer>(
            std::make_unique<NeuralNetwork::MatrixMultiplyStep>(Matrix::Rows(1000), Matrix::Columns(10)),
            std::make_unique<NeuralNetwork::AddStep>(Matrix::Columns(10))    
    ));
    
    auto CE = NeuralNetwork::Computation::Graph::TensorOp(Matrix::Operations::Metric::CrossEntropy{});

    for (int i = 0; i < TRAINING_EPOCS; i++) {
        auto out  = model.forward(ma);
        auto loss = CE(ground_truth, out);        
        loss->backwards();

        for (auto it = loss->parameters().begin(); it != loss->parameters().end(); ++it) {
            *it += -LEARNING_RATE * it.gradient();
        }

    }

```

### Getting Started

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