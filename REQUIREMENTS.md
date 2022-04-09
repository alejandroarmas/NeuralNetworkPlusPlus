

## Software Opportunity 

To create a runtime for training simple neural networks using multithreading architecture on x86.


### Memory Order for Matricies

*Requirement*: Matricies are laid out in Row-major order

*Description*: Provides excellent spatial locality for access patterns for i, j, k

*Priority*: 0 - Highest Priority

*Source*: Alejandro

*Justification*: Neccesary for high performance

*Metric*: Use cache simulator to measure last-level-cache miss rate to be <2%. 



### Optimal Matrix Multiplication

*Requirement*: Parallel Divide and Conquer Matrix Multiplication

*Description*: Leverages system cache locality and multicore processors with recursive MM algorithm and cilk runtime.   

*Priority*: 0 - Highest Priority

*Source*: Alejandro

*Justification*: Part of the core requirements to ship. 

*Metric*: 5000x5000 size matrix multiplication in < 4 sec.  


### Autodiff

*Requirement*: Automatic Differentiation via Jacobian-Vector-Product Backpropagation Engine

*Description*: Maintain a record of data (tensors) and all executed operations (along with the resulting new tensors) in a directed acyclic graph (DAG). 

*Priority*: 0 - Highest Priority

*Source*: Alejandro

*Justification*: Part of the core requirements to ship. 

*Metric*: Train an imagenet data set and achieve >90% accuracy. Training should perform reasonably similar to Pytorch or Tensorflow.  


### Automatic binding generation system

*Description*: Provides bindings of NN++ methods to Python and the command-line


### Docker Installation

*Description*: Installation of dependency via script and docker container. 
