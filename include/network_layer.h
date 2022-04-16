#ifndef NETWORK_LAYER_H
#define NETWORK_LAYER_H

#include "tensor.h"

#include <cstdint>
#include <memory>
#include <functional>
#include <map>

#include <iostream>

#define FLAT 1
    

namespace NeuralNetwork {

    // using T = std::shared_ptr<Tensor>;

    using namespace NeuralNetwork::Computation::Graph;

    class StepInterface {

        public:
            virtual ~StepInterface() = default;
            virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) = 0;
            };


    template <class Implementation>
    class ComputationalStep: public StepInterface {

        public:
            std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override { 

                    // TODO: print, or error checking.
                    std::cout << "Entered ComputationalStep()" << std::endl;

                    return Impl().doForward(input);
            }
            ~ComputationalStep() {}
        private:
            Implementation& Impl() { return *static_cast<Implementation*>(this); }
            // Implementation const& Impl() { return *static_cast<Implementation*>(this); }
            ComputationalStep() = default;
            // ComputationalStep(const ComputationalStep &) = default;
            friend Implementation;
    };



    /*

    DESCRIPTION:

        Wraps around a tensor object, and used in NN abstraction.

    */
    template <class Operation>
    class BinaryOperationStep: public ComputationalStep<BinaryOperationStep<Operation>> {

        public:
            BinaryOperationStep(Matrix::Rows _l, Matrix::Columns _w) : 
                matrix(std::make_shared<Tensor>(_l, _w, Computation::Graph::IsTrackable(true), Computation::Graph::IsLeaf(false))) {}
            std::shared_ptr<Tensor> doForward(std::shared_ptr<Tensor> input) { return Impl()._doForward(input);}
            std::shared_ptr<Tensor> releaseOperand() { return matrix; }
        protected:
            std::shared_ptr<Tensor> matrix;
        private:
            friend Operation;
            Operation& Impl() { return *static_cast<Operation*>(this); }

    };

    
    class MatrixMultiplyStep: public BinaryOperationStep<MatrixMultiplyStep> {

        public:
            MatrixMultiplyStep(Matrix::Rows _l, Matrix::Columns _w) : 
                BinaryOperationStep<MatrixMultiplyStep>(_l, _w) {}
            std::shared_ptr<Tensor> _doForward(std::shared_ptr<Tensor> input);
    };
    
    
    /*

        DESCRIPTION:

            Wrapper for bias term that is added during a perceptron.
    */

    class AddStep: public BinaryOperationStep<AddStep> {

        public:
            AddStep(Matrix::Columns _w) : 
                BinaryOperationStep<AddStep>(Matrix::Rows(FLAT), _w) {}
            std::shared_ptr<Tensor> _doForward(std::shared_ptr<Tensor> input);
    };

    /*
    -----------------------------------------------------------------
    */


    class ComposedStep {

        public:
            virtual void add(std::unique_ptr<StepInterface> layer) = 0;
    };


    class Layer: public ComputationalStep<Layer>, public ComposedStep {

        public:
            Layer(std::unique_ptr<StepInterface> _w, 
                std::unique_ptr<StepInterface> _b) : 
                weights(std::move(_w)), bias(std::move(_b)) {}
                
            std::shared_ptr<Tensor> doForward(std::shared_ptr<Tensor> input);
            void add(std::unique_ptr<StepInterface> layer) override;
            
        private:
            std::unique_ptr<StepInterface> weights;
            std::unique_ptr<StepInterface> bias;
    };


    /*
    DESCRIPTION:

        Responsible for creating a tree of layers to be computed and checking if we
        are allowed to compute in this order.

    USAGE:
            

        using matrix_t = NeuralNetwork::Computation::Graph::Tensor; 

        std::shared_ptr<matrix_t> ma = std::make_shared<matrix_t>(Matrix::Rows(1), Matrix::Columns(2000));


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
        model.add(std::make_unique<NeuralNetwork::ActivationFunctions::ReLU>());
        

        auto out = model.forward(ma);

    */
    class Sequential: public ComputationalStep<Sequential>, public ComposedStep {
        public:
            Sequential() : last_key(0) {}
            ~Sequential() = default;
            std::shared_ptr<Tensor> doForward(std::shared_ptr<Tensor> input);
            void add(std::unique_ptr<StepInterface> layer) override;
        private:
            std::map<const unsigned int, std::unique_ptr<StepInterface>> _modules;
            unsigned int last_key;



    };



}



#endif // NETWORK_LAYER_H