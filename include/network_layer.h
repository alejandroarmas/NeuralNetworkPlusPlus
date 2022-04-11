#ifndef NETWORK_LAYER_H
#define NETWORK_LAYER_H

#include <cstdint>
#include <memory>
#include <functional>

#include "functions.h"
#include "generator.h"
#include "matrix.h"

#include <map>

#define FLAT 1
    

namespace NeuralNetwork {


    class StepInterface {

        public:
            virtual ~StepInterface() = default;
            virtual std::unique_ptr<Matrix::Representation> forward(std::unique_ptr<Matrix::Representation> input) = 0;
            };


    template <class Implementation>
    class ComputationalStep: public StepInterface {

        public:
            std::unique_ptr<Matrix::Representation> forward(std::unique_ptr<Matrix::Representation> input) override { 

                    // TODO: print, or error checking.

                    return Impl().forward(std::move(input));
            }
            ~ComputationalStep() {}
        private:
            ComputationalStep& Impl() { return *static_cast<Implementation*>(this); }
            ComputationalStep() = default;
            friend Implementation;
    };


    class ComposedStep {

        public:
            virtual void add(std::unique_ptr<StepInterface> layer) = 0;
    };


    class BinaryOperationStep: public ComputationalStep<BinaryOperationStep> {

        public:
            BinaryOperationStep(Matrix::Rows _l, Matrix::Columns _w, 
                Matrix::Generation::Base& matrix_init) : 
                matrix(std::make_unique<Matrix::Representation>(_l, _w))
            {
                    
                    this->matrix = matrix_init(std::move(this->matrix));
            }
            std::unique_ptr<Matrix::Representation> giveOperand() { return std::move(this->matrix); }
        protected:
            std::unique_ptr<Matrix::Representation> matrix;

    };

    
    class MatrixMultiplyStep: public BinaryOperationStep {

        public:
            MatrixMultiplyStep(Matrix::Rows _l, Matrix::Columns _w, 
                Matrix::Generation::Base& matrix_init) : 
                BinaryOperationStep(_l, _w, matrix_init) {}
            std::unique_ptr<Matrix::Representation> forward(std::unique_ptr<Matrix::Representation> input) override;
    };
    
    
    class AddStep: public BinaryOperationStep {

        public:
            AddStep(Matrix::Columns _w, 
                Matrix::Generation::Base& matrix_init) : 
                BinaryOperationStep(Matrix::Rows(FLAT), _w, matrix_init) {}
            std::unique_ptr<Matrix::Representation> forward(std::unique_ptr<Matrix::Representation> input) override;
    };


    class Layer: public ComputationalStep<Layer>, ComposedStep {

        public:
            Layer(std::unique_ptr<StepInterface> _w, std::unique_ptr<StepInterface> _b) : 
                weights(std::move(_w)), bias(std::move(_b)) {}
                
            std::unique_ptr<Matrix::Representation> forward(std::unique_ptr<Matrix::Representation> input) override;
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
            
        using matrix_t = Matrix::Representation; 

        std::unique_ptr<matrix_t> ma = std::make_unique<matrix_t>(Matrix::Rows(1), Matrix::Columns(2000));
        Matrix::Generation::Normal<0, 1> normal_distribution_init;
        ma = normal_distribution_init(std::move(ma));

        NeuralNetwork::Sequential model;

        model.add(std::make_unique<NeuralNetwork::Layer>(
                std::make_unique<NeuralNetwork::MatrixMultiplyStep>(Matrix::Rows(2000), Matrix::Columns(1000), normal_distribution_init),
                std::make_unique<NeuralNetwork::AddStep>(Matrix::Columns(1000), normal_distribution_init)    
        ));
        model.add(std::make_unique<NeuralNetwork::ActivationFunctions::ReLU>());
        model.add(std::make_unique<NeuralNetwork::Layer>(
                std::make_unique<NeuralNetwork::MatrixMultiplyStep>(Matrix::Rows(1000), Matrix::Columns(10), normal_distribution_init),
                std::make_unique<NeuralNetwork::AddStep>(Matrix::Columns(10), normal_distribution_init)    
        ));
        model.add(std::make_unique<NeuralNetwork::ActivationFunctions::ReLU>());
        
        auto out = model.forward(std::move(ma));

    */
    class Sequential: public ComputationalStep<Sequential>, public ComposedStep {
        public:
            Sequential() : last_key(0) {}
            ~Sequential() = default;
            std::unique_ptr<Matrix::Representation> forward(std::unique_ptr<Matrix::Representation> input) override;
            void add(std::unique_ptr<StepInterface> layer) override;
        private:
            std::map<const unsigned int, std::unique_ptr<StepInterface>> _modules;
            unsigned int last_key;



    };



}



#endif // NETWORK_LAYER_H