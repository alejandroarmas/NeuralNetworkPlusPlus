#ifndef TENSOR_FORWARD_WRAPPER
#define TENSOR_FORWARD_WRAPPER


#include "tensor.h"
#include "matrix.h"
#include "m_algorithms.h"
#include "m_algorithms_register.h"
#include "m_algorithms_concepts.h"
#include "matrix_benchmark.h"

#include <chrono>
#include <optional>
#include <memory>



namespace NeuralNetwork {

    namespace Computation {

        namespace Graph {


            /*

            DESCRIPTION:

                Functor follows 'Strategy' behavioral pattern for defining a family
                of functions on the permutations of either benchmarking (or not) an
                operation as well as the operation being binary (or unary).


            USAGE:

                TensorOp mm(std::make_unique<
                Matrix::Operations::Binary::Multiplication::ParallelDNC>());

                auto out = mm(input, this->matrix);


            */

            
            template <Matrix::Operations::MatrixOperatable Operator>
            class TensorOp {

                public:
                    TensorOp(const Operator& _op) : op_type(_op) {}

                    std::shared_ptr<Tensor> operator()(
                        const std::shared_ptr<Tensor> l, 
                        const std::shared_ptr<Tensor> r = nullptr);
                private:
                    Operator op_type; 
            };


            class ComputeTag;
            class RecordTag;


            class PerformTensorStrategy {

                public:
                    PerformTensorStrategy() = default;

                    template <Matrix::Operations::MatrixOperatable Operator>
                    std::shared_ptr<Tensor> compute(
                        Operator _op, 
                        const std::shared_ptr<Tensor> l,
                        const std::shared_ptr<Tensor> r, 
                        ComputeTag _);

                    template <Matrix::Operations::MatrixOperatable Operator>
                    std::shared_ptr<Tensor> compute(
                        Operator _op, 
                        const std::shared_ptr<Tensor> l,
                        const std::shared_ptr<Tensor> r, 
                        RecordTag _);

            };
            

            
            /*

            DESCRIPTION:

                Curiosily recurring Template Pattern for 
                accepting Strategy implementation visitor

            USAGE:

                  if (recordTensorOperation && isBinaryOp) {
                        
                        RecordBinaryTag _;

                        return _.compute_tensor(std::move(op_type), l, r, implementation);
                  }

            */
            template <class StrategyType>
            struct StrategyTag {

                template <Matrix::Operations::MatrixOperatable Operator>
                std::shared_ptr<Tensor> compute_tensor(
                    Operator _op,
                    const std::shared_ptr<Tensor> l, 
                    const std::shared_ptr<Tensor> r,
                    PerformTensorStrategy& strat_implementation) {
                        
                    return strat_implementation.compute(
                        _op, l, r, *static_cast<
                        StrategyType const*>(this));
            } };

            class ComputeTag : public StrategyTag<ComputeTag> {
                public:
                    ComputeTag() = default;
            };
            class RecordTag : public StrategyTag<RecordTag> {
                public:
                    RecordTag() = default;
            };




        }

    }

}


#endif // TENSOR_FORWARD_WRAPPER