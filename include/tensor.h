#ifndef TENSOR_DEFINITION_H
#define TENSOR_DEFINITION_H


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

            class RegisteredOperation;

            class TensorStatistics {
                

                using time_t = std::chrono::steady_clock::time_point;

                public:

                    constexpr void set_operation_code(Matrix::Operations::Code _c) { op_type = _c; }
                    constexpr Matrix::Operations::Code get_operation_code() const { return op_type; }
                    
                    constexpr void set_operation_string(const std::string_view& _s) { op_string = _s; }
                    constexpr std::string_view get_operation_string() const { return op_string; }

                    constexpr void set_graph_start( time_t _t) { tensor_graph_t1     = _t; }
                    constexpr void set_graph_end(   time_t _t) { tensor_graph_t2     = _t; }
                    constexpr void set_matrix_start(time_t _t) { matrix_operation_t1 = _t; }
                    constexpr void set_matrix_end(  time_t _t) { matrix_operation_t2 = _t; }
                    
                    constexpr time_t get_graph_start( ) const { return tensor_graph_t1    ; }
                    constexpr time_t get_graph_end(   ) const { return tensor_graph_t2    ; }
                    constexpr time_t get_matrix_start() const { return matrix_operation_t1; }
                    constexpr time_t get_matrix_end(  ) const { return matrix_operation_t2; }

                    
                private:

                    Matrix::Operations::Code op_type;
                    std::string_view op_string;

                    time_t tensor_graph_t1;
                    time_t tensor_graph_t2;
                    time_t matrix_operation_t1;
                    time_t matrix_operation_t2;
            };

            using IsTrackable  = 
                Matrix::NamedType<bool, struct TrackParameter>;
            
            using IsLeaf       = 
                Matrix::NamedType<bool, struct LeafParameter>;
            
            using IsRecordable = 
                Matrix::NamedType<bool, struct RecordParameter>;


            class Tensor {

                using matrix_t = std::unique_ptr<Matrix::Representation>;

                public:
                    Tensor(Matrix::Rows _l, Matrix::Columns _w, 
                        IsTrackable _t  = IsTrackable(true), 
                        IsLeaf _f       = IsLeaf(true),
                        IsRecordable _r = IsRecordable(true));
                                                            
                    Tensor(std::unique_ptr<Matrix::Representation> _m, 
                        IsTrackable _t  = IsTrackable(true), 
                        IsLeaf _f       = IsLeaf(true),
                        IsRecordable _r = IsRecordable(true));


                    bool     is_tensor_leaf()   { 
                        return is_leaf;           }

                    bool     is_requires_grad() { 
                        return requires_grad;     }

                    bool     is_recorded()      { 
                        return record_statistics; }

                    matrix_t release_matrix()   { 
                        return std::move(matrix); }

                    matrix_t release_grad()     { 
                        return std::move(grad);   }
                    
                    void     register_operation(
                        const std::shared_ptr<
                        RegisteredOperation> _node) { 
                            graph_node = _node;   }

                    std::shared_ptr<RegisteredOperation> 
                        get_operation() {
                         return graph_node;       }
                         
                    std::optional<TensorStatistics> stats;
                protected:
                    void register_leaf_op(void);
                private:
                    matrix_t matrix;
                    matrix_t grad;
                    std::shared_ptr<RegisteredOperation> graph_node;
                    bool is_leaf;
                    bool requires_grad;
                    bool record_statistics;

            };



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


#endif // TENSOR_DEFINITION_H