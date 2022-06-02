#ifndef TENSOR_DEFINITION_H
#define TENSOR_DEFINITION_H

#include "computational_graph_map.h"
#include "strong_types.h"

#include "matrix.h"
#include "m_algorithms.h"
#include "m_algorithms_register.h"
#include "m_algorithms_concepts.h"
#include "function_object_iterator.h"


#include <chrono>
#include <optional>
#include <memory>
#include <stack>

namespace NeuralNetwork {

    namespace Computation {

        namespace Graph {


            class MatrixParameter {

                using iterator = LevelOrderIterator<ReadParameterPolicy>; 

                public:
                    explicit MatrixParameter(TensorID _tid) : id(_tid) {}

                    iterator begin() noexcept {
                            return iterator{id}; 
                        }

                    iterator end() noexcept {
                        return iterator{TensorID(0)}; 
                    }
                private:
                    TensorID id;
                

            };



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



            class Tensor {

                using matrix_t = Matrix::Representation;
                using iterator = LevelOrderIterator<ComputeGradientPolicy>; 

                public:
                    ~Tensor() noexcept {}
                    explicit Tensor(Matrix::Rows _l, Matrix::Columns _w, 
                        IsTrackable _t  = IsTrackable(true), 
                        IsLeaf _f       = IsLeaf(true),
                        IsRecordable _r = IsRecordable(true)) noexcept;
                                                            
                    explicit Tensor(const Matrix::Representation& _m, 
                        IsTrackable _t  = IsTrackable(true), 
                        IsLeaf _f       = IsLeaf(true),
                        IsRecordable _r = IsRecordable(true)) noexcept;

                    explicit Tensor(const Tensor& other) noexcept;

                    Tensor& operator=(const Tensor& other) noexcept;

                    friend class TensorConstructor;
                    
                    void backwards() noexcept;

                    bool is_requires_grad() const noexcept;
                    bool is_tensor_leaf() const noexcept;
                    bool is_recorded() const noexcept;

                    void become_parent() noexcept;

                    matrix_t& release_matrix() noexcept;
                    matrix_t& get_grad() noexcept;

                    Matrix::Rows num_rows(void) const noexcept;
                    Matrix::Columns num_cols(void) const noexcept;

                    std::optional<TensorStatistics> stats;

                    FunctionObject get_operation() noexcept;
                    TensorID get_tensor_id() const noexcept { return my_tensor_id; }
                    void detatch_from_computational_graph() noexcept;

                    iterator begin() noexcept {
                        return iterator{my_tensor_id}; 
                    }

                    iterator end() noexcept {
                        return iterator{TensorID(0)}; 
                    }

                    MatrixParameter parameters() noexcept {
                        return MatrixParameter{my_tensor_id}; 
                    }

                private:
                    matrix_t matrix;
                    matrix_t grad;
                    TensorID my_tensor_id;
                    bool is_leaf;
                    bool requires_grad;
                    bool record_statistics;

            };


            
        }

    }

}


#endif // TENSOR_DEFINITION_H