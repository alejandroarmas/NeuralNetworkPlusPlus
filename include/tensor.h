#ifndef TENSOR_DEFINITION_H
#define TENSOR_DEFINITION_H


#include "matrix.h"
#include "m_algorithms.h"
#include "m_algorithms_register.h"
#include "m_algorithms_concepts.h"

#include <chrono>
#include <optional>
#include <memory>

namespace NeuralNetwork {

    namespace Computation {

        namespace Graph {

            using IsTrackable  = 
                Matrix::NamedType<bool, struct TrackParameter>;


            using IsLeaf       = 
                Matrix::NamedType<bool, struct LeafParameter>;


            using IsRecordable = 
                Matrix::NamedType<bool, struct RecordParameter>;


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



            class Tensor {

                using matrix_t = Matrix::Representation;

                public:
                    Tensor(Matrix::Rows _l, Matrix::Columns _w, 
                        IsTrackable _t  = IsTrackable(true), 
                        IsLeaf _f       = IsLeaf(true),
                        IsRecordable _r = IsRecordable(true));
                                                            
                    Tensor(const Matrix::Representation& _m, 
                        IsTrackable _t  = IsTrackable(true), 
                        IsLeaf _f       = IsLeaf(true),
                        IsRecordable _r = IsRecordable(true));

                    void backwards();

                    bool     is_tensor_leaf()   
                        { return is_leaf;           }

                    bool     is_requires_grad()
                        { return requires_grad; }

                    bool     is_recorded()      
                        { return record_statistics; }

                    matrix_t release_matrix()   
                        { return matrix; }

                    matrix_t release_grad()     
                        { return grad.value_or(Matrix::Representation()); }
                    
                    void     register_operation(
                        const std::shared_ptr<
                        RegisteredOperation> _node) { 
                            graph_node = _node;   }

                    std::shared_ptr<RegisteredOperation> 
                        get_operation() {
                         return graph_node;       }
                         
                    std::optional<TensorStatistics> stats;
                private:
                    matrix_t matrix;
                    std::optional<matrix_t> grad;
                    std::shared_ptr<RegisteredOperation> graph_node;
                    bool is_leaf;
                    bool requires_grad;
                    bool record_statistics;

            };


            
        }

    }

}


#endif // TENSOR_DEFINITION_H