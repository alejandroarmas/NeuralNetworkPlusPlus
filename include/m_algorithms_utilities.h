#ifndef MATRIX_ALGORITHMS_UTILITIES_H
#define MATRIX_ALGORITHMS_UTILITIES_H


#include <algorithm>
#include <chrono>
#include <functional>
#include <exception>
#include <iostream>
#include <variant>


#include "matrix.h"
#include "tensor.h"
#include "m_algorithms.h"


namespace Matrix {

    namespace Operations {

                
        namespace Utility {

                    

                    std::string debug_message(const std::unique_ptr<Matrix::Representation>& l, 
                                            const std::unique_ptr<Matrix::Representation>& r);
                    
                    std::string debug_message_2(const std::unique_ptr<Matrix::Representation>& l, 
                                            const std::unique_ptr<Matrix::Representation>& r);
                    


                    /*

                        DESCRIPTION:
                            Returns a constant contiguous sequence of characters representing
                            the operation type in question

                    */
                    struct Stringify {
                            constexpr std::string_view operator()(
                                const Unary::ReLU& _) { 
                                    return "ReLU"; }
                            constexpr std::string_view operator()(
                                const Binary::HadamardProduct::Std& _) { 
                                    return "HadamardProduct"; }
                            constexpr std::string_view operator()(
                                const Binary::Multiplication::ParallelDNC& _) { 
                                    return "MatrixMultiply"; }
                            constexpr std::string_view operator()(
                                const Binary::Multiplication::Naive& _) { 
                                    return "MatrixMultiply"; }
                            constexpr std::string_view operator()(
                                const Binary::Multiplication::Square& _) { 
                                    return "MatrixMultiply"; }
                            constexpr std::string_view operator()(
                                const Binary::Addition::Std& _) { 
                                    return "Addition"; }
                            constexpr std::string_view operator()(
                                const Binary::OuterProduct::Naive& _) { 
                                    return "OuterProduct"; }
                    };

                    struct Codify {
                            constexpr Code operator()(
                                const Unary::ReLU& _)                         
                                { return Code::ReLU; }
                            constexpr Code operator()(
                                const Binary::HadamardProduct::Std& _)        
                                { return Code::HADAMARD; }
                            constexpr Code operator()(
                                const Binary::Multiplication::ParallelDNC& _) 
                                { return Code::MULTIPLY; }
                            constexpr Code operator()(
                                const Binary::Multiplication::Naive& _)       
                                { return Code::MULTIPLY; }
                            constexpr Code operator()(
                                const Binary::Multiplication::Square& _)      
                                { return Code::MULTIPLY; }
                            constexpr Code operator()(
                                const Binary::Addition::Std& _)               
                                { return Code::PLUS; }
                            constexpr Code operator()(
                                const Binary::OuterProduct::Naive& _)         
                                { return Code::OUTER_PRODUCT; }
                    };



                    /*

                    DESCRIPTION:
                        
                        Justifyication of duck typing because each object is same size.

                    */
                    struct Function {
                        
                        [[nodiscard]] static auto from(const Code& _c) {
                            
                            std::variant<
                                    Unary::ReLU, 
                                    Binary::HadamardProduct::Std, 
                                    Binary::Multiplication::ParallelDNC,
                                    Binary::Addition::Std,
                                    Binary::OuterProduct::Naive
                                > output;
                                
                            switch (_c) {
                                case Code::ReLU:
                                    output = Unary::ReLU();
                                    break;
                                case Code::HADAMARD:
                                    output = Binary::HadamardProduct::Std();
                                    break;

                                case Code::MULTIPLY:
                                    output = Binary::Multiplication::ParallelDNC();
                                    break;
                                    
                                case Code::PLUS:
                                    output = Binary::Addition::Std();
                                    break;

                                case Code::OUTER_PRODUCT:
                                    output = Binary::OuterProduct::Naive();
                                    break;
                                default:
                                    throw std::invalid_argument("Use valid Code::");

                            }

                            return output;
                        }



                    };

                    

                    class PrintTensorStats {

                        public:
                             void operator()(
                                const NeuralNetwork::Computation::Graph::TensorStatistics& stats) {

                                    auto m2 = stats.get_matrix_end();
                                    auto m1 = stats.get_matrix_start();

                                    auto g2 = stats.get_graph_end();
                                    auto g1 = stats.get_graph_start();

                                    auto op_str = stats.get_operation_string();


                                    auto time_performing_operation = std::chrono::duration_cast<std::chrono::duration<int, std::micro>>(m2 - m1).count(); 
                                    auto time_making_graph = std::chrono::duration_cast<std::chrono::duration<int, std::micro>>(g2 - g1).count() - time_performing_operation;
                                    
                                    std::cout << op_str << " performance: " << std::endl;
                                    std::cout << "\t Time making graph (ms): " << time_making_graph << std::endl;
                                    std::cout << "\t Time performing operation (ms): " << time_performing_operation << std::endl;

                           

                            }


                    };



                }


    }

}

#endif // MATRIX_ALGORITHMS_UTILITIES_H
