#ifndef MATRIX_ALGORITHMS_UTILITIES_H
#define MATRIX_ALGORITHMS_UTILITIES_H


#include <algorithm>
#include <functional>
#include <exception>
#include <iostream>
#include <variant>


#include "matrix.h"
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

                    


                }


    }

}

#endif // MATRIX_ALGORITHMS_UTILITIES_H
