#ifndef TENSOR_FUNCTION_OBJECT_H
#define TENSOR_FUNCTION_OBJECT_H

#include "m_algorithms_concepts.h"
#include "m_algorithms_register.h"

#include <array>
#include <optional>

namespace NeuralNetwork {

    namespace Computation {

        namespace Graph {


            namespace States {

                struct TensorRegistered : public RegisteredTensor {
                    public:
                        void is_state(void) {}
                        TensorRegistered(RegisteredTensor tid) : RegisteredTensor(tid) {}
                };
                struct UnaryRegistered : public RegisteredUnaryOperation {
                    public:
                        void is_state(void) {}
                        UnaryRegistered(RegisteredUnaryOperation unary_mapping) : RegisteredUnaryOperation(unary_mapping) {}
                };
                struct BinaryRegistered : public RegisteredBinaryOperation{
                    public:
                        void is_state(void) {}
                        BinaryRegistered(RegisteredBinaryOperation binary_mapping) : RegisteredBinaryOperation(binary_mapping) {}
                };

                /* -------------------------------------------------- */

                struct NoOperation : public TensorRegistered {
                    NoOperation(RegisteredTensor tid) : TensorRegistered(tid) {}
                    NoOperation(NoOperation&) = default; 
                    NoOperation(NoOperation&&) = default; 
                    NoOperation& operator=(const NoOperation&) = default; 
                    NoOperation& operator=(NoOperation&&) = default; 
                };
                static_assert(NoOperandRegistry<NoOperation>);


                struct Invalidated {
                    public:
                        void is_state(void) {}
                };
                static_assert(IsStateFull<Invalidated>);

                struct ErrorOccurred {
                    public:
                        void is_state(void) {}
                };
                static_assert(IsStateFull<ErrorOccurred>);

                struct MatrixMultiply : public BinaryRegistered {
                    public:
                        void is_state(void) {}
                    MatrixMultiply(BinaryRegistered other) : BinaryRegistered(other) {}
                    MatrixMultiply(MatrixMultiply&) = default; 
                    MatrixMultiply(MatrixMultiply&&) = default; 
                    MatrixMultiply& operator=(const MatrixMultiply&) = default; 
                    MatrixMultiply& operator=(MatrixMultiply&&) = default; 
                };

                static_assert(BinaryRegistry<MatrixMultiply>);

                struct Plus : public BinaryRegistered  {
                    Plus(BinaryRegistered other) : BinaryRegistered(other) {}
                    Plus(Plus&) = default; 
                    Plus(Plus&&) = default; 
                    Plus& operator=(const Plus&) = default; 
                    Plus& operator=(Plus&&) = default; 
                };

                static_assert(BinaryRegistry<Plus>);

                struct Minus : public BinaryRegistered  {
                    Minus(BinaryRegistered other) : BinaryRegistered(other) {}
                    Minus(Minus&) = default; 
                    Minus(Minus&&) = default; 
                    Minus& operator=(const Minus&) = default; 
                    Minus& operator=(Minus&&) = default; 
                };
                static_assert(BinaryRegistry<Minus>);


                struct ReLU : public UnaryRegistered {
                    ReLU(UnaryRegistered other) : UnaryRegistered(other) {}
                    ReLU(ReLU&) = default; 
                    ReLU(ReLU&&) = default; 
                    ReLU& operator=(const ReLU&) = default; 
                    ReLU& operator=(ReLU&&) = default; 
                };
                static_assert(UnaryRegistry<ReLU>);


                struct SoftMax : public UnaryRegistered {
                    SoftMax(UnaryRegistered other) : UnaryRegistered(other) {}
                    SoftMax(SoftMax&) = default; 
                    SoftMax(SoftMax&&) = default; 
                    SoftMax& operator=(const SoftMax&) = default; 
                    SoftMax& operator=(SoftMax&&) = default; 
                };
                static_assert(UnaryRegistry<SoftMax>);


                struct OuterProduct : public BinaryRegistered  {
                    OuterProduct(BinaryRegistered other) : BinaryRegistered(other) {}
                    OuterProduct(OuterProduct&) = default; 
                    OuterProduct(OuterProduct&&) = default; 
                    OuterProduct& operator=(const OuterProduct&) = default; 
                    OuterProduct& operator=(OuterProduct&&) = default; 

                };
                static_assert(BinaryRegistry<OuterProduct>);


                struct Hadamard : public BinaryRegistered  {
                    Hadamard(BinaryRegistered other) : BinaryRegistered(other) {}
                    Hadamard(Hadamard&) = default; 
                    Hadamard(Hadamard&&) = default; 
                    Hadamard& operator=(const Hadamard&) = default; 
                    Hadamard& operator=(Hadamard&&) = default; 
                };
                static_assert(BinaryRegistry<Hadamard>);


                struct CrossEntropy : public BinaryRegistered  {
                    CrossEntropy(BinaryRegistered other) : BinaryRegistered(other) {}
                    CrossEntropy(CrossEntropy&) = default; 
                    CrossEntropy(CrossEntropy&&) = default; 
                    CrossEntropy& operator=(const CrossEntropy&) = default; 
                    CrossEntropy& operator=(CrossEntropy&&) = default; 
                };
                static_assert(BinaryRegistry<CrossEntropy>);


            } // States

            namespace Events {


                template <Matrix::Operations::MatrixOperatable RegisteryType>
                struct InstantiateTrait {};


                template <Matrix::Operations::UnaryMatrixOperatable RegisteryType>
                struct InstantiateTrait<RegisteryType> {
                    using Type = RegisteredUnaryOperation;
                };

                template <Matrix::Operations::BinaryMatrixOperatable RegisteryType>
                struct InstantiateTrait<RegisteryType>{
                    using Type = RegisteredBinaryOperation;
                };


                template <Matrix::Operations::MatrixOperatable RegisteryType>
                struct Instantiate {

                    using TensorRegistry = typename InstantiateTrait<RegisteryType>::Type; 

                    public:
                        Instantiate(RegisteryType _op, TensorRegistry _pl) : _operation(_op), _payload(_pl) {}
                        RegisteryType _operation;
                        TensorRegistry _payload;
                };
                

                struct Differentiate {};

            } // Events


            struct EventTrait {
                using Event = std::variant
                    <
                        NeuralNetwork::Computation::Graph::Events::Instantiate<Matrix::Operations::Unary::ReLU>,
                        NeuralNetwork::Computation::Graph::Events::Instantiate<Matrix::Operations::Unary::SoftMax>,
                        NeuralNetwork::Computation::Graph::Events::Instantiate<Matrix::Operations::Binary::HadamardProduct::Std>,
                        NeuralNetwork::Computation::Graph::Events::Instantiate<Matrix::Operations::Binary::Multiplication::ParallelDNC>,
                        NeuralNetwork::Computation::Graph::Events::Instantiate<Matrix::Operations::Binary::Multiplication::Naive>,
                        NeuralNetwork::Computation::Graph::Events::Instantiate<Matrix::Operations::Binary::Multiplication::Square>,
                        NeuralNetwork::Computation::Graph::Events::Instantiate<Matrix::Operations::Binary::Addition::Std>,
                        NeuralNetwork::Computation::Graph::Events::Instantiate<Matrix::Operations::Binary::OuterProduct::Naive>,
                        NeuralNetwork::Computation::Graph::Events::Instantiate<Matrix::Operations::Metric::CrossEntropy>
                    >;
            };

            struct StateTrait {
                using State = std::variant
                    <
                        // Binary Operations
                        States::NoOperation,
                        States::Invalidated,
                        States::ErrorOccurred,
                        States::MatrixMultiply,
                        States::Plus,
                        States::Minus,
                        States::OuterProduct,
                        States::Hadamard,
                        // Unary Operations
                        States::ReLU,
                        States::SoftMax,
                        // Metrics
                        States::CrossEntropy
                    >;
            };


            class OperationTransitioner {

                using State = StateTrait::State;

                public:
                    template <Matrix::Operations::MatrixOperatable RegisteryType>
                    State operator()(States::NoOperation nop, Events::Instantiate<RegisteryType> i) {
                        return on_event(nop, i);
                    }


                    /*
                        Default Case. Return error.
                    */
                    template <typename UnspecifiedState, typename UnspecifiedEvent>
                    State operator()(UnspecifiedState s, UnspecifiedEvent) {
                        return States::ErrorOccurred{};
                    }

                private:   
                    // static State on_event(States::CrossEntropy ce, Events::Differentiate df) {
                    //     return States::Invalidated{};
                    // }


                    template <Matrix::Operations::BinaryMatrixOperatable RegisteryType>
                    requires Same_as<RegisteryType, Matrix::Operations::Binary::Multiplication::ParallelDNC> ||
                        Same_as<RegisteryType, Matrix::Operations::Binary::Multiplication::Naive> ||
                        Same_as<RegisteryType, Matrix::Operations::Binary::Multiplication::Square>
                    static State on_event(States::NoOperation, Events::Instantiate<RegisteryType> i) {
                            return States::MatrixMultiply{i._payload};
                    }

                    template <Matrix::Operations::BinaryMatrixOperatable RegisteryType>
                    requires Same_as<RegisteryType, Matrix::Operations::Binary::Addition::Std>
                    static State on_event(States::NoOperation, Events::Instantiate<RegisteryType> i) {
                            return States::Plus{i._payload};
                    }

                    template <Matrix::Operations::BinaryMatrixOperatable RegisteryType>
                    requires Same_as<RegisteryType, Matrix::Operations::Binary::Subtraction::Std>
                    static State on_event(States::NoOperation, Events::Instantiate<RegisteryType> i) {
                            return States::Minus{i._payload};
                    }

                    template <Matrix::Operations::BinaryMatrixOperatable RegisteryType>
                    requires Same_as<RegisteryType, Matrix::Operations::Binary::OuterProduct::Naive>
                    static State on_event(States::NoOperation, Events::Instantiate<RegisteryType> i) {
                            return States::OuterProduct{i._payload};
                    }

                    template <Matrix::Operations::BinaryMatrixOperatable RegisteryType>
                    requires Same_as<RegisteryType, Matrix::Operations::Binary::HadamardProduct::Naive> ||
                        Same_as<RegisteryType, Matrix::Operations::Binary::HadamardProduct::Std>
                    static State on_event(States::NoOperation, Events::Instantiate<RegisteryType> i) {
                            return States::Hadamard{i._payload};
                    }
                    
                    template <Matrix::Operations::BinaryMatrixOperatable RegisteryType>
                    requires Same_as<RegisteryType, Matrix::Operations::Metric::CrossEntropy>
                    static State on_event(States::NoOperation, Events::Instantiate<RegisteryType> i) {
                            return States::CrossEntropy{i._payload};
                    }

                    
                    template <Matrix::Operations::UnaryMatrixOperatable RegisteryType>
                    requires Same_as<RegisteryType, Matrix::Operations::Unary::ReLU>
                    static State on_event(States::NoOperation, Events::Instantiate<RegisteryType> i) {
                            return States::ReLU{i._payload};
                    }

                    template <Matrix::Operations::UnaryMatrixOperatable RegisteryType>
                    requires Same_as<RegisteryType, Matrix::Operations::Unary::SoftMax>
                    static State on_event(States::NoOperation, Events::Instantiate<RegisteryType> i) {
                            return States::SoftMax{i._payload};
                    }

                    

            };


            class FunctionObjectInfo {

                public:
                    std::string_view operator()(States::NoOperation){
                        return "States::NoOperation";
                    }
                    std::string_view operator()(States::CrossEntropy){
                        return "States::CrossEntropy";
                    }
                    std::string_view operator()(States::ErrorOccurred){
                        return "States::ErrorOccurred";
                    }
                    std::string_view operator()(States::Hadamard){
                        return "States::Hadamard";
                    }
                    std::string_view operator()(States::MatrixMultiply){
                        return "States::MatrixMultiply";
                    }
                    std::string_view operator()(States::Minus){
                        return "States::Minus";
                    }
                    std::string_view operator()(States::Plus){
                        return "States::Plus";
                    }
                    std::string_view operator()(States::ReLU){
                        return "States::ReLU";
                    }
                    std::string_view operator()(States::SoftMax){
                        return "States::SoftMax";
                    }
                    std::string_view operator()(States::Invalidated){
                        return "States::Invalidated";
                    }
                    std::string_view operator()(States::OuterProduct){
                        return "States::OuterProduct";
                    }

            };


            class FunctionObjectSerializer {

                
                constexpr static size_t NoOpIdx   = 0;
                constexpr static size_t UnaryIdx  = 1;
                constexpr static size_t BinaryIdx = 2;

                public:
                constexpr static size_t BinaryOperandSize = 3;
                    
                    template <BinaryRegistry State>
                    std::array<std::optional<TensorID>, BinaryOperandSize> operator()(State s) {
                        std::array<std::optional<TensorID>, BinaryOperandSize> arr;
                        arr[NoOpIdx]   = s.get_tensor_id();
                        arr[UnaryIdx]  = s.left_op_id();
                        arr[BinaryIdx] = s.right_op_id();
                        return arr;
                    }


                    template <UnaryRegistry State>
                    std::array<std::optional<TensorID>, BinaryOperandSize> operator()(State s) {
                        std::array<std::optional<TensorID>, BinaryOperandSize> arr;
                        arr[NoOpIdx]   = s.get_tensor_id();
                        arr[UnaryIdx]  = s.left_op_id();
                        arr[BinaryIdx] = {};
                        return arr;
                    } 
                    
                    
                    template <NoOperandRegistry State>
                    std::array<std::optional<TensorID>, BinaryOperandSize> operator()(State s) {
                        std::array<std::optional<TensorID>, BinaryOperandSize> arr;
                        arr[NoOpIdx]   = s.get_tensor_id();
                        arr[UnaryIdx]  = {};
                        arr[BinaryIdx] = {};
                        return arr;
                    }


                    template <IsStateFull UndefinedState>
                    std::array<std::optional<TensorID>, BinaryOperandSize> operator()(UndefinedState) {
                        std::array<std::optional<TensorID>, BinaryOperandSize> arr;
                        arr[NoOpIdx]   = {};
                        arr[UnaryIdx]  = {};
                        arr[BinaryIdx] = {};
                        return arr;
                    }
                    

            };



            /*
                Flyweight object of 12 Bytes, uses State as 
                disciminated union to avoid extra 8 byte v-table 
                pointer allocated by compiler for late binding
                in runtime polymorphism. 

                FunctionObject encapsulates a registered operation
                State machine will be responsible for tracking
                if a tensor is ready to be differentiated or 
                have its gradient computed. 

                This is the object pointed to by FunctionObjectIterator
                which will have a policy template argument controlling,
                whether one prints, computes derivate, or updates gradients. 
            */

            class FunctionObject {
                
                public:
                    using State = StateTrait::State;
                    using Event = EventTrait::Event;

                    FunctionObject(RegisteredTensor nop) : state_(States::NoOperation{nop}) {}
                    FunctionObject() : state_(States::NoOperation{RegisteredTensor{}}) {}
                    
                    FunctionObject operator=(const FunctionObject& other) {
                        state_ = other.state_;
                        return *this;
                    }

                    void process_event(Event event) {
                        state_ = std::visit(
                            OperationTransitioner{},
                            state_, 
                            event
                        );
                    }

                    void stringify_type() {
                        std::cout << std::visit(
                            FunctionObjectInfo{},
                            state_
                        ) << std::endl;
                    }

                    std::array<
                        std::optional<TensorID>, 
                        FunctionObjectSerializer::BinaryOperandSize> serialize(void) {
                        auto data = std::visit(
                            FunctionObjectSerializer{},
                            state_
                        );
                        return data;
                    }
                private:
                    State state_;
                };


        }
    }
}


#endif // TENSOR_FUNCTION_OBJECT_H