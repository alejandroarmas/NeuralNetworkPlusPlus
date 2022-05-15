#ifndef TENSOR_BACKWARDS_PASS
#define TENSOR_BACKWARDS_PASS

#include "tensor.h"

namespace NeuralNetwork {

    namespace Computation {

        namespace Graph {

            class PrintTag;
            class GradientTag;


            class ReversePass {

                public:
                    ReversePass() :
                        map(ComputationalGraphMap::get()) {}

                    void backwards(
                        Tensor& _t, 
                        PrintTag _);

                    void backwards(
                        Tensor& _t, 
                        GradientTag _);
                private:
                    ComputationalGraphMap& map;

            };
            

            
            template <class StrategyType>
            struct ReverseTag {

                void _backwards(
                    Tensor& _t,
                    ReversePass& strat_implementation) {
                        
                    return strat_implementation.backwards(
                        _t, *static_cast<
                        StrategyType const*>(this));
            } };

            class PrintTag : public ReverseTag<PrintTag> {
                public:
                    PrintTag() = default;
            };
            class GradientTag : public ReverseTag<GradientTag> {
                public:
                    GradientTag() = default;
            };
    

 
        }

    }

}


#endif // TENSOR_BACKWARDS_PASS