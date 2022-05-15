#ifndef TENSOR_FACTORY_H
#define TENSOR_FACTORY_H

#include "tensor.h"
#include "m_algorithms_concepts.h"
#include "m_algorithms_utilities.h"

#include <memory>


namespace NeuralNetwork {

    namespace Computation {

        namespace Graph {



            class TensorConstructor {

                public:
                    static std::shared_ptr<Tensor> create(
                        Matrix::Rows _l, Matrix::Columns _w, 
                        IsTrackable _t  = IsTrackable(true), 
                        IsLeaf _f       = IsLeaf(true),
                        IsRecordable _r = IsRecordable(true));


                template <Matrix::Operations::MatrixOperatable Operator>
                    static std::shared_ptr<Tensor> create(
                        Operator _operator,
                        const Matrix::Representation& _m,
                        TensorID _op  = TensorID(0), 
                        TensorID _op2 = TensorID(0),  
                        IsTrackable _t  = IsTrackable(true), 
                        IsLeaf _f       = IsLeaf(true),
                        IsRecordable _r = IsRecordable(true));
            };


        }

    }

}


#endif // TENSOR_FACTORY_H