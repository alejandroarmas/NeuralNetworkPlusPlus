
#include <math.h>   // fabs(), isinf(), isnan()
#include <string.h> // memset: zeros out all the bytes in an array
#include <stdint.h> // uint_64t, INT32_MAX
#include <stdlib.h>
#include <stdio.h>

#include "functions.h"
#include "config.h"



template <class T>
T Functions::Activation::Sigmoid<T>::operate(T x) {

    T denominator = 1 + ::exp(x); 
    T output = 1 / denominator; 

    return output;
}

template <class T>
T Functions::Activation::Differentiable::Sigmoid<T>::operate(T x) {

    Functions::Activation::Sigmoid<T> sigmoid;

    T result = sigmoid.operate(x);
 
    T output = result * (1 - result);

    return output;
}



// --------------------------------------------------
/* --------------------------------------------------
Code from: https://bitbashing.io/comparing-floats.html
*/
volatile bool Functions::Utility::compare_float(float a, float b) {
    
    const float difference = fabs(a - b);
    if (difference <= EPSILON) return true;

    return Functions::Utility::ulpsDistance(a, b) <= ULPS_EPSILON;
}

/*
Units of Least Precisixon: How many mantisa bits of difference we tolerate.
*/
int32_t Functions::Utility::ulpsDistance(const float a, const float b)
{
    // Save work if the floats are equal.
    // Also handles +0 == -0
    if (a == b) return 0;

    
    // Max distance for NaN
    if (isnan(a) || isnan(b)) return INT32_MAX;

    // If one's infinite and they're not equal, max distance.
    if (isinf(a) || isinf(b)) return INT32_MAX;

    int32_t ia, ib;
    memcpy(&ia, &a, sizeof(float));
    memcpy(&ib, &b, sizeof(float));

    // Don't compare differently-signed floats.
    if ((ia < 0) != (ib < 0)) return INT32_MAX;

    // Return the absolute value of the distance in ULPs.
    int32_t distance = ia - ib;
    if (distance < 0) distance = -distance;
    return distance;
}

// --------------------------------------------------