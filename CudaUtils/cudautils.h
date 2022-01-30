#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace CudaUtils {

void convertFloatToFloat3(float3 *output, const float *input, const unsigned long OUTPUT_SIZE);
void convertFloat3ToFloat(float *output, const float3 *input, const unsigned long INPUT_SIZE);

void vecSubVec(float *output, const float *first, const float *second, const unsigned long SIZE);
void vecAddVec(float *output, const float *first, const float *second, const unsigned long SIZE);
void scalarMulVec(float *output, const float scalar, const float *vector, const unsigned long SIZE);

}

#endif // CUDAUTILS_H
