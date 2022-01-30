#include "cudautils.h"

extern "C"
__global__
void vecSubVec_GPU(float *output, const float *first, const float *second, const unsigned long SIZE)
{
    unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= SIZE)
        return;

    output[index] = first[index] - second[index];
}

void CudaUtils::vecSubVec(float *output, const float *first, const float *second, const unsigned long SIZE)
{
    int blockSize;
    int minGridSize;
    int gridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vecSubVec_GPU, 0, SIZE);

    gridSize = (SIZE + blockSize - 1) / blockSize;
    vecSubVec_GPU<<<gridSize, blockSize>>>(output, first, second, SIZE);
}

extern "C"
__global__
void vecAddVec_GPU(float *output, const float *first, const float *second, const unsigned long SIZE)
{
    unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= SIZE)
        return;

    output[index] = first[index] + second[index];
}

void CudaUtils::vecAddVec(float *output, const float *first, const float *second, const unsigned long SIZE)
{
    int blockSize;
    int minGridSize;
    int gridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vecAddVec_GPU, 0, SIZE);

    gridSize = (SIZE + blockSize - 1) / blockSize;
    vecAddVec_GPU<<<gridSize, blockSize>>>(output, first, second, SIZE);
}

extern "C"
__global__
void scalarMulVec_GPU(float *output, const float scalar, const float *vector, const unsigned long SIZE)
{
    unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= SIZE)
        return;

    output[index] = vector[index] * scalar;
}

void CudaUtils::scalarMulVec(float *output, const float scalar, const float *vector, const unsigned long SIZE)
{
    int blockSize;
    int minGridSize;
    int gridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, scalarMulVec_GPU, 0, SIZE);

    gridSize = (SIZE + blockSize - 1) / blockSize;
    scalarMulVec_GPU<<<gridSize, blockSize>>>(output, scalar, vector, SIZE);
}

extern "C"
__global__
void convertFloatToFloat3_GPU(float3 *output, const float *input, const unsigned long OUTPUT_SIZE)
{
    unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= OUTPUT_SIZE)
        return;

    unsigned long indexFloat = index*3;
    output[index] = make_float3(input[indexFloat], input[indexFloat+1], input[indexFloat+2]);
}

extern "C"
__global__
void convertFloat3ToFloat_GPU(float *output, const float3 *input, const unsigned long INPUT_SIZE)
{
    unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= INPUT_SIZE)
        return;

    unsigned long indexFloat = index*3;
    output[indexFloat] = input[index].x;
    output[indexFloat+1] = input[index].y;
    output[indexFloat+2] = input[index].z;
}

void CudaUtils::convertFloatToFloat3(float3 *output, const float *input, const unsigned long OUTPUT_SIZE)
{
    int blockSize;
    int minGridSize;
    int gridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, convertFloatToFloat3_GPU, 0, OUTPUT_SIZE);

    gridSize = (OUTPUT_SIZE + blockSize - 1) / blockSize;
    convertFloatToFloat3_GPU<<<gridSize, blockSize>>>(output, input, OUTPUT_SIZE);
}


void CudaUtils::convertFloat3ToFloat(float *output, const float3 *input, const unsigned long INPUT_SIZE)
{
    int blockSize;
    int minGridSize;
    int gridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, convertFloat3ToFloat_GPU, 0, INPUT_SIZE);

    gridSize = (INPUT_SIZE + blockSize - 1) / blockSize;
    convertFloat3ToFloat_GPU<<<gridSize, blockSize>>>(output, input, INPUT_SIZE);

}
