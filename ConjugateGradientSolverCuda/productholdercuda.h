#ifndef PRODUCTHOLDERCUDA_H
#define PRODUCTHOLDERCUDA_H

#include <vector_types.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace IterativeLinearSolvers {
namespace ConjugateGradient {

class ProductHolderCuda
{
public:
    virtual ~ProductHolderCuda() = default;

    virtual void computeJtJv(const cublasHandle_t handle, const float *d_vec, float *d_output) = 0;
    virtual void computeJtv(const cublasHandle_t handle, const float *d_vec, float *d_output) = 0;

    virtual unsigned long cols() const = 0;
    virtual unsigned long rows() const = 0;
};
}
}

#endif // PRODUCTHOLDERCUDA_H
