#ifndef CONJUGATEGRADIENTSOLVERCUDA_H
#define CONJUGATEGRADIENTSOLVERCUDA_H

#include "productholdercuda.h"
#include "Eigen/Dense"

#include "vector"
#include "iterator"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace IterativeLinearSolvers {
namespace ConjugateGradient {

class SolverCuda
{

public:
    explicit SolverCuda(ProductHolderCuda *holder);
    ~SolverCuda();

    Eigen::VectorXf solve(const Eigen::VectorXf &vec);

    void setEpsilon(const float epsilon);
    void setIterations(const Eigen::Index iterationsLimit);

    int iterations() const;
    int iterationsLimit() const;
    float error() const;
    float epsilon() const;

protected:
    void allocateMem();
    void clearMem();

    ProductHolderCuda *m_holder;
    int m_iteration = 0;
    int m_iterations = INT_MAX;
    float m_error = 0.0f;
    float m_epsilon = Eigen::NumTraits<float>::epsilon();

    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    float *d_x;
    float *d_b;
    float *d_JtJv;
    float *d_r;
    float *d_p;
    float *d_rPrev;
    float *d_tmp;

    unsigned long VARIABLES_COUNT;
    unsigned long RESIDUALS_COUNT;

    unsigned long VARIABLES_BYTES;
    unsigned long RESIDUALS_BYTES;
};

}

}


#endif // CONJUGATEGRADIENTSOLVERCUDA_H
