#include "conjugategradientsolvercuda.h"

#include "CudaUtils/cudautils.h"

using namespace IterativeLinearSolvers;
using namespace ConjugateGradient;

SolverCuda::SolverCuda(ProductHolderCuda *holder)
    :m_holder(holder)
{
    allocateMem();
}

SolverCuda::~SolverCuda()
{
    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_JtJv);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_rPrev);
    cudaFree(d_tmp);
    cublasDestroy(handle);
}

Eigen::VectorXf SolverCuda::solve(const Eigen::VectorXf &vec)
{
    //https://en.wikipedia.org/wiki/Conjugate_gradient_method
    //using namespace CudaUtils;
    clearMem();

    float *h_vec = new float[RESIDUALS_COUNT];
    Eigen::Map<Eigen::VectorXf>(h_vec, vec.size()) = vec;

    float *d_vec;
    cudaMalloc(&d_vec, RESIDUALS_BYTES);
    cudaMemcpy(d_vec, h_vec, RESIDUALS_BYTES, cudaMemcpyKind::cudaMemcpyHostToDevice);

    m_holder->computeJtv(handle, d_vec, d_b);
    m_holder->computeJtJv(handle, d_x, d_JtJv);
    CudaUtils::vecSubVec(d_r, d_b, d_JtJv, VARIABLES_COUNT);

    cudaMemcpy(d_p, d_r, VARIABLES_BYTES, cudaMemcpyKind::cudaMemcpyDeviceToDevice);

    float norm = 0.0f;
    cublasSnrm2_v2(handle, VARIABLES_COUNT, d_b, 1, &norm);
    float threshold = m_epsilon * m_epsilon * norm * norm;

    for (m_iteration = 0; m_iteration < m_iterations; m_iteration++){
        cudaMemcpy(d_rPrev, d_r, VARIABLES_BYTES, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        cudaMemset(d_JtJv, 0.0f, VARIABLES_BYTES);

        float t1 = 0;
        cublasSdot (handle, VARIABLES_COUNT, d_r, 1,  d_r, 1, &t1);

        m_holder->computeJtJv(handle, d_p, d_JtJv);

        float t2 = 0;
        cublasSdot (handle, VARIABLES_COUNT, d_p, 1,  d_JtJv, 1, &t2);
        float a = t1 / t2;

        CudaUtils::scalarMulVec(d_tmp, a, d_p, VARIABLES_COUNT);
        CudaUtils::vecAddVec(d_x, d_x, d_tmp, VARIABLES_COUNT);

        CudaUtils::scalarMulVec(d_tmp, a, d_JtJv, VARIABLES_COUNT);
        CudaUtils::vecSubVec(d_r, d_r, d_tmp, VARIABLES_COUNT);

        cublasSnrm2_v2(handle, VARIABLES_COUNT, d_r, 1, &m_error);
        m_error *= m_error;
        if (m_error < threshold)
            break;

        cublasSdot (handle, VARIABLES_COUNT, d_r, 1,  d_r, 1, &t1);
        cublasSdot (handle, VARIABLES_COUNT, d_rPrev, 1,  d_rPrev, 1, &t2);
        float bi = t1 / t2;

        CudaUtils::scalarMulVec(d_tmp, bi, d_p, VARIABLES_COUNT);
        CudaUtils::vecAddVec(d_p, d_r, d_tmp, VARIABLES_COUNT);
    }

    cublasSnrm2_v2(handle, VARIABLES_COUNT, d_b, 1, &norm);
    norm *= norm;
    m_error = sqrt(m_error / norm);
    float *h_x = new float[VARIABLES_COUNT];
    cudaMemcpy(h_x, d_x, VARIABLES_BYTES, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    Eigen::VectorXf outputX(VARIABLES_COUNT);
    for (int ind = 0; ind < VARIABLES_COUNT; ind++)
        outputX[ind] = h_x[ind];

    delete[] h_vec;
    cudaFree(d_vec);
    delete [] h_x;

    return outputX;
}

void SolverCuda::setEpsilon(const float epsilon)
{
    m_epsilon = epsilon;
}

void SolverCuda::setIterations(const Eigen::Index iterations)
{
    m_iterations = iterations;
}

int SolverCuda::iterations() const
{
    return m_iteration;
}

int SolverCuda::iterationsLimit() const
{
    return m_iterations;
}

float SolverCuda::error() const
{
    return m_error;
}

float SolverCuda::epsilon() const
{
    return m_epsilon;
}

void SolverCuda::allocateMem()
{
    VARIABLES_COUNT = m_holder->cols();
    RESIDUALS_COUNT = m_holder->rows();

    VARIABLES_BYTES = VARIABLES_COUNT * sizeof (float);
    RESIDUALS_BYTES = RESIDUALS_COUNT * sizeof (float);

    cudaMalloc(&d_x, VARIABLES_BYTES);
    cudaMalloc(&d_b, VARIABLES_BYTES);
    cudaMalloc(&d_JtJv, VARIABLES_BYTES);
    cudaMalloc(&d_r, VARIABLES_BYTES);
    cudaMalloc(&d_p, VARIABLES_BYTES);
    cudaMalloc(&d_rPrev, VARIABLES_BYTES);
    cudaMalloc(&d_tmp, VARIABLES_BYTES);

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        exit(EXIT_FAILURE);
    }
}

void SolverCuda::clearMem()
{
    cudaMemset(d_x, 0.0f, VARIABLES_BYTES);
    cudaMemset(d_b, 0.0f, VARIABLES_BYTES);
    cudaMemset(d_JtJv, 0.0f, VARIABLES_BYTES);
    cudaMemset(d_r, 0.0f, VARIABLES_BYTES);
    cudaMemset(d_p, 0.0f, VARIABLES_BYTES);
    cudaMemset(d_rPrev, 0.0f, VARIABLES_BYTES);
    cudaMemset(d_tmp, 0.0f, VARIABLES_BYTES);
}
