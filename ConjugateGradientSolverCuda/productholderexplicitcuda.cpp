#include "productholderexplicitcuda.h"

void IterativeLinearSolvers::ConjugateGradient::ProductHolderExplicitCuda::computeJtJv(const cublasHandle_t handle, const float *d_vec, float *d_output)
{
    float alpha = 1;
    float beta = 0;
    cublasSgemv(handle, CUBLAS_OP_N, m_cols, m_rows, &alpha, m_j, m_cols, d_vec, 1, &beta, m_jvBuffer, 1);
    cublasSgemv(handle, CUBLAS_OP_T, m_cols, m_rows, &alpha, m_j, m_cols, m_jvBuffer, 1, &beta, d_output, 1);
}


void IterativeLinearSolvers::ConjugateGradient::ProductHolderExplicitCuda::computeJtv(const cublasHandle_t handle, const float *d_vec, float *d_output)
{
    float alpha = 1;
    float beta = 0;
    cublasSgemv(handle, CUBLAS_OP_T, m_cols, m_rows, &alpha, m_j, m_cols, d_vec, 1, &beta, d_output, 1);
}

unsigned long IterativeLinearSolvers::ConjugateGradient::ProductHolderExplicitCuda::cols() const
{
    return m_cols;
}

unsigned long IterativeLinearSolvers::ConjugateGradient::ProductHolderExplicitCuda::rows() const
{
    return m_rows;
}

IterativeLinearSolvers::ConjugateGradient::ProductHolderExplicitCuda::ProductHolderExplicitCuda(const Eigen::MatrixXf &j)
{
    m_cols = j.cols();
    m_rows = j.rows();

    unsigned long BYTES = m_cols * m_rows * sizeof (float);
    unsigned long JV_BUFFER_BYTES = m_rows * sizeof (float);

    cudaMalloc(&m_jvBuffer, JV_BUFFER_BYTES);

    cudaMalloc(&m_j, BYTES);
    cudaMemcpy(m_j, j.data(), BYTES, cudaMemcpyKind::cudaMemcpyHostToDevice);
}

IterativeLinearSolvers::ConjugateGradient::ProductHolderExplicitCuda::~ProductHolderExplicitCuda()
{
    cudaFree(m_jvBuffer);
    cudaFree(m_j);
}
