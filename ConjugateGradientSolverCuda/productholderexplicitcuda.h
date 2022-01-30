#ifndef PRODUCTHOLDEREXPLICITCUDA_H
#define PRODUCTHOLDEREXPLICITCUDA_H

#include "productholdercuda.h"
#include "Eigen/Dense"

namespace IterativeLinearSolvers {
namespace ConjugateGradient {

class ProductHolderExplicitCuda : public ProductHolderCuda
{
public:
    explicit ProductHolderExplicitCuda(const Eigen::MatrixXf &j);
    ~ProductHolderExplicitCuda();

    void computeJtJv(const cublasHandle_t handle, const float *d_vec, float *d_output) override;
    void computeJtv(const cublasHandle_t handle, const float *d_vec, float *d_output) override;

    unsigned long cols() const override;
    unsigned long rows() const override;

protected:
    unsigned long m_cols;
    unsigned long m_rows;

    float *m_j;
    float *m_jvBuffer;
};

}
}

#endif // PRODUCTHOLDEREXPLICITCUDA_H
