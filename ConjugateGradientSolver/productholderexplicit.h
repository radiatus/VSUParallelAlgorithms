#ifndef PRODUCTHOLDEREXPLICIT_H
#define PRODUCTHOLDEREXPLICIT_H

#include "productholder.h"
#include "eigenadditionalmatrix.h"
#include "Eigen/Sparse"

namespace IterativeLinearSolvers {
namespace ConjugateGradient {

template <typename T>
class ProductHolderExplicit : public ProductHolder<T>
{
public:
    explicit ProductHolderExplicit(const Eigen::MatrixX<T> &j);

    void computeJtJv(const Eigen::VectorX<T> &vec, Eigen::VectorX<T> &output) override;
    void computeJtv(const Eigen::VectorX<T> &vec, Eigen::VectorX<T> &output) override;

    unsigned long cols() const;
    unsigned long rows() const;

protected:
    Eigen::MatrixX<T> m_j;
};




template<typename T>
void ProductHolderExplicit<T>::computeJtJv(const Eigen::VectorX<T> &vec, Eigen::VectorX<T> &output)
{
    output = m_j.adjoint() * (m_j * vec);
}

template<typename T>
void ProductHolderExplicit<T>::computeJtv(const Eigen::VectorX<T> &vec, Eigen::VectorX<T> &output)
{
    output = m_j.adjoint() * vec;
}


template<typename T>
unsigned long ProductHolderExplicit<T>::cols() const
{
    return m_j.cols();
}

template<typename T>
unsigned long ProductHolderExplicit<T>::rows() const
{
    return m_j.rows();
}

template<typename T>
ProductHolderExplicit<T>::ProductHolderExplicit(const Eigen::MatrixX<T> &j)
    :m_j(j)
{
}

}
}

#endif // PRODUCTHOLDEREXPLICIT_H
