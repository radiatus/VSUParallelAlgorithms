#ifndef PRODUCTHOLDER_H
#define PRODUCTHOLDER_H

#include "Eigen/Core"
#include "Eigen/IterativeLinearSolvers"
#include "eigenadditionalmatrix.h"

namespace IterativeLinearSolvers {
namespace ConjugateGradient {

template <typename T>
class ProductHolder
{
public:
    virtual ~ProductHolder() = default;

    virtual void computeJtJv(const Eigen::VectorX<T> &vec, Eigen::VectorX<T> &output) = 0;
    virtual void computeJtv(const Eigen::VectorX<T> &vec, Eigen::VectorX<T> &output) = 0;

    virtual unsigned long cols() const = 0;
    virtual unsigned long rows() const = 0;
};
}
}

#endif // PRODUCTHOLDER_H
