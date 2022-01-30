#ifndef CONJUGATEGRADIENTSOLVER_H
#define CONJUGATEGRADIENTSOLVER_H

#include "productholder.h"
#include "eigenadditionalmatrix.h"
#include "Eigen/IterativeLinearSolvers"

namespace IterativeLinearSolvers {
namespace ConjugateGradient {

template <typename T>
class Solver
{

public:
    explicit Solver(ProductHolder<T> *holder);

    Eigen::VectorX<T> solve(const Eigen::VectorX<T> &vec);

    void setEpsilon(const T epsilon);
    void setIterations(const int iterationsLimit);

    int iterations() const;
    int iterationsLimit() const;

    T error() const;
    T epsilon() const;

protected:
    ProductHolder<T> *m_holder;
    int m_iteration = 0;
    int m_iterations = INT_MAX;
    T m_error = T(0);
    T m_epsilon = Eigen::NumTraits<T>::epsilon();
};





template<typename T>
Solver<T>::Solver(ProductHolder<T> *holder)
    :m_holder(holder)
{
}

template<typename T>
Eigen::VectorX<T> Solver<T>::solve(const Eigen::VectorX<T> &vec)
{
    //https://en.wikipedia.org/wiki/Conjugate_gradient_method

    Eigen::VectorX<T> x = Eigen::VectorX<T>::Zero(m_holder->cols());

    Eigen::VectorX<T> b = Eigen::VectorX<T>::Zero(m_holder->cols());
    m_holder->computeJtv(vec, b);

    Eigen::VectorX<T> JtJv = Eigen::VectorX<T>::Zero(m_holder->cols());
    m_holder->computeJtJv(x, JtJv);
    Eigen::VectorX<T> r = b - JtJv;

    Eigen::VectorX<T> p = r;

    T threshold = m_epsilon * m_epsilon * b.norm() * b.norm();

    Eigen::VectorX<T> rPrev(r.size());

    for (m_iteration = 0; m_iteration < m_iterations; m_iteration++){
        rPrev = r;
        JtJv.setZero();

        T t1 = r.transpose() * r;
        m_holder->computeJtJv(p, JtJv);
        T t2 = p.transpose() * JtJv;
        T a = t1 / t2;

        x = x + a * p;
        r = r - a * JtJv;

        m_error = r.norm() * r.norm();
        if (m_error < threshold)
            break;

        t1 = r.transpose() * r;
        t2 = rPrev.transpose() * rPrev;
        T bi = t1 / t2;

        p = r + bi * p;
    }
    m_error = sqrt(m_error / (b.norm() * b.norm()));
    return x;
}

template<typename T>
void Solver<T>::setEpsilon(const T epsilon)
{
    m_epsilon = epsilon;
}

template<typename T>
void Solver<T>::setIterations(const int iterations)
{
    m_iterations = iterations;
}

template<typename T>
int Solver<T>::iterations() const
{
    return m_iteration;
}

template<typename T>
int Solver<T>::iterationsLimit() const
{
    return m_iterations;
}

template<typename T>
T Solver<T>::error() const
{
    return m_error;
}

template<typename T>
T Solver<T>::epsilon() const
{
    return m_epsilon;
}

}

}

#endif // CONJUGATEGRADIENTSOLVER_H
