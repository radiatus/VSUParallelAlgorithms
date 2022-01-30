#ifndef EIGENADDITIONALMATRIX_H
#define EIGENADDITIONALMATRIX_H

#include "Eigen/Core"

namespace Eigen {
template<typename S>
using Vector2 = Eigen::Matrix<S, 2, 1> ;
template<typename S>
using Vector3 = Eigen::Matrix<S, 3, 1> ;
template<typename Scalar>
using Vector4 = Matrix<Scalar, 4, 1>;
template<typename Scalar>
using Matrix3 = Matrix<Scalar, 3, 3>;
template<typename Scalar>
using Matrix4 = Matrix<Scalar, 4, 4>;
template<typename S>
using MatrixX = Eigen::Matrix<S, Dynamic, Dynamic>;
template<typename S>
using VectorX = Eigen::Matrix<S, Dynamic, 1>;
}

#endif // EIGENADDITIONALMATRIX_H
