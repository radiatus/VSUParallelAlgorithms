#include "testproductholderexplicit.h"
#include "Eigen/SparseCore"
#include "Eigen/Sparse"
#include "Eigen/IterativeLinearSolvers"

using namespace IterativeLinearSolvers::ConjugateGradient;
using namespace Eigen;

TestProductHolderExplicit::TestProductHolderExplicit(QObject *parent)
    :QObject (parent)
{

}

void TestProductHolderExplicit::testInit()
{
    SparseMatrix<float> matrix(2, 2);
    matrix.insert(0,0) = 1; matrix.insert(0,1) = 2;
    matrix.insert(1,0) = 3; matrix.insert(1,1) = 4;

    ProductHolder<float> *holder = new ProductHolderExplicit<float>(matrix);

    unsigned long cols = holder->cols();
    unsigned long rows = holder->cols();

    QCOMPARE(cols, 2);
    QCOMPARE(rows, 2);

    delete holder;
}

void TestProductHolderExplicit::testJtJv()
{
    SparseMatrix<float> matrix(2, 2);
    matrix.insert(0,0) = 2; matrix.insert(0,1) = 20;
    matrix.insert(1,0) = 10; matrix.insert(1,1) = 5;

    SparseVector<float> vec(2);
    vec.insert(0) = 1;
    vec.insert(1) = 3;

    ProductHolder<float> *holder = new ProductHolderExplicit<float>(matrix);
    Eigen::VectorX<float> answer = Eigen::VectorX<float>::Zero(holder->cols());
    holder->computeJtJv(vec, answer);

    float sum = answer.sum();
    QCOMPARE(sum, float(1739));

    delete holder;
}

void TestProductHolderExplicit::testJtv()
{
    SparseMatrix<float> matrix(2, 2);
    matrix.insert(0,0) = 2; matrix.insert(0,1) = 20;
    matrix.insert(1,0) = 10; matrix.insert(1,1) = 5;

    SparseVector<float> vec(2);
    vec.insert(0) = 1;
    vec.insert(1) = 3;

    ProductHolder<float> *holder = new ProductHolderExplicit<float>(matrix);
    Eigen::VectorX<float> answer = Eigen::VectorX<float>::Zero(holder->cols());
    holder->computeJtv(vec, answer);

    float sum = answer.sum();
    QCOMPARE(sum, float(67));

    delete holder;
}
