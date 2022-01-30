#include "testconjugategradient.h"

TestConjugateGradient::TestConjugateGradient(QObject *parent)
    :QObject (parent)
{

}

void TestConjugateGradient::testSolve()
{
    using namespace IterativeLinearSolvers::ConjugateGradient;

    Eigen::SparseMatrix<float> A(4, 2);
    A.insert(0,0) = 1; A.insert(0,1) = 1;
    A.insert(1,0) = 1; A.insert(1,1) = 2;
    A.insert(2,0) = 1; A.insert(2,1) = 3;
    A.insert(3,0) = 1; A.insert(3,1) = 4;

    Eigen::SparseVector<float> b(4);
    b.insert(0) = 1;
    b.insert(1) = -2;
    b.insert(2) = -5;
    b.insert(3) = -8;

    ProductHolder<float> *holder = new ProductHolderExplicit<float>(A);
    Solver<float> solverOur(holder);
    Eigen::VectorX<float> w2 = solverOur.solve(b);
    delete holder;

    QCOMPARE((w2[0] - 3.99999904633) < 0.0001, true);
    QCOMPARE((w2[1] + 2.99999904633) < 0.0001, true);
}

void TestConjugateGradient::testSolve2()
{
    using namespace IterativeLinearSolvers::ConjugateGradient;

    Eigen::SparseMatrix<float> A(4, 2);
    A.insert(0,0) = 1; A.insert(0,1) = 1;
    A.insert(1,0) = 1; A.insert(1,1) = 2;
    A.insert(2,0) = 1; A.insert(2,1) = 3;
    A.insert(3,0) = 1; A.insert(3,1) = 4;

    Eigen::SparseVector<float> b(4);
    b.insert(0) = 1;
    b.insert(1) = -2;
    b.insert(2) = -5;
    b.insert(3) = -8;

    ProductHolder<float> *holder = new ProductHolderExplicit<float>(A);
    Solver<float> solverOur(holder);
    Eigen::VectorXf w2 = solverOur.solve(b.toDense());

    Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<float>> solver(A);
    Eigen::SparseVector<float> result = solver.solve(b);

    delete holder;

    QCOMPARE((w2[0] - 3.99999904633) < 0.0001, true);
    QCOMPARE((w2[1] + 2.99999904633) < 0.0001, true);
    QCOMPARE(solver.iterations(), solverOur.iterations());
}
