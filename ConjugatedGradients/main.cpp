#include <QCoreApplication>

#include <QDebug>
#include <QElapsedTimer>

#include "ConjugateGradientSolver/conjugategradientsolver.h"
#include "ConjugateGradientSolver/productholderexplicit.h"

#include "ConjugateGradientSolverCuda/conjugategradientsolvercuda.h"
#include "ConjugateGradientSolverCuda/productholderexplicitcuda.h"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    const int64_t n = 5000;
    Eigen::MatrixXf jac = Eigen::MatrixXf::Random(n, n);

    Eigen::VectorXf b(n);
    b.setRandom();

    QElapsedTimer timer;
    timer.start();

    IterativeLinearSolvers::ConjugateGradient::ProductHolderExplicit<float> productHolder(jac);
    IterativeLinearSolvers::ConjugateGradient::Solver<float> solver(&productHolder);
    Eigen::VectorXf x = solver.solve(b);
    qDebug() << "Timer CPU = " + QString::number(timer.elapsed() / 1000.0f);

    timer.start();
    IterativeLinearSolvers::ConjugateGradient::ProductHolderExplicitCuda productHolderCuda(jac);
    IterativeLinearSolvers::ConjugateGradient::SolverCuda solverCuda(&productHolderCuda);
    Eigen::VectorXf xCuda = solverCuda.solve(b);
    qDebug() << "Timer GPU = " + QString::number(timer.elapsed() / 1000.0f);

    return a.exec();
}
