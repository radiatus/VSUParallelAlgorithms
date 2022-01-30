#ifndef TESTCONJUGATEGRADIENT_H
#define TESTCONJUGATEGRADIENT_H

#include <QObject>
#include <QTest>

#include "conjugategradientsolver.h"
#include "productholderexplicit.h"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Cholesky>

class TestConjugateGradient : public QObject
{
    Q_OBJECT
public:
    explicit TestConjugateGradient(QObject *parent = nullptr);

private slots:
    void testSolve();
    void testSolve2();
};

#endif // TESTCONJUGATEGRADIENT_H
