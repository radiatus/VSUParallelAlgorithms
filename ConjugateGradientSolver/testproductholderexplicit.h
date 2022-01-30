#ifndef TESTPRODUCTHOLDEREXPLICIT_H
#define TESTPRODUCTHOLDEREXPLICIT_H

#include <QObject>
#include <QTest>

#include "productholderexplicit.h"

class TestProductHolderExplicit : public QObject
{
    Q_OBJECT

public:
    explicit TestProductHolderExplicit(QObject *parent = nullptr);

private slots:
    void testInit();
    void testJtJv();
    void testJtv();
};

#endif // TESTPRODUCTHOLDEREXPLICIT_H
