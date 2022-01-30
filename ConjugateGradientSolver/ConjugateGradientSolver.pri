!CONJUGATE_GRADIENT_SOLVER_PRI {
CONFIG += CONJUGATE_GRADIENT_SOLVER_PRI
INCLUDEPATH += $$PWD/../

SOURCES += \
    $$PWD/testproductholderexplicit.cpp \
    $$PWD/testconjugategradient.cpp

HEADERS += \
    $$PWD/productholder.h \
    $$PWD/productholderexplicit.h \
    $$PWD/testproductholderexplicit.h \
    $$PWD/conjugategradientsolver.h \
    $$PWD/testconjugategradient.h \
    $$PWD/eigenadditionalmatrix.h
}


