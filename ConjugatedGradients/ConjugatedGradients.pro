QT += core gui testlib

CONFIG += c++17
CONFIG -= app_bundle

include($$PWD/../Cuda/Cuda.pri)
include($$PWD/../eigen-3.2.4/eigen.pri)
include($$PWD/../CudaUtils/CudaUtils.pri)
include($$PWD/../ConjugateGradientSolver/ConjugateGradientSolver.pri)
include($$PWD/../ConjugateGradientSolverCuda/ConjugateGradientSolverCuda.pri)

SOURCES += \
        main.cpp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
