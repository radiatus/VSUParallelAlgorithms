!CONJUGATE_GRADIENT_SOLVER_CUDA_PRI {
CONFIG += CONJUGATE_GRADIENT_SOLVER_CUDA_PRI
INCLUDEPATH += $$PWD/../

include($$PWD/../CudaUtils/CudaUtils.pri)

HEADERS += \
    $$PWD/conjugategradientsolvercuda.h \
    $$PWD/productholdercuda.h

SOURCES += \
    $$PWD/conjugategradientsolvercuda.cpp

}

HEADERS += \
    $$PWD/productholderexplicitcuda.h

SOURCES += \
    $$PWD/productholderexplicitcuda.cpp




