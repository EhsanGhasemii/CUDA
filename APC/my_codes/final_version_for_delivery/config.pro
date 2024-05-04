QT       += core gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
CONFIG += c++11

TEMPLATE = app
INCLUDEPATH += .

DEFINES += QT_DEPRECATED_WARNINGS

# CUDA settings
CUDA_DIR = /usr
SYSTEM_TYPE = 64
NVCC_OPTIONS = --use_fast_math
INCLUDEPATH += $$CUDA_DIR/include
LIBS += -L$$CUDA_DIR/lib64 -lcudart
CUDA_SOURCES += src/cuda_main.cu

CUDA_OBJECTS_DIR = ./cuda_objects
system(mkdir -p $$CUDA_OBJECTS_DIR)

cuda.input = CUDA_SOURCES
cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
cuda.commands = $$CUDA_DIR/bin/nvcc -c $$NVCC_OPTIONS $$CUDA_INC $$LIBS --machine $$SYSTEM_TYPE -arch=sm_75 -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
cuda.dependency_type = TYPE_C
QMAKE_EXTRA_COMPILERS += cuda

# Common sources and headers
SOURCES += \
    Plotters/qcustomplot.cpp \
    Plotters/jcustomplot.cpp \
    src/input.cpp \
	src/matchfilter.cpp \
	src/general_apc.cpp \
	src/gpuFunctions.cpp

HEADERS += \
    Plotters/qcustomplot.h \
    Plotters/jcustomplot.h \
	include/general_apc.h \
	include/matchfilter.h \
	include/cuda_main.h \
	include/gpuerrors.h \
	include/gputimer.h \
	include/gpuFunctions.h

FORMS += files/mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

LIBS += -larmadillo

# First target
TARGET = output
SOURCES += src/mainwindow.cpp
HEADERS += include/mainwindow.h

# Second target
CONFIG += debug_and_release
CONFIG(debug, debug|release) {
    TARGET = output2
	SOURCES -= src/input.cpp
    SOURCES -= src/mainwindow.cpp
    SOURCES += src/mainwindow2.cpp
    HEADERS -= include/mainwindow.h
    HEADERS += include/mainwindow2.h
}
