QT       += core gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
CONFIG += c++11



# 
TEMPLATE = app
TARGET = output
INCLUDEPATH += .



# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


###################################################
# CUDA settings -- 
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
###################################################




SOURCES += \
    Plotters/qcustomplot.cpp \
    Plotters/jcustomplot.cpp \
#    code3.cpp \
    src/input.cpp \
	src/mainwindow.cpp \
	src/matchfilter.cpp \ 
	src/general_apc.cpp \
#	src/abas.cpp


HEADERS += \
    Plotters/qcustomplot.h \
    Plotters/jcustomplot.h \
#    code3.h \
	include/mainwindow.h \
	include/general_apc.h \
	include/matchfilter.h \
	include/cuda_main.h \
	include/gpuerrors.h \
	include/gputimer.h

FORMS += \
    files/mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target


LIBS += -larmadillo

