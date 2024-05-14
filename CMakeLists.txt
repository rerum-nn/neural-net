cmake_minimum_required(VERSION 3.23)
project(NeuralNet
        VERSION 1.0.0
        DESCRIPTION "Course project on creating a library for neural networks"
        LANGUAGES CXX)

include(cmake/utils.cmake)
include(GNUInstallDirs)

add_library(NeuralNet)
add_library(NeuralNet::NeuralNet ALIAS NeuralNet)

option(NEURAL_NET_BUILD_TESTS "Build NeuralNet tests" ON)
option(NEURAL_NET_INSTALL "Generate target for installing NeuralNet" ${PROJECT_IS_TOP_LEVEL})

set_if_undefined(CMAKE_CXX_VISIBILITY_PRESET hidden)
set_if_undefined(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(NEURAL_NET_HEADERS
        src/Sequential.h
        src/Types.h
        src/Datasets/MNIST/MnistDataset.h
        src/Layers/Linear.h
        src/Layers/Activations/Activation.h
        src/Layers/Activations/ActivationNone.h
        src/Layers/Activations/Sigmoid.h
        src/Layers/Activations/Tanh.h
        src/Layers/Activations/Softmax.h
        src/Layers/Activations/ReLU.h
        src/Layers/Activations/LeakyReLU.h
        src/LossFunctions/LossFunction.h
        src/LossFunctions/MSE.h
        src/LossFunctions/MAE.h
        src/LossFunctions/BinaryCrossEntropy.h
        src/LossFunctions/CategoricalCrossEntropy.h
        src/Metrics/Metric.h
        src/Optimizers/SGD.h
        src/Optimizers/Adam.h
        src/Optimizers/Optimizer.h
        src/Utils/Random.h
        src/Utils/BatchSlicer.h
        src/Utils/DataManipulate.h
        src/Utils/Timer.h
)

set(NEURAL_NET_SOURCES
        src/Sequential.cpp
        src/Datasets/MNIST/MnistDataset.cpp
        src/Layers/Linear.cpp
        src/Layers/Activations/Sigmoid.cpp
        src/Layers/Activations/Tanh.cpp
        src/Layers/Activations/Softmax.cpp
        src/Layers/Activations/ReLU.cpp
        src/Layers/Activations/LeakyReLU.cpp
        src/Layers/Activations/ActivationNone.cpp
        src/LossFunctions/MSE.cpp
        src/LossFunctions/MAE.cpp
        src/LossFunctions/BinaryCrossEntropy.cpp
        src/LossFunctions/CategoricalCrossEntropy.cpp
        src/Metrics/Metric.cpp
        src/Optimizers/SGD.cpp
        src/Optimizers/Adam.cpp
        src/Optimizers/Optimizer.cpp
        src/Utils/Random.cpp
        src/Utils/BatchSlicer.cpp
        src/Utils/DataManipulate.cpp
        src/Utils/Timer.cpp
)
set(EIGEN_INCLUDE_DIRECTORIES eigen EigenRand)

include(CMakePackageConfigHelpers)

target_sources(NeuralNet PRIVATE ${NEURAL_NET_SOURCES})
target_include_directories(NeuralNet PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src" ${EIGEN_INCLUDE_DIRECTORIES})

if (NEURAL_NET_BUILD_TESTS)
    add_subdirectory(tests)
    add_subdirectory(googletest)
    enable_testing()
endif ()
