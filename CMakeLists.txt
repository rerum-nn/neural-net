cmake_minimum_required(VERSION 3.10)
project(NeuralNet
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

set(NEURAL_NET
        src/Layer.cpp
        src/Network.cpp
        src/ActivationFunctions/Sigmoid.cpp
        src/WeightGenerators/NormalRandom.cpp
        src/LossFunctions/MSE.cpp
        src/LinearPrimitives.cpp)

add_library(NeuralNet ${NEURAL_NET})
