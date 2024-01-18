cmake_minimum_required(VERSION 3.10)
project(NeuralNet
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

set(NEURAL_NET
        src/Layer.cpp
        src/Network.cpp
        src/ActivationFunctions/Sigmoid.cpp
        src/LossFunctions/MSE.cpp)

add_library(NeuralNet ${NEURAL_NET})

target_include_directories(NeuralNet PRIVATE "eigen")
target_include_directories(NeuralNet PRIVATE "EigenRand")
