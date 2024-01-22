cmake_minimum_required(VERSION 3.10)
project(NeuralNet
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(NEURAL_NET
        src/Layer.cpp
        src/Network.cpp
        src/ActivationFunctions/Sigmoid.cpp
        src/LossFunctions/MSE.cpp
        src/Random.cpp
        src/Random.h)
set(INCLUDE_DIRECTORIES eigen EigenRand)
set(GOOGLE_TEST googletest/googlemock/include googletest/googletest/include)

add_library(NeuralNet ${NEURAL_NET})
target_include_directories(NeuralNet PRIVATE ${INCLUDE_DIRECTORIES})

enable_testing()
add_subdirectory(googletest)
add_executable(tests tests/main.cpp ${NEURAL_NET})
target_include_directories(tests PRIVATE ${GOOGLE_TEST} ${INCLUDE_DIRECTORIES})
target_link_libraries(tests gtest_main gmock_main)

