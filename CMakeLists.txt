cmake_minimum_required(VERSION 3.10)
project(NeuralNet
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -march=native -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -fno-math-errno")

set(NEURAL_NET
        src/Layers/Linear.cpp
        src/Sequential.cpp
        src/Layers/Sigmoid.cpp
        src/LossFunctions/MSE.cpp
        src/Utils/Random.cpp
        src/Optimizers/SGD.cpp
        src/LossFunctions/BinaryCrossEntropy.cpp
        src/Layers/Softmax.cpp
        src/LossFunctions/CategoricalCrossEntropy.cpp
        src/Optimizers/Adam.cpp
        src/Optimizers/Optimizer.cpp
        src/Utils/BatchSlicer.cpp
        src/Utils/DataManipulate.cpp
        src/Datasets/MNIST/MnistDataset.cpp
        src/Layers/ReLU.cpp)
set(INCLUDE_DIRECTORIES eigen EigenRand src)
set(GOOGLE_TEST googletest/googlemock/include googletest/googletest/include)

add_library(NeuralNet ${NEURAL_NET})
target_include_directories(NeuralNet PRIVATE ${INCLUDE_DIRECTORIES})

add_executable(mnist tests/mnist.cpp ${NEURAL_NET})
target_include_directories(mnist PRIVATE ${INCLUDE_DIRECTORIES})

enable_testing()
add_subdirectory(googletest)
add_executable(tests tests/main.cpp ${NEURAL_NET})
target_include_directories(tests PRIVATE ${GOOGLE_TEST} ${INCLUDE_DIRECTORIES})
target_link_libraries(tests gtest_main gmock_main)

