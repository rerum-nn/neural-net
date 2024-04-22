cmake_minimum_required(VERSION 3.10)
project(NeuralNet
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -march=native -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops -fno-math-errno")

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
        src/Layers/ReLU.cpp
        src/Metrics/Metric.cpp
)
set(INCLUDE_DIRECTORIES eigen EigenRand src)
set(GOOGLE_TEST googletest/googlemock/include googletest/googletest/include)

add_library(NeuralNet ${NEURAL_NET})
target_include_directories(NeuralNet PRIVATE ${INCLUDE_DIRECTORIES})

add_executable(mnist tests/mnist.cpp ${NEURAL_NET})
target_include_directories(mnist PRIVATE ${INCLUDE_DIRECTORIES})

enable_testing()
add_subdirectory(googletest)
add_executable(tests tests/trash_tests.cpp ${NEURAL_NET})
target_include_directories(tests PRIVATE ${GOOGLE_TEST} ${INCLUDE_DIRECTORIES})
target_link_libraries(tests gtest_main gmock_main)

add_executable(test_eigen tests/test_eigen.cpp )
target_include_directories(test_eigen PRIVATE ${INCLUDE_DIRECTORIES})

add_executable(test_metrics tests/test_metrics.cpp ${NEURAL_NET})
target_include_directories(test_metrics PRIVATE ${GOOGLE_TEST} ${INCLUDE_DIRECTORIES})
target_link_libraries(test_metrics gtest_main gmock_main)

add_executable(test_optimizers tests/test_optimizers.cpp ${NEURAL_NET}
        tests/TestHelpers.h)
target_include_directories(test_optimizers PRIVATE ${GOOGLE_TEST} ${INCLUDE_DIRECTORIES})
target_link_libraries(test_optimizers gtest_main gmock_main)

add_executable(test_layers tests/test_layers.cpp ${NEURAL_NET}
        tests/TestHelpers.h)
target_include_directories(test_layers PRIVATE ${GOOGLE_TEST} ${INCLUDE_DIRECTORIES})
target_link_libraries(test_layers gtest_main gmock_main)

