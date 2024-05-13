cmake_minimum_required(VERSION 3.10)
project(NeuralNet-tests LANGUAGES CXX)

set(TEST_SOURCES
        mnist.cpp
        test_metrics.cpp
        test_optimizers.cpp
        test_layers.cpp
        test_deserialize.cpp
)
set(GOOGLE_TEST_INCLUDE_DIRS ../googletest/googlemock/include ../googletest/googletest/include)

function(add_test_executable TEST_NAME)
    add_executable(${TEST_NAME} ${ARGN})
    target_include_directories(${TEST_NAME} PRIVATE ${GOOGLE_TEST_INCLUDE_DIRS})
    target_link_libraries(${TEST_NAME} gtest_main gmock_main NeuralNet::NeuralNet)
endfunction()

foreach (TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_test_executable(${TEST_NAME} ${TEST_SOURCE})
endforeach()

if(DEFINED THREADS)
    add_definitions(-DTHREADS={THREADS})
endif()
