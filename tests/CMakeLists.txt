cmake_minimum_required(VERSION 3.19)
project(NeuralNet-tests LANGUAGES CXX)

set(TEST_SOURCES
        test_metrics.cpp
        test_optimizers.cpp
        test_layers.cpp
        test_loss.cpp
)
set(TEST_WO_GOOGLE_SOURCES
        mnist.cpp
        test_deserialize.cpp
)

set(GOOGLE_TEST_INCLUDE_DIRS ../googletest/googlemock/include ../googletest/googletest/include)

function(add_test_executable TEST_NAME)
    add_executable(${TEST_NAME} ${ARGN})
    target_include_directories(${TEST_NAME} PRIVATE ${GOOGLE_TEST_INCLUDE_DIRS})
    target_link_libraries(${TEST_NAME} gtest_main gmock_main NeuralNet::NeuralNet)
endfunction()

function(add_test_executable_wo_google TEST_NAME)
    add_executable(${TEST_NAME} ${ARGN})
    target_link_libraries(${TEST_NAME} NeuralNet::NeuralNet)
endfunction()

foreach (TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_test_executable(${TEST_NAME} ${TEST_SOURCE})
endforeach ()

foreach (TEST_SOURCE ${TEST_WO_GOOGLE_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_test_executable_wo_google(${TEST_NAME} ${TEST_SOURCE})
endforeach ()

if (DEFINED THREADS)
    add_definitions(-DTHREADS={THREADS})
endif ()

add_executable(all_tests ${TEST_SOURCES})
target_include_directories(all_tests PRIVATE ${GOOGLE_TEST_INCLUDE_DIRS})
target_link_libraries(all_tests gtest_main gmock_main NeuralNet::NeuralNet)
