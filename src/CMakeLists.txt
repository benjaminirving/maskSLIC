cmake_minimum_required(VERSION 3.2)
project(src)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    processing.cpp
    processing.h)

add_executable(Test ${SOURCE_FILES} Test.cpp)