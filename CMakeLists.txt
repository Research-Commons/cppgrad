cmake_minimum_required(VERSION 3.16)
project(cppgrad LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)

add_library(cppgrad
  src/tensor.cpp
  src/autograd.cpp
  src/op.cpp
  src/layer.cpp
  src/optimizer.cpp
)

add_executable(test_xor examples/xor_train.cpp)
target_link_libraries(test_xor cppgrad)
