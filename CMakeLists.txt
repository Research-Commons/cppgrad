cmake_minimum_required(VERSION 3.20)
project(cppgrad LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
message(STATUS "Building cppgrad with build type: ${CMAKE_BUILD_TYPE}")

find_package(CUDAToolkit REQUIRED)

# ==== Options ====
option(USE_PREBUILT_ARRAYFIRE "Use prebuilt ArrayFire instead of building from source" OFF)
set(AF_BACKEND "cpu" CACHE STRING "ArrayFire backend: cpu, cuda, or metal")

# ==== Validate AF_BACKEND ====
if(NOT AF_BACKEND STREQUAL "cpu" AND
        NOT AF_BACKEND STREQUAL "cuda" AND
        NOT AF_BACKEND STREQUAL "metal")
    message(FATAL_ERROR "Invalid AF_BACKEND '${AF_BACKEND}'. Must be one of: cpu, cuda, metal")
endif()

## ==== ArrayFire Setup ====
#if(USE_PREBUILT_ARRAYFIRE)
#    message(STATUS "Using prebuilt ArrayFire")
#    find_package(ArrayFire REQUIRED)
#else()
#    message(STATUS "Using bundled ArrayFire from third_party/arrayfire")
#    add_subdirectory(third_party/arrayfire)
#endif()

# ==== Catch2 Submodule ====
add_subdirectory(third_party/Catch2)

# ==== Google Benchmark Submodule ====
message(STATUS "Adding Google Benchmark from third_party/googlebenchmark")
set(BENCHMARK_ENABLE_TESTING OFF) # disable GB's own tests
set(BENCHMARK_ENABLE_INSTALL OFF)
add_subdirectory(third_party/googlebenchmark)

# ==== Source Files ====
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
        ${CMAKE_SOURCE_DIR}/src/*.cpp
        ${CMAKE_SOURCE_DIR}/src/*.cu
)

# ==== Create Static Library ====
add_library(cppgrad STATIC ${SRC_FILES})

set_target_properties(cppgrad PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)

# ==== Include Directories ====
target_include_directories(cppgrad
        PUBLIC  ${CMAKE_SOURCE_DIR}/include            # Enables: #include "cppgrad/tensor/..."
        PRIVATE ${CMAKE_SOURCE_DIR}/include/cppgrad    # Enables: #include "tensor/..."
)

## ==== Link ArrayFire Backend ====
target_link_libraries(cppgrad PUBLIC CUDA::cudart)

# ==== Enable Testing ====
enable_testing()

# ==== Add subdirectories====
add_subdirectory(examples)
add_subdirectory(test)
add_subdirectory(benchmarks)