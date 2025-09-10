cmake_minimum_required(VERSION 3.20)
project(cppgrad LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
message(STATUS "Building cppgrad with build type: ${CMAKE_BUILD_TYPE}")

find_package(CUDAToolkit REQUIRED)

# ==== Options ====
option(USE_PREBUILT_ARRAYFIRE "Use prebuilt ArrayFire instead of building from source" OFF)
set(AF_BACKEND "cpu" CACHE STRING "ArrayFire backend: cpu, cuda, or metal")
# Build AVX objects (compile-time flags). Turn off if compiler can't support or you want smaller binary.
option(BUILD_AVX2  "Build AVX2 optimized kernels (object library)"  ON)
option(BUILD_AVX512 "Build AVX-512 optimized kernels (object library)" ON)

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
file(GLOB_RECURSE ALL_SRC CONFIGURE_DEPENDS
        ${CMAKE_SOURCE_DIR}/src/*.cpp
        ${CMAKE_SOURCE_DIR}/src/*.cu
)

# remove ISA path files from ALL_SRC so they are compiled separately with special flags
# NOTE: adjust the regex if your ISA path is different
list(FILTER ALL_SRC EXCLUDE REGEX ".*/src/backend/isa/.*")

# ==== AVX sources (explicit) ====
file(GLOB AVX2_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/backend/isa/*avx2.cpp)
file(GLOB AVX512_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/backend/isa/*_avx512.cpp)

# ==== Create object libraries for ISA specializations ====
if(BUILD_AVX2 AND AVX2_SRC)
    add_library(cppgrad_isa_avx2 OBJECT ${AVX2_SRC})
    if(MSVC)
        target_compile_options(cppgrad_isa_avx2 PRIVATE /arch:AVX2)
    else()
        target_compile_options(cppgrad_isa_avx2 PRIVATE -mavx2 -mfma -O3)
        # if you prefer -march=native, replace/add it here
    endif()
    target_include_directories(cppgrad_isa_avx2 PRIVATE ${CMAKE_SOURCE_DIR}/include)
endif()

if(BUILD_AVX512 AND AVX512_SRC)
    add_library(cppgrad_isa_avx512 OBJECT ${AVX512_SRC})
    if(MSVC)
        # MSVC's AVX-512 support varies by version/toolset. Use with caution.
        target_compile_options(cppgrad_isa_avx512 PRIVATE /arch:AVX512)
    else()
        target_compile_options(cppgrad_isa_avx512 PRIVATE -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -O3)
    endif()
    target_include_directories(cppgrad_isa_avx512 PRIVATE ${CMAKE_SOURCE_DIR}/include)
endif()

# ==== Create final static library, including object libs if present ====
# Note: add the generic source files plus any object lib objects
set(CPPGRAD_OBJ_SOURCES ${ALL_SRC})

# Construct the add_library call with OBJECTS expanded if they exist
if(BUILD_AVX2 AND AVX2_SRC)
    list(APPEND CPPGRAD_OBJ_SOURCES $<TARGET_OBJECTS:cppgrad_isa_avx2>)
endif()
if(BUILD_AVX512 AND AVX512_SRC)
    list(APPEND CPPGRAD_OBJ_SOURCES $<TARGET_OBJECTS:cppgrad_isa_avx512>)
endif()

# ==== Create Static Library ====
add_library(cppgrad STATIC ${CPPGRAD_OBJ_SOURCES})

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