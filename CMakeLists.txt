cmake_minimum_required(VERSION 3.16)
project(cppgrad)

set(CMAKE_CXX_STANDARD 20)
message(STATUS "Building cppgrad with build type: ${CMAKE_BUILD_TYPE}")

# ==== Options ====
option(USE_PREBUILT_ARRAYFIRE "Use prebuilt ArrayFire instead of building from source" OFF)
set(AF_BACKEND "cpu" CACHE STRING "ArrayFire backend: cpu, cuda, or metal")

# ==== Validate AF_BACKEND ====
if(NOT AF_BACKEND STREQUAL "cpu" AND
        NOT AF_BACKEND STREQUAL "cuda" AND
        NOT AF_BACKEND STREQUAL "metal")
    message(FATAL_ERROR "Invalid AF_BACKEND '${AF_BACKEND}'. Must be one of: cpu, cuda, metal")
endif()

# ==== ArrayFire Setup ====
if(USE_PREBUILT_ARRAYFIRE)
    message(STATUS "Using prebuilt ArrayFire")
    find_package(ArrayFire REQUIRED)
else()
    message(STATUS "Using bundled ArrayFire from third_party/arrayfire")
    add_subdirectory(third_party/arrayfire)
endif()

# ==== Catch2 Submodule ====
add_subdirectory(third_party/Catch2)

# ==== Source Files ====
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)

# ==== Create Static Library ====
add_library(cppgrad STATIC ${SRC_FILES})

# ==== Include Directories ====
target_include_directories(cppgrad
        PUBLIC  ${CMAKE_SOURCE_DIR}/include            # Enables: #include "cppgrad/tensor/..."
        PRIVATE ${CMAKE_SOURCE_DIR}/include/cppgrad    # Enables: #include "tensor/..."
)

# ==== Link ArrayFire Backend ====
target_link_libraries(cppgrad PUBLIC ArrayFire::af${AF_BACKEND})

# ==== Enable Testing ====
enable_testing()

# ==== Add Examples and Tests ====
add_subdirectory(examples)
add_subdirectory(test)
