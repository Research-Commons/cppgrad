cmake_minimum_required(VERSION 3.16)

project(cppgrad)

set(CMAKE_CXX_STANDARD 20)

#-- if you want to build third_party/arrayfire from source
add_subdirectory(third_party/arrayfire)
# Add Catch2 submodule
add_subdirectory(third_party/Catch2)

#-- if you want to use pre built binaries for ArrayFire
#find_package(ArrayFire REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS src/*.cpp)

#will change to a library build layer
#add_executable(cppgrad ${SRC_FILES})

add_library(cppgrad STATIC ${SRC_FILES})

#-- For cpu build
target_link_libraries(cppgrad PUBLIC ArrayFire::afcpu) #change to afcuda for cuda build

target_include_directories(cppgrad PUBLIC ${CMAKE_SOURCE_DIR}/include)

enable_testing()

#-- Add examples and tests folders
add_subdirectory(examples)
add_subdirectory(test)

