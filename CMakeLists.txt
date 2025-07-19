cmake_minimum_required(VERSION 3.16)

project(cppgrad)

set(CMAKE_CXX_STANDARD 20)

#-- if you want to build third_party/arrayfire from source
add_subdirectory(third_party/arrayfire)

#-- if you want to use pre built binaries for ArrayFire
#find_package(ArrayFire REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS src/*.cpp)

#will change to a library build layer
add_executable(cppgrad ${SRC_FILES}
        include/autograd/autogradmeta.h
        src/autograd/autogradmeta.cpp
        include/autograd/function.h
        src/autograd/function.cpp
        include/tensor/tensorimpl.h
        src/tensor/tensorimpl.cpp)

#-- For cpu build
target_link_libraries(cppgrad PRIVATE ArrayFire::afcpu)

#-- For cuda build ( make sure to have cuda installed first )
#target_link_libraries(cppgrad PRIVATE ArrayFire::afcuda)

#-- ignoring this for now, will build it later once tensor class is properly implemented
# add_executable(test_xor examples/xor_train.cpp)
# target_link_libraries(test_xor PRIVATE cppgrad)