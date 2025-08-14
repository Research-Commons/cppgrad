file(GLOB_RECURSE BENCH_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(cppgrad_bench ${BENCH_FILES})

target_link_libraries(cppgrad_bench
        PRIVATE
        cppgrad
        benchmark::benchmark
        benchmark::benchmark_main
        CUDA::cudart
)

target_include_directories(cppgrad_bench
        PRIVATE ${CMAKE_SOURCE_DIR}/include
)