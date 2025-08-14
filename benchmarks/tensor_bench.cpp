#include <cuda_runtime_api.h>
#include <benchmark/benchmark.h>

#include "cppgrad/tensor/tensor.hpp"
#include "cppgrad/dispatcher/kernelRegistry.h"
#include "cppgrad/backend/cpu_backend.h"
#include "cppgrad/backend/cuda_backend.hpp"


using namespace cppgrad;

static void BM_Add_CPU(benchmark::State& state) {
    // Setup once before the loop
    KernelRegistry::instance().registerKernel(OpType::Add, DeviceType::CPU, CPU::addKernel);

    Tensor A({10000, 10000}, 5.f, DeviceType::CPU);
    Tensor B({10000, 10000}, 5.f, DeviceType::CPU);

    for (auto _ : state) {
        Tensor C = A + B;
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Add_CUDA(benchmark::State& state) {
    // Setup once before the loop
    KernelRegistry::instance().registerKernel(OpType::Add, DeviceType::CUDA, CUDA::addKernel);

    Tensor A({10000, 10000}, 5.f, DeviceType::CUDA);
    Tensor B({10000, 10000}, 5.f, DeviceType::CUDA);

    for (auto _ : state) {
        Tensor C = A + B;
        benchmark::DoNotOptimize(C);
        cudaDeviceSynchronize(); // Ensure timing includes GPU execution
    }
}

BENCHMARK(BM_Add_CPU);
BENCHMARK(BM_Add_CUDA);

BENCHMARK_MAIN();