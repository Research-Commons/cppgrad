#include <cuda_runtime_api.h>
#include <benchmark/benchmark.h>

#include "cppgrad/tensor/tensor.hpp"
#include "cppgrad/dispatcher/kernelRegistry.h"
#include "cppgrad/backend/cpu_backend.h"
#include "cppgrad/backend/cuda_backend.hpp"
#include "cppgrad/backend/isa/avx2.hpp"


using namespace cppgrad;

static void BM_Add_CPU(benchmark::State& state) {
    // Setup once before the loop
    KernelRegistry::instance().registerKernel(OpType::Add, DeviceType::CPU, CPU::addKernel);

    Tensor A({10000, 10000}, 5.f, false, DeviceType::CPU);
    Tensor B({10000, 10000}, 5.f, false, DeviceType::CPU);

    for (auto _ : state) {
        Tensor C = A + B;
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Add_CUDA(benchmark::State& state) {
    // Setup once before the loop
    KernelRegistry::instance().registerKernel(OpType::Add, DeviceType::CUDA, CUDA::addKernel);

    Tensor A({10000, 10000}, 5.f, false, DeviceType::CUDA);
    Tensor B({10000, 10000}, 5.f, false, DeviceType::CUDA);

    for (auto _ : state) {
        Tensor C = A + B;
        benchmark::DoNotOptimize(C);
        cudaDeviceSynchronize(); // Ensure timing includes GPU execution
    }
}

static void BM_Mul_CPU(benchmark::State& state) {
    // Setup once before the loop
    KernelRegistry::instance().registerKernel(OpType::Mul, DeviceType::CPU, CPU::mulKernel);

    Tensor A({10000, 10000}, 5.f, false, DeviceType::CPU);
    Tensor B({10000, 10000}, 5.f, false, DeviceType::CPU);

    for (auto _ : state) {
        Tensor C = A * B;
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Mul_CUDA(benchmark::State& state) {
    // Setup once before the loop
    KernelRegistry::instance().registerKernel(OpType::Mul, DeviceType::CUDA, CUDA::mulKernel);

    Tensor A({10000, 10000}, 5.f, false, DeviceType::CUDA);
    Tensor B({10000, 10000}, 5.f, false, DeviceType::CUDA);

    for (auto _ : state) {
        Tensor C = A * B;
        benchmark::DoNotOptimize(C);
        cudaDeviceSynchronize(); // Ensure timing includes GPU execution
    }
}

static void BM_Add_AVX2(benchmark::State& state) {
    // Setup once before the loop
    KernelRegistry::instance().registerKernel(OpType::Add, DeviceType::CPU, CpuIsa::AVX2, cppgrad::add_f32_avx2);

    Tensor A({10000, 10000}, 5.f, false, DeviceType::CPU);
    Tensor B({10000, 10000}, 5.f, false, DeviceType::CPU);

    for (auto _ : state) {
        Tensor C = A + B;
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Mul_AVX2(benchmark::State& state) {
    // Setup once before the loop
    KernelRegistry::instance().registerKernel(OpType::Mul, DeviceType::CPU, CpuIsa::AVX2, cppgrad::mul_f32_avx2);

    Tensor A({10000, 10000}, 5.f, false, DeviceType::CPU);
    Tensor B({10000, 10000}, 5.f, false, DeviceType::CPU);

    for (auto _ : state) {
        Tensor C = A * B;
        benchmark::DoNotOptimize(C);
    }
}

BENCHMARK(BM_Add_CPU);
BENCHMARK(BM_Add_CUDA);
BENCHMARK(BM_Mul_CPU);
BENCHMARK(BM_Mul_CUDA);
BENCHMARK(BM_Add_AVX2);
BENCHMARK(BM_Mul_AVX2);

BENCHMARK_MAIN();