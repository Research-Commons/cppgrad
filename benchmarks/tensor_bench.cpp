#include <benchmark/benchmark.h>
#include "cppgrad/tensor/tensor.hpp"

// Benchmark: Elementwise addition of two N×N tensors
static void BM_TensorAdd(benchmark::State& state) {
    std::vector<unsigned long>::size_type N = state.range(0);
    // Allocate input tensors once
    cppgrad::Tensor a = cppgrad::Tensor::randn({N, N}, /*requires_grad=*/false);
    cppgrad::Tensor b = cppgrad::Tensor::randn({N, N}, /*requires_grad=*/false);

    for (auto _ : state) {
        // This code is measured
        auto c = a + b;
        af::eval(c.data());
        benchmark::DoNotOptimize(c);
    }
    // Optional: report complexity based on N
    state.SetComplexityN(N);
}

// Register the benchmark for multiple sizes
BENCHMARK(BM_TensorAdd)
    ->Arg(500)
    ->Arg(1000)
    ->Arg(2000)
    ->Complexity();

BENCHMARK_MAIN();
