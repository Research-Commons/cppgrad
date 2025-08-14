#include "cppgrad/backend/cuda_backend.hpp"
#include "cppgrad/dispatcher/kernelRegistry.h" // adjust include path for KernelRegistry/OpType/DeviceType
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) do {                                \
    cudaError_t err = (call);                                \
    if (err != cudaSuccess) {                                \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " (" << err << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("CUDA call failed");        \
    }                                                        \
} while(0)

extern "C" __global__
void add_kernel_broadcast(const float* a, const float* b, float* out,
                          size_t total,
                          const size_t* strideA_eff,
                          const size_t* strideB_eff,
                          const size_t* strideOut,
                          const size_t* out_sh,
                          int n_dims) {
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= total) return;

    size_t idxA = 0;
    size_t idxB = 0;
    // map flat pos -> coordinates and then to input indices using effective strides
    for (int d = 0; d < n_dims; ++d) {
        size_t coord = (pos / strideOut[d]) % out_sh[d];
        idxA += coord * strideA_eff[d];
        idxB += coord * strideB_eff[d];
    }
    out[pos] = a[idxA] + b[idxB];
}

namespace cppgrad {

    void CUDA::addKernel(const Tensor& A, const Tensor& B, Tensor& out) {
        // Prepare shapes and strides (same logic as CPU)
        const auto& a_sh = A.shape();
        const auto& b_sh = B.shape();
        const auto& out_sh = out.shape();
        int n = static_cast<int>(out_sh.size());

        // Compute padded shapes (align right)
        std::vector<size_t> a_pad(n, 1), b_pad(n, 1);
        size_t na = a_sh.size(), nb = b_sh.size();
        for (int i = 0; i < n; ++i) {
            a_pad[i] = (i < n - (int)na ? 1 : a_sh[i - (n - (int)na)]);
            b_pad[i] = (i < n - (int)nb ? 1 : b_sh[i - (n - (int)nb)]);
        }

        // Compute strides for A,B,out (C-order)
        std::vector<size_t> strideA(n), strideB(n), strideOut(n);
        if (n > 0) {
            strideA[n-1] = strideB[n-1] = strideOut[n-1] = 1;
        }
        for (int i = n - 2; i >= 0; --i) {
            strideA[i] = strideA[i+1] * a_pad[i+1];
            strideB[i] = strideB[i+1] * b_pad[i+1];
            strideOut[i] = strideOut[i+1] * out_sh[i+1];
        }

        // Effective strides: zero for broadcasted dimensions
        std::vector<size_t> strideA_eff(n), strideB_eff(n);
        for (int i = 0; i < n; ++i) {
            strideA_eff[i] = (a_pad[i] == 1) ? 0 : strideA[i];
            strideB_eff[i] = (b_pad[i] == 1) ? 0 : strideB[i];
        }

        // Total number of output elements
        size_t total = 1;
        for (size_t d : out_sh) total *= d;
        if (total == 0) return;

        // Allocate device buffers
        float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
        size_t *d_strideA_eff = nullptr, *d_strideB_eff = nullptr, *d_strideOut = nullptr, *d_out_sh = nullptr;

        // Note: A.data() and B.data() hold compact storage for their shapes (C-contiguous).
        // We copy raw flattened arrays. The index calculation using stride_eff will map correctly.
        CUDA_CHECK(cudaMalloc(&d_a, A.data().size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b, B.data().size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_a, A.data().data(), A.data().size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, B.data().data(), B.data().size() * sizeof(float), cudaMemcpyHostToDevice));

        // copy stride arrays and output shape/strides
        CUDA_CHECK(cudaMalloc(&d_strideA_eff, n * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_strideB_eff, n * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_strideOut, n * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_out_sh, n * sizeof(size_t)));

        CUDA_CHECK(cudaMemcpy(d_strideA_eff, strideA_eff.data(), n * sizeof(size_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_strideB_eff, strideB_eff.data(), n * sizeof(size_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_strideOut, strideOut.data(), n * sizeof(size_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_out_sh, out_sh.data(), n * sizeof(size_t), cudaMemcpyHostToDevice));

        // Launch kernel
        const int block = 256;
        int grid = static_cast<int>((total + block - 1) / block);
        add_kernel_broadcast<<<grid, block>>>(d_a, d_b, d_out, total,
                                              d_strideA_eff, d_strideB_eff, d_strideOut, d_out_sh, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy result back
        CUDA_CHECK(cudaMemcpy(out.data().data(), d_out, total * sizeof(float), cudaMemcpyDeviceToHost));

        // Free device memory
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_strideA_eff));
        CUDA_CHECK(cudaFree(d_strideB_eff));
        CUDA_CHECK(cudaFree(d_strideOut));
        CUDA_CHECK(cudaFree(d_out_sh));
    }
} // namespace cppgrad