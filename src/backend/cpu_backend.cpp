#include "cppgrad/backend/cpu_backend.h"

void cppgrad::CPU::addKernel(const Tensor &A, const Tensor &B, Tensor &out) {
    const auto& a_sh = A.shape();
    const auto& b_sh = B.shape();
    const auto& out_sh = out.shape();
    size_t n = out_sh.size();

    // Build padded shapes
    std::vector<size_t> a_pad(n, 1), b_pad(n, 1);
    size_t na = a_sh.size(), nb = b_sh.size();
    for (size_t i = 0; i < n; ++i) {
        a_pad[i] = (i < n-na ? 1 : a_sh[i-(n-na)]);
        b_pad[i] = (i < n-nb ? 1 : b_sh[i-(n-nb)]);
    }

    // Compute prefix (stride) for A, B, and output
    std::vector<size_t> strideA(n), strideB(n), strideOut(n);
    if (n > 0) {
        strideA[n-1] = strideB[n-1] = strideOut[n-1] = 1;
    }
    for (int i = (int)n-2; i >= 0; --i) {
        strideA[i]   = strideA[i+1] * a_pad[i+1];
        strideB[i]   = strideB[i+1] * b_pad[i+1];
        strideOut[i] = strideOut[i+1] * out_sh[i+1];
    }

    // Elementwise loop over output tensor
    size_t total = 1;
    for (size_t dim : out_sh) total *= dim;
    for (size_t pos = 0; pos < total; ++pos) {
        size_t idxA = 0, idxB = 0;
        // Convert flat index to multi-index, then to each input index
        for (size_t dim = 0; dim < n; ++dim) {
            size_t i = (pos / strideOut[dim]) % out_sh[dim];
            if (a_pad[dim] != 1) idxA += i * strideA[dim];
            if (b_pad[dim] != 1) idxB += i * strideB[dim];
        }
        out.data()[pos] = A.data()[idxA] + B.data()[idxB];
    }
}

void cppgrad::CPU::mulKernel(const Tensor &A, const Tensor &B, Tensor &out) {
    const auto& a_sh = A.shape();
    const auto& b_sh = B.shape();
    const auto& out_sh = out.shape();
    size_t n = out_sh.size();

    std::vector<size_t> a_pad(n, 1), b_pad(n, 1);
    size_t na = a_sh.size(), nb = b_sh.size();
    for (size_t i = 0; i < n; ++i) {
        a_pad[i] = (i < n-na ? 1 : a_sh[i-(n-na)]);
        b_pad[i] = (i < n-nb ? 1 : b_sh[i-(n-nb)]);
    }

    std::vector<size_t> strideA(n), strideB(n), strideOut(n);
    if (n > 0) {
        strideA[n-1] = strideB[n-1] = strideOut[n-1] = 1;
    }
    for (int i = (int)n-2; i >= 0; --i) {
        strideA[i]   = strideA[i+1] * a_pad[i+1];
        strideB[i]   = strideB[i+1] * b_pad[i+1];
        strideOut[i] = strideOut[i+1] * out_sh[i+1];
    }

    size_t total = 1;
    for (size_t dim : out_sh) total *= dim;
    for (size_t pos = 0; pos < total; ++pos) {
        size_t idxA = 0, idxB = 0;
        for (size_t dim = 0; dim < n; ++dim) {
            size_t i = (pos / strideOut[dim]) % out_sh[dim];
            if (a_pad[dim] != 1) idxA += i * strideA[dim];
            if (b_pad[dim] != 1) idxB += i * strideB[dim];
        }
        out.data()[pos] = A.data()[idxA] * B.data()[idxB];
    }
}
