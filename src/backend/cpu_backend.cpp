#include "cppgrad/backend/cpu_backend.h"
#include "cppgrad/utils/broadcastUtils.hpp"
#include <limits>

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


void cppgrad::CPU::subKernel(const Tensor &A, const Tensor &B, Tensor &out) {
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
        out.data()[pos] = A.data()[idxA] - B.data()[idxB];
    }
}

void cppgrad::CPU::divKernel(const Tensor &A, const Tensor &B, Tensor &out) {
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
        float denom = B.data()[idxB];
        if (denom == 0.0f) {
            out.data()[pos] = std::numeric_limits<float>::infinity();
        } else {
            out.data()[pos] = A.data()[idxA] / denom;
        }
    }
}

void cppgrad::CPU::addBackwardKernel(const Tensor &A, const Tensor &B, const Tensor &grad_out, Tensor &grad_a, Tensor &grad_b) {
    auto a_sh = A.shape();
    auto b_sh = B.shape();
    auto out_sh = compute_broadcast_shape(a_sh, b_sh);

    int n = static_cast<int>(out_sh.size());
    auto a_pad = pad_shape_right(a_sh, n);
    auto b_pad = pad_shape_right(b_sh, n);

    auto stride_out = compute_strides(out_sh);
    auto stride_a = compute_strides(a_pad);
    auto stride_b = compute_strides(b_pad);

    size_t out_total = numel(out_sh);
    size_t a_p_total = numel(a_pad);
    size_t b_p_total = numel(b_pad);

    // accumulate in padded buffers
    std::vector<float> a_padded(a_p_total, 0.0f);
    std::vector<float> b_padded(b_p_total, 0.0f);

    const auto &gout = grad_out.data();
    for (size_t pos = 0; pos < out_total; ++pos) {
        size_t idx_a = 0, idx_b = 0;
        for (int d = 0; d < n; ++d) {
            size_t coord = (pos / stride_out[(size_t)d]) % out_sh[(size_t)d];
            if (a_pad[(size_t)d] != 1) idx_a += coord * stride_a[(size_t)d];
            if (b_pad[(size_t)d] != 1) idx_b += coord * stride_b[(size_t)d];
        }
        float v = gout[pos];
        a_padded[idx_a] += v;
        b_padded[idx_b] += v;
    }

    // squeeze to original shapes
    auto a_squeezed = squeeze_padded_to_unpadded(a_padded, a_pad, a_sh);
    auto b_squeezed = squeeze_padded_to_unpadded(b_padded, b_pad, b_sh);

    grad_a.data() = std::move(a_squeezed);
    grad_b.data() = std::move(b_squeezed);
}

void cppgrad::CPU::mulBackwardKernel(const Tensor &A, const Tensor &B, const Tensor &grad_out,
                                    Tensor &grad_a, Tensor &grad_b) {
    auto a_sh = A.shape();
    auto b_sh = B.shape();
    auto out_sh = compute_broadcast_shape(a_sh, b_sh);

    int n = static_cast<int>(out_sh.size());
    auto a_pad = pad_shape_right(a_sh, n);
    auto b_pad = pad_shape_right(b_sh, n);

    auto stride_out = compute_strides(out_sh);
    auto stride_a = compute_strides(a_pad);
    auto stride_b = compute_strides(b_pad);

    size_t out_total = numel(out_sh);
    size_t a_p_total = numel(a_pad);
    size_t b_p_total = numel(b_pad);

    std::vector<float> a_padded(a_p_total, 0.0f);
    std::vector<float> b_padded(b_p_total, 0.0f);

    const auto &gout = grad_out.data();
    const auto &Adata = A.data();
    const auto &Bdata = B.data();

    for (size_t pos = 0; pos < out_total; ++pos) {
        size_t idx_a = 0, idx_b = 0;
        for (int d = 0; d < n; ++d) {
            size_t coord = (pos / stride_out[(size_t)d]) % out_sh[(size_t)d];
            if (a_pad[(size_t)d] != 1) idx_a += coord * stride_a[(size_t)d];
            if (b_pad[(size_t)d] != 1) idx_b += coord * stride_b[(size_t)d];
        }
        float gout_v = gout[pos];
        a_padded[idx_a] += gout_v * Bdata[idx_b];
        b_padded[idx_b] += gout_v * Adata[idx_a];
    }

    auto a_squeezed = squeeze_padded_to_unpadded(a_padded, a_pad, a_sh);
    auto b_squeezed = squeeze_padded_to_unpadded(b_padded, b_pad, b_sh);

    grad_a.data() = std::move(a_squeezed);
    grad_b.data() = std::move(b_squeezed);
}

void cppgrad::CPU::subBackwardKernel(const Tensor &A, const Tensor &B, const Tensor &grad_out,
                                     Tensor &grad_a, Tensor &grad_b) {
    auto a_sh = A.shape();
    auto b_sh = B.shape();
    auto out_sh = compute_broadcast_shape(a_sh, b_sh);

    int n = static_cast<int>(out_sh.size());
    auto a_pad = pad_shape_right(a_sh, n);
    auto b_pad = pad_shape_right(b_sh, n);

    auto stride_out = compute_strides(out_sh);
    auto stride_a = compute_strides(a_pad);
    auto stride_b = compute_strides(b_pad);

    size_t out_total = numel(out_sh);
    size_t a_p_total = numel(a_pad);
    size_t b_p_total = numel(b_pad);

    // accumulate in padded buffers
    std::vector<float> a_padded(a_p_total, 0.0f);
    std::vector<float> b_padded(b_p_total, 0.0f);

    const auto &gout = grad_out.data();
    for (size_t pos = 0; pos < out_total; ++pos) {
        size_t idx_a = 0, idx_b = 0;
        for (int d = 0; d < n; ++d) {
            size_t coord = (pos / stride_out[(size_t)d]) % out_sh[(size_t)d];
            if (a_pad[(size_t)d] != 1) idx_a += coord * stride_a[(size_t)d];
            if (b_pad[(size_t)d] != 1) idx_b += coord * stride_b[(size_t)d];
        }
        float v = gout[pos];
        a_padded[idx_a] += v;    // grad wrt A is +grad_out
        b_padded[idx_b] += -v;   // grad wrt B is -grad_out
    }

    // squeeze to original shapes
    auto a_squeezed = squeeze_padded_to_unpadded(a_padded, a_pad, a_sh);
    auto b_squeezed = squeeze_padded_to_unpadded(b_padded, b_pad, b_sh);

    grad_a.data() = std::move(a_squeezed);
    grad_b.data() = std::move(b_squeezed);
}

void cppgrad::CPU::divBackwardKernel(const Tensor &A, const Tensor &B, const Tensor &grad_out,
                                     Tensor &grad_a, Tensor &grad_b) {
    auto a_sh = A.shape();
    auto b_sh = B.shape();
    auto out_sh = compute_broadcast_shape(a_sh, b_sh);

    int n = static_cast<int>(out_sh.size());
    auto a_pad = pad_shape_right(a_sh, n);
    auto b_pad = pad_shape_right(b_sh, n);

    auto stride_out = compute_strides(out_sh);
    auto stride_a = compute_strides(a_pad);
    auto stride_b = compute_strides(b_pad);

    size_t out_total = numel(out_sh);
    size_t a_p_total = numel(a_pad);
    size_t b_p_total = numel(b_pad);

    std::vector<float> a_padded(a_p_total, 0.0f);
    std::vector<float> b_padded(b_p_total, 0.0f);

    const auto &gout = grad_out.data();
    const auto &Adata = A.data();
    const auto &Bdata = B.data();

    for (size_t pos = 0; pos < out_total; ++pos) {
        size_t idx_a = 0, idx_b = 0;
        for (int d = 0; d < n; ++d) {
            size_t coord = (pos / stride_out[(size_t)d]) % out_sh[(size_t)d];
            if (a_pad[(size_t)d] != 1) idx_a += coord * stride_a[(size_t)d];
            if (b_pad[(size_t)d] != 1) idx_b += coord * stride_b[(size_t)d];
        }

        float gout_v = gout[pos];
        float aval = Adata[idx_a];
        float bval = Bdata[idx_b];

        // grad wrt A: grad_out / b  (handle b==0)
        float contrib_a = (bval == 0.0f) ? std::numeric_limits<float>::infinity() : (gout_v / bval);
        // grad wrt B: -grad_out * a / (b*b)  (handle denom==0)
        float denom = bval * bval;
        float contrib_b = (denom == 0.0f) ? -std::numeric_limits<float>::infinity()
                                          : (-gout_v * aval / denom);

        a_padded[idx_a] += contrib_a;
        b_padded[idx_b] += contrib_b;
    }

    auto a_squeezed = squeeze_padded_to_unpadded(a_padded, a_pad, a_sh);
    auto b_squeezed = squeeze_padded_to_unpadded(b_padded, b_pad, b_sh);

    grad_a.data() = std::move(a_squeezed);
    grad_b.data() = std::move(b_squeezed);
}

