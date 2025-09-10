#include <immintrin.h>
#include "cppgrad/backend/isa/avx2.hpp"
#include "cppgrad/dispatcher/kernelRegistry.h"
#include "cppgrad/enums/dispatcherEnum.h"

namespace cppgrad {

// Reuse the same contiguity checker name you used for AVX-512.
// If it's in a header, include it instead of duplicating. For simplicity,
// we replicate the small helper here (matches the AVX512 version).
static bool is_row_major_contiguous_shape(const std::vector<size_t>& shape,
                                          const std::vector<size_t>& stride) {
    if (shape.empty()) return true;
    std::vector<size_t> expected(shape.size());
    size_t s = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        expected[i] = s;
        s *= shape[i];
    }
    return expected == stride;
}

void add_f32_avx2(const cppgrad::Tensor& A,
                  const cppgrad::Tensor& B,
                  cppgrad::Tensor& Out) {
    // Basic sanity checks
    if (A.device_type() != DeviceType::CPU ||
        B.device_type() != DeviceType::CPU ||
        Out.device_type() != DeviceType::CPU) {
        throw std::runtime_error("AVX2 add: only CPU tensors supported");
    }

    if (A.shape() != B.shape() || A.shape() != Out.shape()) {
        throw std::runtime_error("AVX2 add: shape mismatch");
    }

    // if (!is_row_major_contiguous_shape(A.shape(), A.stride_sizet()) ||
    //     !is_row_major_contiguous_shape(B.shape(), B.stride_sizet()) ||
    //     !is_row_major_contiguous_shape(Out.shape(), Out.stride_sizet())) {
    //     throw std::runtime_error("AVX2 add: tensors are not contiguous; fallback required");
    // }

    const float* a = A.data().data();
    const float* b = B.data().data();
    float* o = Out.data().data();
    const std::int64_t n = static_cast<std::int64_t>(Out.numel());

    const int stride = 8; // 8 floats per __m256
    std::int64_t i = 0;

    for (; i + stride <= n; i += stride) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(o + i, vc);
    }

    // scalar tail
    for (; i < n; ++i) {
        o[i] = a[i] + b[i];
    }
}

void mul_f32_avx2(const cppgrad::Tensor& A,
                  const cppgrad::Tensor& B,
                  cppgrad::Tensor& Out) {
    // Basic sanity checks
    if (A.device_type() != DeviceType::CPU ||
        B.device_type() != DeviceType::CPU ||
        Out.device_type() != DeviceType::CPU) {
        throw std::runtime_error("AVX2 mul: only CPU tensors supported");
    }

    if (A.shape() != B.shape() || A.shape() != Out.shape()) {
        throw std::runtime_error("AVX2 mul: shape mismatch");
    }

    // if (!is_row_major_contiguous_shape(A.shape(), A.stride_sizet()) ||
    //     !is_row_major_contiguous_shape(B.shape(), B.stride_sizet()) ||
    //     !is_row_major_contiguous_shape(Out.shape(), Out.stride_sizet())) {
    //     throw std::runtime_error("AVX2 mul: tensors are not contiguous; fallback required");
    // }

    const float* a = A.data().data();
    const float* b = B.data().data();
    float* o = Out.data().data();
    const std::int64_t n = static_cast<std::int64_t>(Out.numel());

    const int stride = 8; // 8 floats
    std::int64_t i = 0;

    for (; i + stride <= n; i += stride) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(o + i, vc);
    }

    // scalar tail
    for (; i < n; ++i) {
        o[i] = a[i] * b[i];
    }
}

} // namespace cppgrad


static bool _reg_avx2 = []{
    auto &R = cppgrad::KernelRegistry::instance();
    R.registerKernel(OpType::Add, DeviceType::CPU, CpuIsa::AVX2, cppgrad::add_f32_avx2);
    R.registerKernel(OpType::Mul, DeviceType::CPU, CpuIsa::AVX2, cppgrad::mul_f32_avx2);
    return true;
}();
