#include <immintrin.h>
#include <cstdint>
#include <numeric>
#include <stdexcept>

#include "cppgrad/tensor/tensor.hpp"
#include "cppgrad/dispatcher/kernelRegistry.h" // adjust include to your actual path
#include "cppgrad/enums/dispatcherEnum.h"

namespace cppgrad{

// helper: check row-major contiguous (stride matches row-major)
bool is_row_major_contiguous(const std::vector<size_t>& shape,
                             const std::vector<float>& /*unused_data*/,
                             const std::vector<float>& /*unused_grad*/,
                             const std::vector<size_t>& stride) {
    // Compute expected strides for row-major
    if (shape.empty()) return true; // scalar is contiguous
    std::vector<size_t> expected(shape.size());
    size_t s = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        expected[i] = s;
        s *= shape[i];
    }
    return expected == stride;
}

// Slightly different overload: only uses shape & stride
bool is_row_major_contiguous_shape(const std::vector<size_t>& shape,
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

void add_f32_avx512(const cppgrad::Tensor& A,
                    const cppgrad::Tensor& B,
                    cppgrad::Tensor& Out) {
    using namespace cppgrad;

    // Basic sanity checks
    if (A.device_type() != DeviceType::CPU ||
        B.device_type() != DeviceType::CPU ||
        Out.device_type() != DeviceType::CPU) {
        throw std::runtime_error("AVX512 add: only CPU tensors supported");
    }

    if (A.shape() != B.shape() || A.shape() != Out.shape()) {
        throw std::runtime_error("AVX512 add: shape mismatch");
    }

    // Check contiguity (fast path only)
    if (!is_row_major_contiguous_shape(A.shape(), A.stride_sizet()) ||
        !is_row_major_contiguous_shape(B.shape(), B.stride_sizet()) ||
        !is_row_major_contiguous_shape(Out.shape(), Out.stride_sizet())) {
        // Fallback: shape/stride not contiguous — either call generic kernel
        // or perform a copy to contiguous buffers and call vectorized loop.
        // For now, throw so caller can fallback to generic kernel.
        throw std::runtime_error("AVX512 add: tensors are not contiguous; fallback required");
    }

    const float* a = A.data().data();
    const float* b = B.data().data();
    float* o = Out.data().data();
    const std::int64_t n = static_cast<std::int64_t>(Out.numel());

    const int stride = 16; // 16 floats per __m512
    std::int64_t i = 0;

    for (; i + stride <= n; i += stride) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vc = _mm512_add_ps(va, vb);
        _mm512_storeu_ps(o + i, vc);
    }

    int rem = static_cast<int>(n - i);
    if (rem > 0) {
        __mmask16 k = static_cast<__mmask16>((1u << rem) - 1u);
        __m512 va = _mm512_maskz_loadu_ps(k, a + i);
        __m512 vb = _mm512_maskz_loadu_ps(k, b + i);
        __m512 vc = _mm512_add_ps(va, vb);
        _mm512_mask_storeu_ps(o + i, k, vc);
    }
}

void mul_f32_avx512(const cppgrad::Tensor& A,
                    const cppgrad::Tensor& B,
                    cppgrad::Tensor& Out) {
    using namespace cppgrad;

    if (A.device_type() != DeviceType::CPU ||
        B.device_type() != DeviceType::CPU ||
        Out.device_type() != DeviceType::CPU) {
        throw std::runtime_error("AVX512 mul: only CPU tensors supported");
    }

    if (A.shape() != B.shape() || A.shape() != Out.shape()) {
        throw std::runtime_error("AVX512 mul: shape mismatch");
    }

    if (!is_row_major_contiguous_shape(A.shape(), A.stride_sizet()) ||
        !is_row_major_contiguous_shape(B.shape(), B.stride_sizet()) ||
        !is_row_major_contiguous_shape(Out.shape(), Out.stride_sizet())) {
        throw std::runtime_error("AVX512 mul: tensors are not contiguous; fallback required");
    }

    const float* a = A.data().data();
    const float* b = B.data().data();
    float* o = Out.data().data();
    const std::int64_t n = static_cast<std::int64_t>(Out.numel());

    const int stride = 16;
    std::int64_t i = 0;

    for (; i + stride <= n; i += stride) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vc = _mm512_mul_ps(va, vb);
        _mm512_storeu_ps(o + i, vc);
    }

    int rem = static_cast<int>(n - i);
    if (rem > 0) {
        __mmask16 k = static_cast<__mmask16>((1u << rem) - 1u);
        __m512 va = _mm512_maskz_loadu_ps(k, a + i);
        __m512 vb = _mm512_maskz_loadu_ps(k, b + i);
        __m512 vc = _mm512_mul_ps(va, vb);
        _mm512_mask_storeu_ps(o + i, k, vc);
    }
}

} // namespace cppgrad

// registration (called once at startup)
// Make sure this TU is linked into the final static lib (see CMake notes).
static bool _reg_avx512 = []{
    auto& R = cppgrad::KernelRegistry::instance();
    R.registerKernel(OpType::Add, DeviceType::CPU, CpuIsa::AVX512, cppgrad::add_f32_avx512);
    R.registerKernel(OpType::Mul, DeviceType::CPU, CpuIsa::AVX512, cppgrad::mul_f32_avx512);
    return true;
}();