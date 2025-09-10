#pragma once
#include <vector>
#include "cppgrad/tensor/tensor.hpp"

namespace cppgrad {
    void add_f32_avx2(const Tensor& A, const Tensor& B, Tensor& Out);
    void mul_f32_avx2(const Tensor& A, const Tensor& B, Tensor& Out);
} // namespace cppgrad



