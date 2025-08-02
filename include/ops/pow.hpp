#pragma once
#include "tensor/tensor.hpp"

namespace cppgrad {
     Tensor pow(const Tensor& base, const Tensor& exponent);
     Tensor pow(const Tensor& base, float scalar);
     Tensor pow(float scalar, const Tensor& exponent);
}