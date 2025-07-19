#pragma once
#include "../tensor/tensor.hpp"

namespace cppgrad {
    Tensor operator*(const Tensor& a, const Tensor& b);

}