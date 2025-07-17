#pragma once
#include "tensor.hpp"

namespace cppgrad {

    Tensor add(const Tensor& a, const Tensor& b);
    Tensor operator+(const Tensor& a, const Tensor& b);

}