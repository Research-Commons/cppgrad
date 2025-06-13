#pragma once
#include "tensor.hpp"

namespace cppgrad::ops {

TensorRef add(const TensorRef& a, const TensorRef& b);
TensorRef mul(const TensorRef& a, const TensorRef& b);
TensorRef matmul(const TensorRef& a, const TensorRef& b);

}
