#pragma once

#include "tensor.hpp"

namespace cppgrad {

    class TensorUtils {
        public:
            static Tensor clone(const Tensor& input);
            static Tensor clone_with_grad(const Tensor& input);
    };

}