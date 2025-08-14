#pragma once

#include "cppgrad/tensor/tensor.hpp"

namespace cppgrad {

    class CUDA {
    public:

        static void addKernel(const Tensor& A, const Tensor& B, Tensor& out);
    };

} // namespace cppgrad