#pragma once

#include "cppgrad/tensor/tensor.hpp"

namespace cppgrad {
    class CPU {
    public:
        static void addKernel(const Tensor& A, const Tensor& B, Tensor& out);
        static void mulKernel(const Tensor& A, const Tensor& B, Tensor& out);
        static void subKernel(const Tensor& A, const Tensor& B, Tensor& out);
        static void divKernel(const Tensor& A, const Tensor& B, Tensor& out);

        static void addBackwardKernel(const Tensor &A, const Tensor &B, const Tensor &grad_out,
                                    Tensor &grad_a, Tensor &grad_b);
        static void mulBackwardKernel(const Tensor &A, const Tensor &B, const Tensor &grad_out,
                                    Tensor &grad_a, Tensor &grad_b);
        static void subBackwardKernel(const Tensor &A, const Tensor &B, const Tensor &grad_out,
                                     Tensor &grad_a, Tensor &grad_b);
        static void divBackwardKernel(const Tensor &A, const Tensor &B, const Tensor &grad_out,
                                     Tensor &grad_a, Tensor &grad_b);
    };
}