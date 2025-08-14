#pragma once

#include "cppgrad/tensor/tensor.hpp"

namespace cppgrad {
    class CPU {
    public:
        static void addKernel(const Tensor& A, const Tensor& B, Tensor& out);
        static void mulKernel(const Tensor& A, const Tensor& B, Tensor& out);
        static void subKernel(const Tensor& A, const Tensor& B, Tensor& out);
        static void divKernel(const Tensor& A, const Tensor& B, Tensor& out);
    };
}