// #pragma once
//
// namespace cppgrad {
//
//     /**
//      * @file tensorutils.hpp
//      * @brief Utility functions for working with cppgrad Tensors.
//      *
//      * `TensorUtils` is a helper class that provides static utility functions for common
//      * tensor operations that either require lower-level manipulation of the Tensor internals
//      * or are better kept separate from the main `Tensor` class.
//      *
//      * Current responsibilities include:
//      * - Cloning tensors (with and without autograd tracking)
//      * - Matrix multiplication
//      * - Transposing tensors
//      *
//      * Design Notes:
//      * - These are stateless operations and are implemented as static methods.
//      * - `clone_with_grad()` attaches a `CloneFunction` node to the autograd graph,
//      *   unlike `clone()`, which performs a raw copy without tracking gradients.
//      * - This utility layer helps keep the `Tensor` class focused and minimal.
//      *
//      * Analogy: Similar to PyTorch's `at::native` or internal helper functions
//      * that operate on `TensorImpl` or handle autograd edge cases.
//     */
//
//     class Tensor;
//
//     class TensorUtils {
//         public:
//             static Tensor clone(const Tensor& input);
//             static Tensor clone_with_grad(const Tensor& input);
//             static Tensor matmul(const Tensor& a, const Tensor& b);
//             static Tensor transpose(const Tensor& t);
//     };
//
// }