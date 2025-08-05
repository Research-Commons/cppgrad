#include "tensor/tensorutils.hpp"
#include "autograd/function.hpp"
#include "tensor/tensor.hpp"

namespace cppgrad {

    // Clone tensor without tracking autograd.
    // Used when you want a pure data copy.
    Tensor TensorUtils::clone(const Tensor& input) {
        af::array cloned_data = input.data().copy();  // Deep copy of underlying array

        auto new_impl = std::make_shared<TensorImpl>(cloned_data, false);  // No autograd tracking
        return Tensor(new_impl);
    }

    // Clone tensor and preserve autograd tracking if input.requires_grad() is true.
    Tensor TensorUtils::clone_with_grad(const Tensor& input) {
        af::array cloned_data = input.data().copy();  // Deep copy
        bool req_grad = input.requires_grad();        // Carry over autograd flag

        auto new_impl = std::make_shared<TensorImpl>(cloned_data, req_grad);
        Tensor out(new_impl);

        // Register a backward function for autograd graph
        if (req_grad) {
            auto fn = std::make_shared<CloneFunction>();  // Forward clone op
            fn->inputs = { input.impl_ };                 // Save input tensor for backward
            out.impl_->grad_fn() = fn;                    // Attach backward function
        }

        return out;
    }

    // Matrix multiplication: performs af::matmul(a, b)
    // Returns a new tensor with autograd if either input requires gradients.
    Tensor TensorUtils::matmul(const Tensor &a, const Tensor &b) {
        const af::array& a_data = a.data();
        const af::array& b_data = b.data();

        af::array result_data = af::matmul(a_data, b_data);  // Matrix product: M×K × K×N = M×N

        // Enable gradient tracking if either input requires gradients
        auto result_impl = std::make_shared<TensorImpl>(
            result_data,
            /*requires_grad=*/a.requires_grad() || b.requires_grad()
        );

        Tensor result(result_impl);

        // If autograd enabled, attach MatMulFunction to compute backward later
        if (result.requires_grad()) {
            auto fn = std::make_shared<MatMulFunction>();
            fn->inputs = { a.impl_, b.impl_ };         // Save inputs for backward
            result_impl->grad_fn() = fn;               // Attach function to result
        }

        return result;
    }

    // Transpose a 2D tensor (swap rows and columns).
    // Keeps autograd flag from original tensor.
    Tensor TensorUtils::transpose(const Tensor &t) {
        af::array t_data = af::transpose(t.data());  // Transpose: M×N → N×M
        auto new_impl = std::make_shared<TensorImpl>(t_data, t.requires_grad());
        return {new_impl};  // Construct new Tensor
    }

} // namespace cppgrad

