#include "ops/mul.hpp"
#include "autograd/function.hpp"
#include "tensor/tensor.hpp"

#include <stdexcept>

#include "dispatcher/kernelRegistry.h"
#include "ops/add.hpp"

namespace cppgrad {

    Tensor operator*(const Tensor& a, const Tensor& b) {
        if (a.device() != a.device()) {
            throw std::runtime_error("Device mismatch in mul");
        }
        std::vector<size_t> out_shape = computeBroadcastShape(a.shape(), a.shape());
        Tensor out(out_shape, 0.0f, a.device());
        KernelRegistry::instance()
            .getKernel(OpType::Mul, a.device())(a, b, out);
        return out;
    }


    // Tensor operator*(const Tensor& a, const Tensor& b) {
    //     //will change this once broadcasting is implemented, for now it will throw and error if shape doesnt match
    //     if (a.shape() != b.shape())
    //         throw std::runtime_error("Shape mismatch in mul");
    //
    //     Tensor out(a.data() * b.data(), a.requires_grad() || b.requires_grad());
    //
    //     if (out.requires_grad() && out.impl_->grad_fn() == nullptr) {
    //         auto fn = std::make_shared<MulFunction>();
    //         fn->inputs = { a.impl_, b.impl_ };
    //         out.impl_->grad_fn() = fn;   // PIMPL: grad_fn lives in impl_
    //     }
    //
    //     return out;
    // }
    //
    // Tensor operator*(const Tensor& lhs, float scalar) {
    //     return lhs * Tensor::full(lhs.shape(), scalar, false);
    // }
    //
    // Tensor operator*(float scalar, const Tensor& rhs) {
    //     return rhs * Tensor::full(rhs.shape(), scalar, false);
    // }
}
