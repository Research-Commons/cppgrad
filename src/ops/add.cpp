#include "ops/add.hpp"
#include "autograd/function.hpp"
#include "tensor/tensor.hpp"

#include <stdexcept>

#include "dispatcher/kernelRegistry.h"
#include "ops/helperOps.hpp"


namespace cppgrad {

    Tensor operator+(const Tensor& a, const Tensor& b) {
        if (a.device_type() != b.device_type()) {
            throw std::runtime_error("Device mismatch in add");
        }
        // Compute broadcasted shape and create output tensor
        std::vector<size_t> out_shape = computeBroadcastShape(a.shape(), b.shape());
        Tensor out(out_shape, 0.0f, a.requires_grad() || b.requires_grad(), a.device_type());

        if (out.requires_grad()) {
            auto f = std::make_shared<AddFunction>();
            f->inputs = { a.impl(), b.impl() };
            out.impl()->grad_fn() = f;
        }

        // Lookup and call the registered kernel
        KernelRegistry::instance()
            .getKernel(OpType::Add, a.device_type())(a, b, out);
        return out;
    }

    // Tensor operator+(const Tensor& a, const Tensor& b) {
    //     //will change this once broadcasting is implemented, for now it will throw and error if shape doesnt match
    //     if (a.shape() != b.shape())
    //         throw std::runtime_error("shape mismatch");
    //
    //     Tensor out(a.data() + b.data(),
    //                a.requires_grad() || b.requires_grad());
    //
    //     if (out.requires_grad() && out.impl_->grad_fn() == nullptr) {
    //         auto fn = std::make_shared<AddFunction>();
    //         fn->inputs = { a.impl_, b.impl_ };
    //         out.impl_->grad_fn() = fn;   // PIMPL: grad_fn lives in impl_
    //     }
    //
    //     return out;
    // }

    Tensor operator+(const Tensor& lhs, float scalar) {
        return lhs + Tensor::full(lhs.shape(), scalar, false, lhs.device_type());
    }
    Tensor operator+(float scalar, const Tensor& rhs) {
        return rhs + Tensor::full(rhs.shape(), scalar, false, rhs.device_type());
    }
}
