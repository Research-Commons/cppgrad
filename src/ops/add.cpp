#include "ops/add.hpp"
#include "autograd/function.hpp"
#include <stdexcept>

namespace cppgrad {

    Tensor operator+(const Tensor& a, const Tensor& b) {
        //will change this once broadcasting is implemented, for now it will throw and error if shape doesnt match
        if (a.shape() != b.shape())
            throw std::runtime_error("shape mismatch");

        Tensor out(a.data() + b.data(),
                   a.requires_grad() || b.requires_grad());

        if (out.requires_grad() && out.impl_->grad_fn() == nullptr) {
            auto fn = std::make_shared<AddFunction>();
            fn->inputs = { a.impl_, b.impl_ };
            out.impl_->grad_fn() = fn;   // PIMPL: grad_fn lives in impl_
        }

        return out;
    }

    Tensor operator+(const Tensor& lhs, float scalar) {
        return lhs + Tensor::full(lhs.shape(), scalar, false);
    }
    Tensor operator+(float scalar, const Tensor& rhs) {
        return rhs + Tensor::full(rhs.shape(), scalar, false);
    }
}
