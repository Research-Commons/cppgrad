#include "tensor/tensorutils.hpp"
#include "autograd/function.hpp"

namespace cppgrad {

    Tensor TensorUtils::clone(const Tensor& input) {
        af::array cloned_data = input.data().copy();

        auto new_impl = std::make_shared<TensorImpl>(cloned_data, false);
        return Tensor(new_impl);
    }

    Tensor TensorUtils::clone_with_grad(const Tensor& input) {
        af::array cloned_data = input.data().copy();
        bool req_grad = input.requires_grad();

        auto new_impl = std::make_shared<TensorImpl>(cloned_data, req_grad);
        Tensor out(new_impl);

        if (req_grad) {
            auto fn = std::make_shared<CloneFunction>();
            fn->inputs = { input.impl_ };
            out.impl_->grad_fn() = fn;
        }

        return out;
    }

}