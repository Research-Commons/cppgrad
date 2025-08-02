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

    Tensor TensorUtils::matmul(const Tensor &a, const Tensor &b) {
        const af::array& a_data = a.data();
        const af::array& b_data = b.data();

        af::array result_data = af::matmul(a_data, b_data);  // M×N result

        auto result_impl = std::make_shared<TensorImpl>(result_data, /*requires_grad=*/a.requires_grad() || b.requires_grad());
        Tensor result(result_impl);

        // If autograd is enabled, create a backward Function
        if (result.requires_grad()) {
            auto fn = std::make_shared<MatMulFunction>();
            fn->inputs = { a.impl_, b.impl_ };
            result_impl->grad_fn() = fn;
        }

        return result;
    }

    Tensor TensorUtils::transpose(const Tensor &t) {
        af::array t_data = af::transpose(t.data());
        auto new_impl = std::make_shared<TensorImpl>(t_data, t.requires_grad());
        return {new_impl};
    }
}
