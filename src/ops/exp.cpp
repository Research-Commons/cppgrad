// #include "ops/exp.hpp"
// #include "autograd/function.hpp"
// #include "tensor/tensor.hpp"
//
// namespace cppgrad {
//
//     Tensor exp(const Tensor& a) {
//         Tensor out(af::exp(a.data()), a.requires_grad());
//
//         if (out.requires_grad() && out.impl_->grad_fn() == nullptr) {
//             auto fn = std::make_shared<ExpFunction>();
//             fn->inputs = { a.impl_ };
//             out.impl_->grad_fn() = fn;
//         }
//
//         return out;
//     }
//
// }