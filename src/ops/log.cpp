// #include "ops/log.hpp"
// #include "autograd/function.hpp"
// #include "tensor/tensor.hpp"
//
// namespace cppgrad {
//
//     Tensor log(const Tensor& a) {
//         Tensor out(af::log(a.data()), a.requires_grad());
//
//         if (out.requires_grad() && out.impl_->grad_fn() == nullptr) {
//             auto fn = std::make_shared<LogFunction>();
//             fn->inputs = { a.impl_ };
//             out.impl_->grad_fn() = fn;
//         }
//
//         return out;
//     }
//
// }