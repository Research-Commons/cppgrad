// #include "ops/neg.hpp"
// #include "autograd/function.hpp"
// #include "tensor/tensor.hpp"
//
//
// namespace cppgrad {
//
//     Tensor operator-(const Tensor& a) {
//         Tensor out(-a.data(), a.requires_grad());
//
//         if (out.requires_grad() && out.impl_->grad_fn() == nullptr) {
//             auto fn = std::make_shared<NegFunction>();
//             fn->inputs = { a.impl_ };
//             out.impl_->grad_fn() = fn;
//         }
//
//         return out;
//     }
//
// }