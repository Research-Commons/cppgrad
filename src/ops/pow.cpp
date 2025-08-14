// #include "ops/pow.hpp"
// #include "autograd/function.hpp"
// #include "tensor/tensor.hpp"
//
// namespace cppgrad {
//
//     Tensor pow(const Tensor& base, const Tensor& exponent) {
//         if (base.shape() != exponent.shape())
//             throw std::runtime_error("Shape mismatch in pow");
//
//         Tensor out(af::pow(base.data(), exponent.data()),
//                    base.requires_grad() || exponent.requires_grad());
//
//         if (out.requires_grad() && out.impl_->grad_fn() == nullptr) {
//             auto fn = std::make_shared<PowFunction>();
//             fn->inputs = { base.impl_, exponent.impl_ };
//             out.impl_->grad_fn() = fn;
//         }
//
//         return out;
//     }
//
//     // scalar overloads
//     Tensor pow(const Tensor& base, float scalar) {
//         return pow(base, Tensor::full(base.shape(), scalar, false));
//     }
//
//     Tensor pow(float scalar, const Tensor& exponent) {
//         return pow(Tensor::full(exponent.shape(), scalar, false), exponent);
//     }
//
// }