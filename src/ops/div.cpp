// #include "ops/div.hpp"
// #include "autograd/function.hpp"
// #include "tensor/tensor.hpp"
//
// #include <stdexcept>
//
// namespace cppgrad {
//
//     Tensor operator/(const Tensor& a, const Tensor& b) {
//         // Broadcasting not yet supported
//         if (a.shape() != b.shape())
//             throw std::runtime_error("Shape mismatch in div");
//
//         Tensor out(a.data() / b.data(),
//                    a.requires_grad() || b.requires_grad());
//
//         if (out.requires_grad() && out.impl_->grad_fn() == nullptr) {
//             auto fn = std::make_shared<DivFunction>();
//             fn->inputs = { a.impl_, b.impl_ };
//             out.impl_->grad_fn() = fn;
//         }
//
//         return out;
//     }
//
//     Tensor operator/(const Tensor& lhs, float scalar) {
//         return lhs / Tensor::full(lhs.shape(), scalar, false);
//     }
//
//     Tensor operator/(float scalar, const Tensor& rhs) {
//         return Tensor::full(rhs.shape(), scalar, false) / rhs;
//     }
//
// }