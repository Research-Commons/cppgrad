// #include "autograd/autogradmeta.hpp"
//
// namespace cppgrad {
//
//     AutogradMeta::AutogradMeta(bool req, const af::array &data)
//     : requires_grad(req) {
//         if (requires_grad) {
//             grad = af::constant(0, data.dims(), data.type());
//         }
//         has_called_backward = false;
//     }
// }
//
