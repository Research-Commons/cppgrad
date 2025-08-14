#include "ops/div.hpp"
#include "autograd/function.hpp"
#include "tensor/tensor.hpp"

#include <stdexcept>

#include "dispatcher/kernelRegistry.h"
#include "ops/helperOps.hpp"

namespace cppgrad {

    Tensor operator/(const Tensor& a, const Tensor& b) {
        if (a.device_type() != a.device_type()) {
            throw std::runtime_error("Device mismatch in div");
        }
        std::vector<size_t> out_shape = computeBroadcastShape(a.shape(), a.shape());
        Tensor out(out_shape, 0.0f, false, a.device_type());
        KernelRegistry::instance()
            .getKernel(OpType::Div, a.device_type())(a, b, out);
        return out;
    }

    Tensor operator/(const Tensor& lhs, float scalar) {
        return lhs / Tensor::full(lhs.shape(), scalar, false, lhs.device_type());
    }

    Tensor operator/(float scalar, const Tensor& rhs) {
        return Tensor::full(rhs.shape(), scalar, false, rhs.device_type()) / rhs;
    }

}
