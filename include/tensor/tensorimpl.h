#pragma once
#include <arrayfire.h>
#include <memory>

namespace cppgrad {

    class Function;

    struct TensorImpl {
        af::array data_, grad_;
        bool requires_grad_ = false;
        std::shared_ptr<Function> grad_fn;

        TensorImpl(const af::array &d, bool req)
          : data_(d), requires_grad_(req) {
            if (requires_grad_) grad_ = af::constant(0, data_.dims(), data_.type());
        }

    };
}