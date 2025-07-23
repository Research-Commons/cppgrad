#pragma once

#include <arrayfire.h>
#include <memory>

namespace cppgrad{

class Function;

    class AutogradMeta {
        public:
            AutogradMeta(bool req, const af::array &data);

            af::array grad;
            std::shared_ptr<Function> grad_fn;
            bool requires_grad;
            bool has_called_backward = false;
    };

}