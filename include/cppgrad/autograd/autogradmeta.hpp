#pragma once

#include <arrayfire.h>
#include <memory>

namespace cppgrad{

class Function;

    /**
     * @brief Holds autograd-related metadata for a tensor.
     *
     * This class is attached to a TensorImpl when autograd is enabled.
     * It stores:
     * - the accumulated gradient (`grad`),
     * - the backward function (`grad_fn`) that created the tensor,
     * - whether the tensor requires gradients (`requires_grad`),
     * - and a flag to track if `.backward()` has already been called on it.
     *
     * Similar to PyTorch's `AutogradMeta`, it enables construction and traversal
     * of the dynamic computation graph during backward passes.
     */
    class AutogradMeta {
        public:
            AutogradMeta(bool req, const af::array &data);

            af::array grad;
            std::shared_ptr<Function> grad_fn;
            bool requires_grad;
            bool has_called_backward = false;
    };

}