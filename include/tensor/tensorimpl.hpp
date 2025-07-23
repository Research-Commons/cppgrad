#pragma once
#include <arrayfire.h>
#include <memory>

#include "autograd/autogradmeta.hpp"

namespace cppgrad {

    class TensorImpl {
        public:
            TensorImpl(const af::array &d, bool requires_grad);

            // Getters
            const af::array& data() const;
            af::array& data();

            bool requires_grad() const;
            bool has_autograd() const;

            af::array& grad();
            const af::array& grad() const;

            std::shared_ptr<Function>& grad_fn();
            const std::shared_ptr<Function>& grad_fn() const;

            bool has_called_backward() const;
            void set_has_called_backward(bool has_called_backwards);
        private:
            af::array data_;
            std::unique_ptr<AutogradMeta> autograd_;
    };

}