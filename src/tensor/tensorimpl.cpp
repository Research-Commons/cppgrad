#include "tensor/tensorimpl.hpp"

namespace cppgrad {
    TensorImpl::TensorImpl(const af::array &d, bool requires_grad)
    : data_(d) {
        if (requires_grad) {
            autograd_ = std::make_unique<AutogradMeta>(true, data_);
        }
    }

    const af::array& TensorImpl::data() const {
        return data_;
    }

    af::array& TensorImpl::data() {
        return data_;
    }

    bool TensorImpl::requires_grad() const {
        return autograd_ && autograd_->requires_grad;
    }

    bool TensorImpl::has_autograd() const {
        return autograd_ != nullptr;
    }

    af::array& TensorImpl::grad() {
        return autograd_->grad;
    }

    const af::array& TensorImpl::grad() const {
        return autograd_->grad;
    }

    std::shared_ptr<Function>& TensorImpl::grad_fn() {
        return autograd_->grad_fn;
    }

    const std::shared_ptr<Function>& TensorImpl::grad_fn() const {
        return autograd_->grad_fn;
    }

    bool TensorImpl::has_called_backward() const {
        return autograd_->has_called_backward;
    }

    void TensorImpl::set_has_called_backward(bool has_called_backwards) {
        this->autograd_->has_called_backward = has_called_backwards;
    }
}



