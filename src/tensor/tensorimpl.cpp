// #include "tensor/tensorimpl.hpp"
//
//
// namespace cppgrad {
//
//     // Constructor: wraps an ArrayFire array and optionally enables autograd.
//     // If `requires_grad` is true, initializes AutogradMeta to track gradient info.
//     TensorImpl::TensorImpl(const af::array &d, bool requires_grad)
//     : data_(d) {
//         if (requires_grad) {
//             autograd_ = std::make_unique<AutogradMeta>(true, data_);
//         }
//     }
//
//     // Const accessor for the underlying data array.
//     const af::array& TensorImpl::data() const {
//         return data_;
//     }
//
//     // Mutable accessor for the underlying data array.
//     af::array& TensorImpl::data() {
//         return data_;
//     }
//
//     // Checks if autograd is enabled for this tensor.
//     // True only if autograd_ is initialized and requires_grad is set.
//     bool TensorImpl::requires_grad() const {
//         return autograd_ && autograd_->requires_grad;
//     }
//
//     // Returns true if this tensor has any autograd tracking (regardless of requires_grad).
//     bool TensorImpl::has_autograd() const {
//         return autograd_ != nullptr;
//     }
//
//     // Mutable accessor to this tensor’s gradient.
//     // Only valid if autograd_ is initialized.
//     af::array& TensorImpl::grad() {
//         return autograd_->grad;
//     }
//
//     // Const accessor to this tensor’s gradient.
//     const af::array& TensorImpl::grad() const {
//         return autograd_->grad;
//     }
//
//     // Mutable accessor to the backward function responsible for computing this tensor's grad.
//     std::shared_ptr<Function>& TensorImpl::grad_fn() {
//         return autograd_->grad_fn;
//     }
//
//     // Const accessor to the backward function.
//     const std::shared_ptr<Function>& TensorImpl::grad_fn() const {
//         return autograd_->grad_fn;
//     }
//
//     // Check if `.backward()` has already been called on this tensor.
//     bool TensorImpl::has_called_backward() const {
//         return autograd_->has_called_backward;
//     }
//
//     // Set the flag indicating whether `.backward()` has been called.
//     void TensorImpl::set_has_called_backward(bool has_called_backwards) {
//         this->autograd_->has_called_backward = has_called_backwards;
//     }
//
// } // namespace cppgrad
//
//
//
