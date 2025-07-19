#include "../../include/tensor/tensor.hpp"
#include <stdexcept>
#include <utility>

#include "autograd/function.h"

namespace cppgrad {

    //if tensor has more than 4 dimensions, ignore the rest for now
    af::dim4 to_dim4(const std::vector<size_t>& shape) {
        af::dim4 dims(1, 1, 1, 1);
        for (size_t i = 0; i < std::min(static_cast<size_t>(4), shape.size()); ++i)
            dims[i] = shape[i];
        return dims;
    }

    Tensor::Tensor(std::shared_ptr<TensorImpl> impl)
        : impl_(std::move(impl)){}

    Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<float>& values, bool requires_grad) {

        af::dim4 dims = to_dim4(shape);

        size_t expected_numel = dims.elements();  // total elements
        if (values.size() != expected_numel) {
            throw std::invalid_argument("Number of values does not match shape");
        }

        af::array arr(static_cast<unsigned>(values.size()), values.data());
        arr = af::moddims(arr, dims);

        impl_ = std::make_shared<TensorImpl>(arr, requires_grad);
    }

    Tensor::Tensor(const af::array& arr, bool requires_grad)
      : impl_(std::make_shared<TensorImpl>(arr, requires_grad))
    { }

    Tensor Tensor::zeros(const std::vector<size_t>& shape, bool requires_grad) {
        af::dim4 dims = to_dim4(shape);
        af::array arr = af::constant(0.0f, dims);
        return {arr, requires_grad};
    }

    Tensor Tensor::ones(const std::vector<size_t>& shape, bool requires_grad) {
        af::dim4 dims = to_dim4(shape);
        af::array arr = af::constant(1.0f, dims);
        return {arr, requires_grad};
    }

    Tensor Tensor::randn(const std::vector<size_t>& shape, bool requires_grad) {
        af::dim4 dims = to_dim4(shape);
        return {af::randn(dims), requires_grad};
    }

    Tensor Tensor::full(const std::vector<size_t>& shape, float value, bool requires_grad) {
        af::dim4 dims = to_dim4(shape);
        af::array arr = af::constant(value, dims);
        return {arr, requires_grad};
    }

    std::vector<size_t> Tensor::shape() const {
        af::dim4 dims = impl_->data_.dims();
        std::vector<size_t> out;
        for (int i = 0; i < 4 && dims[i] > 1; ++i)
            out.push_back(dims[i]);
        return out;
    }

    size_t Tensor::numel() const {
        return impl_->data_.elements();
    }

    size_t Tensor::ndim() const {
        return impl_->data_.numdims();
    }

    void Tensor::print() const {
        af_print(impl_->data_);
    }

    void Tensor::print_grad() const {
        af_print(impl_->grad_);
    }

    af::array Tensor::data() const{
        return impl_->data_;
    }

    af::array Tensor::grad() const{
        return impl_->grad_;
    }

    bool Tensor::requires_grad() const { return impl_->requires_grad_; }

     // void Tensor::backward(const af::array &grad_output) {
     //    if (!requires_grad()) return;
     //
     //    af::array grad = grad_output.isempty()
     //      ? af::constant(1.0f, impl_->data_.dims(), impl_->data_.type())
     //      : grad_output;
     //
     //    // accumulate into the impl’s grad_
     //    if (impl_->grad_.isempty()) impl_->grad_ = grad;
     //    else impl_->grad_ += grad;
     //
     //    // recurse via the stored Function node
     //    if (impl_->grad_fn) {
     //        impl_->grad_fn->apply(grad);
     //    }
     // }

    void Tensor::backward(const af::array &grad_output) {
        if (!requires_grad()) return;
        // seed the root gradient
        impl_->grad_ = af::constant(1, impl_->data_.dims());
        // then kick off the chain
        if (impl_->grad_fn)
            impl_->grad_fn->apply(impl_->grad_);
    }


} // namespace cppgrad