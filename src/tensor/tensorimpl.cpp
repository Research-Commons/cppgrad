#include "cppgrad/tensor/tensorimpl.hpp"
#include <numeric>
#include <stdexcept>

namespace cppgrad {

    TensorImpl::TensorImpl(const std::vector<size_t>& shape,
                           const std::vector<float>& data,
                           bool requires_grad,
                           DeviceType device)
        : data_(data),
          requires_grad_(requires_grad),
          has_called_backward_(false),
          shape_(shape),
          device_(device)
    {
        size_t total = 1;
        for (auto s : shape_) total *= s;
        if (data_.size() != total) {
            throw std::runtime_error("TensorImpl: data size does not match shape");
        }
        compute_strides(shape);
    }

    TensorImpl::TensorImpl(const std::vector<size_t>& shape,
                           float fill_value,
                           bool requires_grad,
                           DeviceType device)
        : data_(),
          requires_grad_(requires_grad),
          has_called_backward_(false),
          shape_(shape),
          device_(device)
    {
        size_t total = 1;
        for (auto s : shape_) total *= s;
        data_.assign(total, fill_value);
        compute_strides(shape);
    }

    const std::vector<float>& TensorImpl::data() const { return data_; }
    std::vector<float>& TensorImpl::data() { return data_; }

    bool TensorImpl::requires_grad() const { return requires_grad_; }
    bool TensorImpl::has_autograd() const { return (bool)grad_fn_; }

    std::vector<float>& TensorImpl::grad() {
        if (grad_.empty()) grad_.assign(numel(), 0.0f);
        return grad_;
    }
    const std::vector<float>& TensorImpl::grad() const {
        return grad_;
    }

    std::vector<float>& TensorImpl::stride(){ return stride_; }
    const std::vector<float>& TensorImpl::stride() const { return stride_; }

    std::shared_ptr<Function>& TensorImpl::grad_fn() { return grad_fn_; }
    const std::shared_ptr<Function>& TensorImpl::grad_fn() const { return grad_fn_; }

    bool TensorImpl::has_called_backward() const { return has_called_backward_; }
    void TensorImpl::set_has_called_backward(bool val) { has_called_backward_ = val; }

    const std::vector<size_t>& TensorImpl::shape() const { return shape_; }
    size_t TensorImpl::numel() const {
        size_t total = 1;
        for (auto s : shape_) total *= s;
        return total;
    }

    DeviceType TensorImpl::device() const { return device_; }
    void TensorImpl::set_device(DeviceType dev) { device_ = dev; }

    std::vector<size_t> TensorImpl::compute_strides(const std::vector<size_t>& shape){
        int n = static_cast<int>(shape.size());
        std::vector<size_t> stride((size_t)n, 0);
        if (n == 0) return stride;
        stride[(size_t)n - 1] = 1;
        for (int i = n - 2; i >= 0; --i) {
            stride[(size_t)i] = stride[(size_t)i + 1] * shape[(size_t)i + 1];
        }
        return stride;
    }

} // namespace cppgrad