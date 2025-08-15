#pragma once
#include <vector>
#include <memory>
#include <stdexcept>

#include "cppgrad/enums/dispatcherEnum.h"

namespace cppgrad {

    class Function; // forward

    /**
     * TensorImpl
     *
     * Internal storage for Tensor. Replaces the old ArrayFire af::array storage
     * with std::vector<float>. Holds optional autograd metadata (grad vector,
     * grad_fn pointer, bookkeeping).
     */
    class TensorImpl {
    public:
        // Construct from shape and initial data (row-major)
        // device defaults to DeviceType::CPU for backward compatibility
        TensorImpl(const std::vector<size_t>& shape,
                   const std::vector<float>& data,
                   bool requires_grad,
                   DeviceType device = DeviceType::CPU);

        // Construct from shape and a single fill value
        TensorImpl(const std::vector<size_t>& shape,
                   float fill_value,
                   bool requires_grad,
                   DeviceType device = DeviceType::CPU);

        // Accessors
        const std::vector<float>& data() const;
        std::vector<float>& data();

        bool requires_grad() const;
        bool has_autograd() const;

        // Gradient access (will create grad storage lazily)
        std::vector<float>& grad();
        const std::vector<float>& grad() const;

        const std::vector<float>& stride() const;
        std::vector<float>& stride();

        std::shared_ptr<Function>& grad_fn();
        const std::shared_ptr<Function>& grad_fn() const;

        bool has_called_backward() const;
        void set_has_called_backward(bool val);

        const std::vector<size_t>& shape() const;
        size_t numel() const;

        // Device accessor / mutator
        DeviceType device() const;
        void set_device(DeviceType dev);

    private:
        std::vector<float> data_;
        std::vector<float> grad_; // empty until needed
        std::vector<float> stride_; // empty until needed
        std::shared_ptr<Function> grad_fn_;
        bool requires_grad_ = false;
        bool has_called_backward_ = false;
        std::vector<size_t> shape_;
        DeviceType device_ = DeviceType::CPU;

        std::vector<size_t> compute_strides(const std::vector<size_t>& shape);
    };

} // namespace cppgrad