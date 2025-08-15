#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <sstream>

#include "cppgrad/tensor/tensorimpl.hpp"
#include "cppgrad/enums/dispatcherEnum.h"   // for DeviceType

namespace cppgrad {

/**
 * Public-facing Tensor class.
 * API modeled after your previous header, but using std::vector<float>
 * instead of ArrayFire's af::array.
 */
class Tensor {
public:
    // -------- Constructors --------
    // Construct from shape + values (row-major)
    Tensor(const std::vector<size_t>& shape,
           const std::vector<float>& values,
           bool requires_grad = false,
           DeviceType device = DeviceType::CPU);

    // Construct scalar or filled tensor
    Tensor(const std::vector<size_t>& shape,
           float value,
           bool requires_grad = false,
           DeviceType device = DeviceType::CPU);

    // Internal constructor (from impl)
    explicit Tensor(std::shared_ptr<TensorImpl> impl);

    Tensor() = default;
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    // -------- Factory Methods --------
    static Tensor zeros(const std::vector<size_t>& shape,
                        bool requires_grad = false,
                        DeviceType device = DeviceType::CPU);
    static Tensor ones(const std::vector<size_t>& shape,
                       bool requires_grad = false,
                       DeviceType device = DeviceType::CPU);
    static Tensor randn(const std::vector<size_t>& shape,
                        bool requires_grad = false,
                        DeviceType device = DeviceType::CPU);
    static Tensor full(const std::vector<size_t>& shape,
                       float value,
                       bool requires_grad = false,
                       DeviceType device = DeviceType::CPU);

    // -------- Shape and Info --------
    std::vector<size_t> shape() const;
    size_t numel() const;
    size_t ndim() const;
    bool requires_grad() const;
    DeviceType device_type() const;

    void zero_grad();
    void print() const;
    void print_pretty() const;
    void print_grad() const;

    // -------- Autograd  --------
    void backward(const std::vector<float>& grad_output = std::vector<float>());

    const std::vector<float>& grad() const;

    // -------- Data Access --------
    const std::vector<float>& data() const;
    std::vector<float>& data();

    const std::vector<float>& stride() const;
    std::vector<float>& stride();

    std::shared_ptr<TensorImpl> impl() const;

    // -------- Reduction Ops --------
    // Tensor sum(int dim = -1, bool keepdim = false) const;
    // Tensor mean(int dim = -1, bool keepdim = false) const;
    // Tensor max(int dim = -1, bool keepdim = false) const;

    // Operator overloads (elementwise). Implemented in tensor.cpp
    friend Tensor operator+(const Tensor&, const Tensor&);
    friend Tensor operator-(const Tensor&, const Tensor&);
    friend Tensor operator*(const Tensor&, const Tensor&);
    friend Tensor operator/(const Tensor&, const Tensor&);

    friend Tensor operator+(const Tensor&, float);
    friend Tensor operator+(float, const Tensor&);
    friend Tensor operator-(const Tensor&, float);
    friend Tensor operator-(float, const Tensor&);
    friend Tensor operator*(const Tensor&, float);
    friend Tensor operator*(float, const Tensor&);
    friend Tensor operator/(const Tensor&, float);
    friend Tensor operator/(float, const Tensor&);

    friend Tensor operator-(const Tensor&);  // unary minus

private:
    std::shared_ptr<TensorImpl> impl_;
};

} // namespace cppgrad