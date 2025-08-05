#pragma once

#include <vector>
#include <memory>
#include <arrayfire.h>

#include "tensorimpl.hpp"

namespace cppgrad {

    /**
     * @file tensor.hpp
     * @brief Public-facing Tensor class for cppgrad.
     *
     * This class defines the main user interface for tensor operations in cppgrad.
     * It wraps a `TensorImpl` object and exposes high-level creation, manipulation,
     * and autograd capabilities, similar to PyTorch's `torch.Tensor`.
     *
     * Key Features:
     * - Tensor creation: `zeros`, `ones`, `randn`, `full`, `from_array_column_major`
     * - Data inspection: shape, numel, ndim, gradient info, print utilities
     * - Operator overloading for elementwise math (+, -, *, /) and broadcasting
     * - Autograd support: attaches backward functions and triggers `.backward()`
     * - Reduction operations: `sum`, `mean`, `max`
     *
     * Design:
     * - Wraps a `std::shared_ptr<TensorImpl>` to allow internal tensor reuse.
     * - Construction from raw `af::array` supported internally for efficient interop.
     * - Friend functions used for operator overloads and mathematical operations.
     * - Actual autograd logic resides in `Function` subclasses attached via `TensorImpl`.
    */

    class Tensor {
    public:
        // -------- Constructors --------
        Tensor(const std::vector<size_t>& shape, const std::vector<float>& values, bool requires_grad = false);

        // -------- Factory Methods --------
        static Tensor zeros(const std::vector<size_t>& shape, bool requires_grad = false);
        static Tensor ones(const std::vector<size_t>& shape, bool requires_grad = false);
        static Tensor randn(const std::vector<size_t>& shape, bool requires_grad = false);
        static Tensor full(const std::vector<size_t>& shape, float value, bool requires_grad = false);
        static Tensor from_array_column_major(const std::vector<size_t>& shape,
                                              const std::vector<float>& values,
                                              bool requires_grad = false);

        // -------- Shape and Info --------
        std::vector<size_t> shape() const;
        size_t numel() const;
        size_t ndim() const;
        bool requires_grad() const;

        void zero_grad() const;
        void print() const;
        void print_pretty() const;
        void print_grad() const;

        // -------- Autograd --------
        void backward(const af::array& grad_output = af::array());
        af::array grad() const;

        // -------- Data Access --------
        af::array data() const;
        std::shared_ptr<TensorImpl> impl() const;

        // -------- Reduction Ops --------
        /// Sum over one axis (or all), optionally keeping reduced dim.
        Tensor sum(int dim = -1, bool keepdim = false) const;
        Tensor mean(int dim = -1, bool keepdim = false) const;
        Tensor max(int dim = -1, bool keepdim = false) const;

    private:
        std::shared_ptr<TensorImpl> impl_;

        // -------- Internal Constructors --------
        Tensor(std::shared_ptr<TensorImpl> impl);
        Tensor(const af::array& arr, bool requires_grad = true);

        static af::dim4 to_dim4(const std::vector<size_t>& shape);

        // -------- Operator Overloads --------
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

        friend Tensor operator-(const Tensor&);  // Unary minus

        // -------- Unary/Binary Functions --------
        friend Tensor exp(const Tensor&);
        friend Tensor log(const Tensor&);
        friend Tensor pow(const Tensor& base, const Tensor& exponent);
        friend Tensor pow(const Tensor& base, float scalar);
        friend Tensor pow(float scalar, const Tensor& exponent);

        // -------- Tensor Utilities --------
        friend class TensorUtils;
    };

} // namespace cppgrad
