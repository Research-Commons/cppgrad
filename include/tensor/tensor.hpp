#pragma once
#include <vector>
#include <arrayfire.h>
#include <memory>
#include "tensor/tensorimpl.hpp"

namespace cppgrad {

    class Function;

    class Tensor {
    public:
        // Constructors
        Tensor(const std::vector<size_t>& shape, const std::vector<float>& values, bool requires_grad = false);
        // Factory methods
        static Tensor zeros(const std::vector<size_t>& shape, bool requires_grad = false);
        static Tensor ones(const std::vector<size_t>& shape, bool requires_grad = false);
        static Tensor randn(const std::vector<size_t>& shape, bool requires_grad = false);
        static Tensor full(const std::vector<size_t>& shape, float value, bool requires_grad = false);
        static Tensor from_array_column_major(const std::vector<size_t>& shape,
                                      const std::vector<float>& values,
                                      bool requires_grad = false);

        // Shape and info
        std::vector<size_t> shape() const;
        size_t numel() const;
        size_t ndim() const;
        bool requires_grad() const;
        void zero_grad() const;
        void print() const;
        void print_pretty() const;
        void print_grad() const;

        //additional ops
        /// Sum over one axis, optionally keeping that dimension.
        /// - dim: which axis to reduce (0-based)
        /// - keepdim: if true, output has same rank with size-1 on reduced dim
        Tensor sum(int dim = -1, bool keepdim = false) const;
        Tensor mean(int dim = -1, bool keepdim = false) const;
        Tensor max(int dim = -1, bool keepdim = false) const;
        //Autograd
        void backward(const af::array& grad_output = af::array());

        af::array data() const;
        af::array grad() const;

    private:
        std::shared_ptr<TensorImpl> impl_;

    public:
        std::shared_ptr<TensorImpl> impl() const;

    private:
        // Private constructor that takes ownership of an Impl
        Tensor(std::shared_ptr<TensorImpl> impl);
        Tensor(const af::array& arr, bool requires_grad = true);

        //Tensor ops implemented in another file for better readability
        friend Tensor operator+(const Tensor&, const Tensor&);
        friend Tensor operator-(const Tensor&, const Tensor&);
        friend Tensor operator*(const Tensor&, const Tensor&);
        friend Tensor operator+(const Tensor& lhs, float scalar);
        friend Tensor operator+(float scalar, const Tensor& rhs);
        friend Tensor operator*(const Tensor& lhs, float scalar);
        friend Tensor operator*(float scalar, const Tensor& rhs);
        friend Tensor operator-(const Tensor& a, const Tensor& b);
        friend Tensor operator-(const Tensor& lhs, float scalar);
        friend Tensor operator-(float scalar, const Tensor& rhs);
        friend Tensor operator/(const Tensor& a, const Tensor& b);
        friend Tensor operator/(const Tensor& lhs, float scalar);
        friend Tensor operator/(float scalar, const Tensor& rhs);

        friend Tensor operator-(const Tensor& a);
        friend Tensor exp(const Tensor& a);
        friend Tensor log(const Tensor& a);
        friend Tensor pow(const Tensor& base, const Tensor& exponent);
        friend Tensor pow(const Tensor& base, float scalar);
        friend Tensor pow(float scalar, const Tensor& exponent);

        friend class TensorUtils;
    };

} // namespace cppgrad