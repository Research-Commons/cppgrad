#pragma once
#include <vector>
#include <arrayfire.h>
#include <memory>
#include "tensor/tensorimpl.h"

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

        // Shape and info
        std::vector<size_t> shape() const;
        size_t numel() const;
        size_t ndim() const;
        bool requires_grad() const;
        void print() const;
        void print_grad() const;

        //Autograd
        void backward(const af::array& grad_output = af::array());

        af::array data() const;
        af::array grad() const;

    private:
        std::shared_ptr<TensorImpl> impl_;

        // Private constructor that takes ownership of an Impl
        Tensor(std::shared_ptr<TensorImpl> impl);

        Tensor(const af::array& arr, bool requires_grad = true);

        friend Tensor operator+(const Tensor&, const Tensor&);
        friend Tensor operator-(const Tensor&, const Tensor&);
        friend Tensor operator*(const Tensor&, const Tensor&);
    };

} // namespace cppgrad