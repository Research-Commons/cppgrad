#pragma once
#include <vector>
#include <arrayfire.h>

namespace cppgrad {

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
        bool requires_grad() const { return requires_grad_; }
        void print() const;

        //Variables
        bool requires_grad_ = false;

    private:
        af::array data_;

        Tensor(const af::array& arr, bool requires_grad = true);

        friend Tensor operator+(const Tensor&, const Tensor&);
        friend Tensor operator-(const Tensor&, const Tensor&);
        friend Tensor operator*(const Tensor&, const Tensor&);
    };

} // namespace cppgrad