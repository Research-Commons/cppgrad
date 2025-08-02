#include "tensor/tensor.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "autograd/function.hpp"

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


        // 2) Load raw linear data
        af::array arr(static_cast<unsigned>(values.size()), values.data());

        // 3) Reverse the shape vector (e.g. {D0,D1,D2} -> {D2,D1,D0})
        std::vector<size_t> rev_shape(shape.rbegin(), shape.rend());
        af::dim4 rev_dims = to_dim4(rev_shape);

        // 4) Reshape into reversed dims (column‑major fill aligns with row‑major input)
        arr = af::moddims(arr, rev_dims);

        // 5) Build a reorder‑axes list: {N‑1, N‑2, …, 0}
        std::vector<unsigned> axes(shape.size());
        std::iota(axes.begin(), axes.end(), 0);
        std::reverse(axes.begin(), axes.end());

        // 6) Reorder back to original axis order (choose the right overload)
        switch (axes.size()) {
            case 1:
                // 1D: nothing to do (reversing a single axis is a no‑op)
                break;
            case 2:
                arr = af::reorder(arr, axes[0], axes[1]);
                break;
            case 3:
                arr = af::reorder(arr, axes[2], axes[1], axes[0]);
                break;
            case 4:
                arr = af::reorder(arr, axes[3], axes[2], axes[1], axes[0]);
                break;
            default:
                throw std::runtime_error("Tensor constructor only supports up to 4D");
        }

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

    Tensor Tensor::from_array_column_major(const std::vector<size_t> &shape, const std::vector<float> &values, bool requires_grad) {
        size_t expected = 1;
        for (auto s : shape) expected *= s;

        if (values.size() != expected) {
            throw std::invalid_argument("Value count doesn't match shape");
        }

        af::dim4 dims = to_dim4(shape);
        af::array arr(static_cast<unsigned>(values.size()), values.data());
        arr = af::moddims(arr, dims);

        return {std::make_shared<TensorImpl>(arr, requires_grad)};
    }


    std::vector<size_t> Tensor::shape() const {
        af::dim4 dims = impl_->data().dims();
        std::vector<size_t> out;
        for (int i = 0; i < 4 && dims[i] > 1; ++i)
            out.push_back(dims[i]);
        return out;
    }

    size_t Tensor::numel() const {
        return impl_->data().elements();
    }

    size_t Tensor::ndim() const {
        return impl_->data().numdims();
    }

    void Tensor::print() const {
        af_print(impl_->data());
    }

    void Tensor::print_pretty() const {
        af::array data = impl_->data();
        std::vector<float> host(data.elements());
        data.host(host.data());

        std::cout << "Tensor(shape=[";
        af::dim4 dims = data.dims();
        for (int i = 0; i < data.numdims(); ++i) {
            std::cout << dims[i];
            if (i < data.numdims() - 1)
                std::cout << ", ";
        }
        std::cout << "], values=";

        if (data.elements() == 1) {
            std::cout << host[0];
        } else {
            std::cout << "[";
            for (size_t i = 0; i < host.size(); ++i) {
                std::cout << host[i];
                if (i < host.size() - 1)
                    std::cout << ", ";
            }
            std::cout << "]";
        }

        std::cout << ")\n";
    }

    void Tensor::print_grad() const {
        if (requires_grad()) {
            af_print(impl_->grad());
        } else {
            af::array empty;
            af_print(empty);
        }
    }

    af::array Tensor::data() const{
        return impl_->data();
    }

    std::shared_ptr<TensorImpl> Tensor::impl() const {
        return impl_;
    }

    af::array Tensor::grad() const {
        if (!requires_grad() || !impl_->has_autograd()) {
            #ifndef NDEBUG
                std::cerr << "[warning] grad() called on tensor with no grad.\n";
            #endif
            return {}; // Empty and safe
        }
        return impl_->grad();
    }

    bool Tensor::requires_grad() const { return impl_->requires_grad(); }

    void Tensor::zero_grad() const {
        if (requires_grad() && impl_->has_autograd()) {
            impl_->grad() = af::constant(0.0f, impl_->data().dims(), impl_->data().type());
        }
    }

    Tensor Tensor::sum(int dim, bool keepdim) const {
        af::array result;

        if (dim == -1) {
            // Sum all elements
            result = af::sum(af::flat(this->data()));
        } else {
            // Sum along specific dimension
            result = af::sum(this->data(), dim);

            if (keepdim) {
                af::dim4 id = this->data().dims();
                std::vector<dim_t> dims = { id[0], id[1], id[2], id[3] };
                dims[dim] = 1;
                af::dim4 keep_dims(dims[0], dims[1], dims[2], dims[3]);
                result = af::moddims(result, keep_dims);
            }
        }

        Tensor out(result, this->requires_grad());

        if (out.requires_grad()) {
            auto fn = std::make_shared<SumFunction>(this->data().dims(), dim, keepdim);
            fn->inputs = { this->impl_ };
            out.impl_->grad_fn() = fn;
        }

        return out;
    }

    Tensor Tensor::mean(int dim, bool keepdim) const {
        af::array result;
        dim_t divisor;

        if (dim == -1) {
            // Mean of all elements
            result = af::mean(af::flat(this->data()));
            divisor = this->data().elements();
        } else {
            // Mean along specific dimension
            result = af::sum(this->data(), dim);
            divisor = this->data().dims(dim);

            if (keepdim) {
                af::dim4 id = this->data().dims();
                std::vector<dim_t> dims = { id[0], id[1], id[2], id[3] };
                dims[dim] = 1;
                af::dim4 keep_dims(dims[0], dims[1], dims[2], dims[3]);
                result = af::moddims(result, keep_dims);
            }

            result = result / static_cast<float>(divisor);
        }

        Tensor out(result, this->requires_grad());

        if (out.requires_grad()) {
            auto fn = std::make_shared<MeanFunction>(this->data().dims(), dim, keepdim);
            fn->inputs = { this->impl_ };
            out.impl_->grad_fn() = fn;
        }

        return out;
    }

    Tensor Tensor::max(int dim, bool keepdim) const {
        af::array result;

        if (dim == -1) {
            // Max over all elements → scalar
            result = af::max<af::array>(af::flat(this->data()));  // correct version
            result = af::moddims(result, af::dim4(1, 1, 1, 1));   // to make it 4D scalar
        } else {
            // Max along dim
            result = af::max(this->data(), dim);

            if (keepdim) {
                af::dim4 shape = this->data().dims();
                shape[dim] = 1;
                result = af::moddims(result, shape);
            }
        }

        Tensor out(result, this->requires_grad());

        if (out.requires_grad()) {
            auto fn = std::make_shared<MaxFunction>(this->data(), dim, keepdim);
            fn->inputs = { this->impl_ };
            out.impl_->grad_fn() = fn;
        }

        return out;
    }

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
        if (!requires_grad() || !impl_->has_autograd()) {
            throw std::runtime_error("You are calling backward on tensor which does not require gradient");
        }

        // Check if this tensor already has a grad
        if (impl_->has_called_backward()) {
            #ifndef NDEBUG
                std::cerr << "[debug] backward() called more than once on the same tensor\n";
            #endif
        }

        impl_->set_has_called_backward(true);

        // seed the root gradient
        impl_->grad() = af::constant(1, impl_->data().dims());
        // then kick off the chain
        if (impl_->grad_fn())
            impl_->grad_fn()->apply(impl_->grad());
    }


} // namespace cppgrad