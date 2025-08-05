#include "tensor/tensor.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "autograd/function.hpp"

namespace cppgrad {

    // ----------------------------------------
    // Constructors - Public
    // ----------------------------------------

    /// Main constructor: takes a row-major values vector, reshapes and reorders
    /// it into ArrayFire’s column-major layout, then wraps it in TensorImpl.
    ///
    /// Steps:
    /// 1. Validate that values.size() equals product(shape).
    /// 2. Load raw 1D data into an ArrayFire array.
    /// 3. Reverse `shape` to get dimensions for the column-major reshape.
    /// 4. Call `af::moddims` to reshape into reversed dims.
    /// 5. Build an axes list `[0,1,2,…]`, reverse it, then call `af::reorder`
    ///    to permute back to the original axis order.
    /// 6. Store the result in a new `TensorImpl`.
    ///
    /// ArrayFire specifics:
    /// - Uses **column-major** storage: the fastest-moving index is the first (`dim0`).
    /// - `af::dim4(d0,d1,d2,d3)` corresponds to sizes in x,y,z,w axes.
    /// - `moddims`/`reorder` reinterpret the same buffer; no copy of data.
    ///
    /// How this plays out for 3D and 4D shapes:
    ///  • **3D** `(D0, D1, D2)`
    ///    - Input shape vector: `{D0, D1, D2}`
    ///    - `to_dim4` yields `(D0, D1, D2, 1)`
    ///    - Reverse → `(D2, D1, D0, 1)`, reshape to that, then reorder axes
    ///      back to `(D0, D1, D2)` via `af::reorder` overload for 3 dims.
    ///
    ///  • **4D** `(D0, D1, D2, D3)`
    ///    - Input shape vector: `{D0, D1, D2, D3}`
    ///    - `to_dim4` yields exactly `(D0, D1, D2, D3)`
    ///    - Reverse → `(D3, D2, D1, D0)`, reshape to that, then reorder axes
    ///      via the 4-arg `af::reorder` call to `(D0, D1, D2, D3)`.
    Tensor::Tensor(const std::vector<size_t>& shape,
                   const std::vector<float>& values,
                   bool requires_grad) {
        // 1) Shape → af::dim4; verify element count
        af::dim4 dims = to_dim4(shape);
        size_t expected = dims.elements();
        if (values.size() != expected) {
            throw std::invalid_argument("Number of values does not match shape");
        }

        // 2) Load raw linear data
        af::array arr(static_cast<unsigned>(values.size()), values.data());

        // 3) Reverse shape for column-major reshape
        std::vector<size_t> rev_shape(shape.rbegin(), shape.rend());
        af::dim4 rev_dims = to_dim4(rev_shape);

        // 4) Reshape into reversed dims
        arr = af::moddims(arr, rev_dims);

        // 5) Build and reverse axes list to restore original ordering
        std::vector<unsigned> axes(shape.size());
        std::iota(axes.begin(), axes.end(), 0);
        std::reverse(axes.begin(), axes.end());

        // 6) Reorder axes back to row-major layout
        switch (axes.size()) {
            case 1:
                // 1D: no-op
                break;
            case 2:
                arr = af::reorder(arr, axes[0], axes[1]);
                break;
            case 3:
                arr = af::reorder(arr, axes[1], axes[2], axes[0]);
                break;
            case 4:
                arr = af::reorder(arr, axes[3], axes[2], axes[1], axes[0]);
                break;
            default:
                throw std::runtime_error("Tensor constructor only supports up to 4D");
        }

        // Store in impl
        impl_ = std::make_shared<TensorImpl>(arr, requires_grad);
    }

    // ----------------------------------------
    // Constructors - Private
    // ----------------------------------------

    /// Wrap an existing TensorImpl pointer. Used internally.
    Tensor::Tensor(std::shared_ptr<TensorImpl> impl)
        : impl_(std::move(impl)) { }

    /// Construct directly from an ArrayFire array (no reshape/reorder).
    Tensor::Tensor(const af::array& arr, bool requires_grad)
        : impl_(std::make_shared<TensorImpl>(arr, requires_grad)) { }

    // ----------------------------------------
    // Factory Methods
    // ----------------------------------------

    /// Create a zero-filled tensor.
    Tensor Tensor::zeros(const std::vector<size_t>& shape, bool requires_grad) {
        af::dim4 dims = to_dim4(shape);
        return { af::constant(0.0f, dims), requires_grad };
    }

    /// Create a one-filled tensor.
    Tensor Tensor::ones(const std::vector<size_t>& shape, bool requires_grad) {
        af::dim4 dims = to_dim4(shape);
        return { af::constant(1.0f, dims), requires_grad };
    }

    /// Create a tensor with all values = `value`.
    Tensor Tensor::full(const std::vector<size_t>& shape,
                        float value,
                        bool requires_grad) {
        af::dim4 dims = to_dim4(shape);
        return { af::constant(value, dims), requires_grad };
    }

    /// Create a tensor of Gaussian noise.
    Tensor Tensor::randn(const std::vector<size_t>& shape, bool requires_grad) {
        af::dim4 dims = to_dim4(shape);
        return { af::randn(dims), requires_grad };
    }

    /// Build tensor from a column-major values vector (simpler than main ctor).
    Tensor Tensor::from_array_column_major(const std::vector<size_t>& shape,
                                           const std::vector<float>& values,
                                           bool requires_grad) {
        // Validate size
        size_t expected = 1;
        for (auto s : shape) expected *= s;
        if (values.size() != expected) {
            throw std::invalid_argument("Value count doesn't match shape");
        }
        // Reshape directly
        af::dim4 dims = to_dim4(shape);
        af::array arr(static_cast<unsigned>(values.size()), values.data());
        arr = af::moddims(arr, dims);
        return { std::make_shared<TensorImpl>(arr, requires_grad) };
    }

    // ----------------------------------------
    // Shape & Metadata
    // ----------------------------------------

    /// Return tensor shape as vector<size_t>.
    std::vector<size_t> Tensor::shape() const {
        af::dim4 d = impl_->data().dims();
        std::vector<size_t> out;
        for (int i = 0; i < 4 && d[i] > 1; ++i) {
            out.push_back(d[i]);
        }
        return out;
    }

    /// Total number of elements.
    size_t Tensor::numel() const {
        return impl_->data().elements();
    }

    /// Number of dimensions (1–4).
    size_t Tensor::ndim() const {
        return impl_->data().numdims();
    }

    // ----------------------------------------
    // I/O
    // ----------------------------------------

    /// Raw ArrayFire print.
    void Tensor::print() const {
        af_print(impl_->data());
    }

    /// Human-readable print with shape & flat values list.
    void Tensor::print_pretty() const {
        af::array data = impl_->data();
        std::vector<float> host(data.elements());
        data.host(host.data());

        // Header
        std::cout << "Tensor(shape=[";
        for (int i = 0; i < data.numdims(); ++i) {
            std::cout << data.dims()[i];
            if (i + 1 < data.numdims()) std::cout << ", ";
        }
        std::cout << "], values=";

        // Content
        if (host.size() == 1) {
            std::cout << host[0];
        } else {
            std::cout << "[";
            for (size_t i = 0; i < host.size(); ++i) {
                std::cout << host[i]
                          << (i + 1 < host.size() ? ", " : "");
            }
            std::cout << "]";
        }
        std::cout << ")\n";
    }

    /// Print gradient array (or empty if none).
    void Tensor::print_grad() const {
        if (requires_grad()) {
            af_print(impl_->grad());
        } else {
            af_print(af::array());  // prints nothing
        }
    }

    // ----------------------------------------
    // Autograd Control
    // ----------------------------------------

    /// Does this tensor require a gradient?
    bool Tensor::requires_grad() const {
        return impl_->requires_grad();
    }

    /// Reset stored gradient to zeros.
    void Tensor::zero_grad() const {
        if (requires_grad() && impl_->has_autograd()) {
            impl_->grad() = af::constant(
                0.0f,
                impl_->data().dims(),
                impl_->data().type()
            );
        }
    }

    /// Retrieve the stored gradient array (if any).
    af::array Tensor::grad() const {
        if (!requires_grad() || !impl_->has_autograd()) {
    #ifndef NDEBUG
            std::cerr << "[warning] grad() called on tensor with no grad.\n";
    #endif
            return {};
        }
        return impl_->grad();
    }

    /// Backpropagate from this tensor’s value (seeded with ones).
    /// Throws if tensor wasn’t created with requires_grad=true.
    void Tensor::backward(const af::array & /*ignored*/) {
        if (!requires_grad() || !impl_->has_autograd()) {
            throw std::runtime_error(
                "You are calling backward on tensor which does not require gradient"
            );
        }

        if (impl_->has_called_backward()) {
    #ifndef NDEBUG
            std::cerr << "[debug] backward() called more than once on the same tensor\n";
    #endif
        }

        impl_->set_has_called_backward(true);
        // Seed gradient = 1 for all elements
        impl_->grad() = af::constant(1, impl_->data().dims());

        // Recursively apply stored Function nodes
        if (impl_->grad_fn()) {
            impl_->grad_fn()->apply(impl_->grad());
        }
    }

    // ----------------------------------------
    // Data Access
    // ----------------------------------------

    /// Direct access to the underlying ArrayFire array.
    af::array Tensor::data() const {
        return impl_->data();
    }

    /// Access to internal implementation (for advanced use).
    std::shared_ptr<TensorImpl> Tensor::impl() const {
        return impl_;
    }

    // ----------------------------------------
    // Reduction Operations
    // ----------------------------------------

    /// Sum of elements. If dim==-1 sums all, otherwise along `dim`.
    /// If keepdim=true, retains reduced dimension as size=1.
    Tensor Tensor::sum(int dim, bool keepdim) const {
        af::array result;
        if (dim == -1) {
            result = af::sum(af::flat(this->data()));
        } else {
            result = af::sum(this->data(), dim);
            if (keepdim) {
                af::dim4 d = this->data().dims();
                d[dim] = 1;
                result = af::moddims(result, d);
            }
        }

        Tensor out(result, requires_grad());
        if (out.requires_grad()) {
            auto fn = std::make_shared<SumFunction>(
                this->data().dims(), dim, keepdim
            );
            fn->inputs = { impl_ };
            out.impl_->grad_fn() = fn;
        }
        return out;
    }

    /// Mean of elements (divides sum by count).
    /// Behavior and keepdim logic similar to sum().
    Tensor Tensor::mean(int dim, bool keepdim) const {
        af::array result;
        dim_t count;
        if (dim == -1) {
            result = af::mean(af::flat(this->data()));
            count = this->data().elements();
        } else {
            result = af::sum(this->data(), dim);
            count = this->data().dims(dim);
            if (keepdim) {
                af::dim4 d = this->data().dims();
                d[dim] = 1;
                result = af::moddims(result, d);
            }
            result /= static_cast<float>(count);
        }

        Tensor out(result, requires_grad());
        if (out.requires_grad()) {
            auto fn = std::make_shared<MeanFunction>(
                this->data().dims(), dim, keepdim
            );
            fn->inputs = { impl_ };
            out.impl_->grad_fn() = fn;
        }
        return out;
    }

    /// Maximum of elements. dim==-1 → global max (scalar), otherwise along `dim`.
    /// Retains dimension when keepdim=true.
    Tensor Tensor::max(int dim, bool keepdim) const {
        af::array result;
        if (dim == -1) {
            result = af::max<af::array>(af::flat(this->data()));
            result = af::moddims(result, af::dim4(1,1,1,1));
        } else {
            result = af::max(this->data(), dim);
            if (keepdim) {
                af::dim4 d = this->data().dims();
                d[dim] = 1;
                result = af::moddims(result, d);
            }
        }

        Tensor out(result, requires_grad());
        if (out.requires_grad()) {
            auto fn = std::make_shared<MaxFunction>(
                this->data(), dim, keepdim
            );
            fn->inputs = { impl_ };
            out.impl_->grad_fn() = fn;
        }
        return out;
    }


    // ----------------------------------------
    // Utility
    // ----------------------------------------

    /// Convert a shape vector (row-major) into ArrayFire’s 4D dims.
    /// If shape.size() > 4, higher dimensions are ignored.
    af::dim4 Tensor::to_dim4(const std::vector<size_t>& shape) {
        af::dim4 dims(1, 1, 1, 1);
        for (size_t i = 0; i < std::min(static_cast<size_t>(4), shape.size()); ++i) {
            dims[i] = shape[i];
        }
        return dims;
    }

}  // namespace cppgrad
