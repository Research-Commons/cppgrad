// #pragma once
//
// #include <memory>
// #include <arrayfire.h>
//
// #include "cppgrad/autograd/autogradmeta.hpp"
//
// namespace cppgrad {
//
//     /**
//      * @file tensorimpl.hpp
//      * @brief Internal tensor implementation class for cppgrad.
//      *
//      * `TensorImpl` is the internal representation of a tensor, holding both the raw
//      * data (as an `af::array`) and optional autograd metadata. This class is not
//      * exposed directly to users—instead, it is wrapped by the public `Tensor` class.
//      *
//      * Responsibilities:
//      * - Stores the tensor data using ArrayFire (`af::array`)
//      * - Maintains autograd metadata when `requires_grad` is true
//      *   - Gradient (`grad`)
//      *   - Backward function (`grad_fn`)
//      *   - Bookkeeping (`has_called_backward`)
//      *
//      * Design Notes:
//      * - Uses `std::unique_ptr<AutogradMeta>` to lazily allocate autograd info only when needed
//      * - Supports both const and mutable access to data and gradients
//      * - Gradients are computed during the backward pass and stored here
//      *
//      * Analogy: Similar to `at::TensorImpl` in PyTorch's C++ internals.
//     */
//
//     class Function;
//     //class AutogradMeta;
//
//     class TensorImpl {
//     public:
//         // -------- Constructor --------
//         TensorImpl(const af::array& d, bool requires_grad);
//
//         // -------- Data Access --------
//         const af::array& data() const;
//         af::array& data();
//
//         // -------- Autograd Info --------
//         bool requires_grad() const;
//         bool has_autograd() const;
//
//         af::array& grad();
//         const af::array& grad() const;
//
//         std::shared_ptr<Function>& grad_fn();
//         const std::shared_ptr<Function>& grad_fn() const;
//
//         bool has_called_backward() const;
//         void set_has_called_backward(bool has_called_backwards);
//
//     private:
//         af::array data_;                                // Underlying ArrayFire data
//         std::unique_ptr<AutogradMeta> autograd_;        // Autograd metadata (optional)
//     };
//
// } // namespace cppgrad
