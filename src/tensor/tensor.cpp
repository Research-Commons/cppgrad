#include "cppgrad/tensor/tensor.hpp"

#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <iostream>

#include "autograd/function.hpp"

namespace cppgrad {

// ---------- Constructors ----------
Tensor::Tensor(const std::vector<size_t>& shape,
               const std::vector<float>& values,
               bool requires_grad,
               DeviceType device)
    : impl_(std::make_shared<TensorImpl>(shape, values, requires_grad, device))
{}

Tensor::Tensor(const std::vector<size_t>& shape,
               float value,
               bool requires_grad,
               DeviceType device)
    : impl_(std::make_shared<TensorImpl>(shape, value, requires_grad, device))
{}

Tensor::Tensor(std::shared_ptr<TensorImpl> impl)
    : impl_(std::move(impl))
{}

// ---------- Factories ----------
Tensor Tensor::zeros(const std::vector<size_t>& shape,
                     bool requires_grad,
                     DeviceType device) {
    return Tensor(shape, 0.0f, requires_grad, device);
}

Tensor Tensor::ones(const std::vector<size_t>& shape,
                    bool requires_grad,
                    DeviceType device) {
    return Tensor(shape, 1.0f, requires_grad, device);
}

Tensor Tensor::full(const std::vector<size_t>& shape,
                    float value,
                    bool requires_grad,
                    DeviceType device) {
    return Tensor(shape, value, requires_grad, device);
}

Tensor Tensor::randn(const std::vector<size_t>& shape,
                     bool requires_grad,
                     DeviceType device) {
    size_t total = 1;
    for (auto s : shape) total *= s;
    std::vector<float> data(total);
    static thread_local std::mt19937_64 gen((unsigned)std::random_device{}());
    std::normal_distribution<float> d(0.0f, 1.0f);
    for (size_t i = 0; i < total; ++i) data[i] = d(gen);
    return Tensor(shape, data, requires_grad, device);
}

// ---------- Shape & Info ----------
std::vector<size_t> Tensor::shape() const { return impl_->shape(); }
size_t Tensor::numel() const { return impl_->numel(); }
size_t Tensor::ndim() const { return impl_->shape().size(); }
bool Tensor::requires_grad() const { return impl_->requires_grad(); }
DeviceType Tensor::device_type() const { return impl_->device(); }


void Tensor::zero_grad() {
    auto &g = impl_->grad();
    std::fill(g.begin(), g.end(), 0.0f);
}

void Tensor::print() const {
    const auto &s = impl_->shape();
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < s.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << s[i];
    }
    std::cout << "], values=[";
    const auto &d = impl_->data();
    for (size_t i = 0; i < d.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << d[i];
        if (i >= 31) { std::cout << ", ..."; break; }
    }
    std::cout << "])\n";
}

void Tensor::print_pretty() const {
    // small pretty printer: only for 1D or 2D tensors
    const auto s = impl_->shape();
    const auto &d = impl_->data();
    if (s.size() == 1) {
        std::cout << "[";
        for (size_t i = 0; i < s[0]; ++i) {
            if (i) std::cout << ", ";
            std::cout << d[i];
        }
        std::cout << "]\n";
    } else if (s.size() == 2) {
        for (size_t r = 0; r < s[0]; ++r) {
            std::cout << "[";
            for (size_t c = 0; c < s[1]; ++c) {
                if (c) std::cout << ", ";
                std::cout << d[r * s[1] + c];
            }
            std::cout << "]\n";
        }
    } else {
        print();
    }
}

void Tensor::print_grad() const {
    if (!impl_->requires_grad()) {
        std::cout << "No grad (requires_grad == false)\n";
        return;
    }
    if (impl_->grad().empty()) {
        std::cout << "grad: (empty)\n";
        return;
    }
    const auto &g = impl_->grad();
    std::cout << "grad=[";
    for (size_t i = 0; i < g.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << g[i];
        if (i >= 31) { std::cout << ", ..."; break; }
    }
    std::cout << "]\n";
}

// -------- Minimal autograd support --------
void Tensor::backward(const std::vector<float>& grad_output) {
    if (!impl_->requires_grad()) {
        throw std::runtime_error("Tensor::backward: tensor does not require grad");
    }

    size_t n = numel();
    if (!grad_output.empty()) {
        if (grad_output.size() != n)
            throw std::runtime_error("Tensor::backward: grad_output size does not match tensor");
        impl_->grad() = grad_output; // copy
    } else {
        // default: if scalar -> grad 1.0, else ones
        impl_->grad().assign(n, 1.0f);
    }

    impl_->set_has_called_backward(true);

    // Hook for a real autograd graph traversal:
    if (impl_->has_autograd() && impl_->grad_fn()) {
        // TODO: Call grad_fn->apply(...) / traverse graph
        // For now, just a stub to show where autograd would run.
        // e.g., impl_->grad_fn()->backward(impl_->grad());
        impl_->grad_fn()->apply(impl_->grad());
    }
}

const std::vector<float>& Tensor::grad() const {
    return impl_->grad();
}

// Data access
const std::vector<float>& Tensor::data() const { return impl_->data(); }
std::vector<float>& Tensor::data() { return impl_->data(); }
std::shared_ptr<TensorImpl> Tensor::impl() const { return impl_; }

} // namespace cppgrad

// // -------- Reductions (only global dim implemented for now) --------
// Tensor Tensor::sum(int dim, bool keepdim) const {
//     if (dim != -1) throw std::runtime_error("sum(dim) not implemented yet (only global sum)");
//     float acc = 0.0f;
//     for (float v : impl_->data()) acc += v;
//     // return scalar as shape {1}
//     return Tensor({1}, std::vector<float>{acc}, false);
// }
//
// Tensor Tensor::mean(int dim, bool keepdim) const {
//     if (dim != -1) throw std::runtime_error("mean(dim) not implemented yet (only global mean)");
//     float acc = 0.0f;
//     for (float v : impl_->data()) acc += v;
//     return Tensor({1}, std::vector<float>{acc / (float)numel()}, false);
// }
//
// Tensor Tensor::max(int dim, bool keepdim) const {
//     if (dim != -1) throw std::runtime_error("max(dim) not implemented yet (only global max)");
//     const auto &d = impl_->data();
//     if (d.empty()) throw std::runtime_error("max on empty tensor");
//     float m = d[0];
//     for (auto v : d) if (v > m) m = v;
//     return Tensor({1}, std::vector<float>{m}, false);
// }
//
// // -------- Elementwise operators (simple implementations) --------
// static void check_shapes_match(const Tensor& a, const Tensor& b) {
//     if (a.shape() != b.shape()) throw std::runtime_error("shape mismatch in binary op");
// }
//
// Tensor operator+(const Tensor& a, const Tensor& b) {
//     check_shapes_match(a,b);
//     auto out = Tensor::full(a.shape(), 0.0f, a.requires_grad() || b.requires_grad());
//     const auto &Ad = a.data();
//     const auto &Bd = b.data();
//     auto &Od = out.data();
//     size_t N = Ad.size();
//     for (size_t i = 0; i < N; ++i) Od[i] = Ad[i] + Bd[i];
//     return out;
// }
//
// Tensor operator-(const Tensor& a, const Tensor& b) {
//     check_shapes_match(a,b);
//     auto out = Tensor::full(a.shape(), 0.0f, a.requires_grad() || b.requires_grad());
//     const auto &Ad = a.data();
//     const auto &Bd = b.data();
//     auto &Od = out.data();
//     size_t N = Ad.size();
//     for (size_t i = 0; i < N; ++i) Od[i] = Ad[i] - Bd[i];
//     return out;
// }
//
// Tensor operator*(const Tensor& a, const Tensor& b) {
//     check_shapes_match(a,b);
//     auto out = Tensor::full(a.shape(), 0.0f, a.requires_grad() || b.requires_grad());
//     const auto &Ad = a.data();
//     const auto &Bd = b.data();
//     auto &Od = out.data();
//     size_t N = Ad.size();
//     for (size_t i = 0; i < N; ++i) Od[i] = Ad[i] * Bd[i];
//     return out;
// }
//
// Tensor operator/(const Tensor& a, const Tensor& b) {
//     check_shapes_match(a,b);
//     auto out = Tensor::full(a.shape(), 0.0f, a.requires_grad() || b.requires_grad());
//     const auto &Ad = a.data();
//     const auto &Bd = b.data();
//     auto &Od = out.data();
//     size_t N = Ad.size();
//     for (size_t i = 0; i < N; ++i) {
//         if (Bd[i] == 0.0f) Od[i] = std::numeric_limits<float>::infinity();
//         else Od[i] = Ad[i] / Bd[i];
//     }
//     return out;
// }
//
// // Scalar ops
// Tensor operator+(const Tensor& a, float s) {
//     auto out = Tensor::full(a.shape(), 0.0f, a.requires_grad());
//     const auto &Ad = a.data();
//     auto &Od = out.data();
//     for (size_t i = 0; i < Ad.size(); ++i) Od[i] = Ad[i] + s;
//     return out;
// }
// Tensor operator+(float s, const Tensor& a) { return a + s; }
//
// Tensor operator-(const Tensor& a, float s) {
//     auto out = Tensor::full(a.shape(), 0.0f, a.requires_grad());
//     const auto &Ad = a.data();
//     auto &Od = out.data();
//     for (size_t i = 0; i < Ad.size(); ++i) Od[i] = Ad[i] - s;
//     return out;
// }
// Tensor operator-(float s, const Tensor& a) {
//     auto out = Tensor::full(a.shape(), 0.0f, a.requires_grad());
//     const auto &Ad = a.data();
//     auto &Od = out.data();
//     for (size_t i = 0; i < Ad.size(); ++i) Od[i] = s - Ad[i];
//     return out;
// }
//
// Tensor operator*(const Tensor& a, float s) {
//     auto out = Tensor::full(a.shape(), 0.0f, a.requires_grad());
//     const auto &Ad = a.data();
//     auto &Od = out.data();
//     for (size_t i = 0; i < Ad.size(); ++i) Od[i] = Ad[i] * s;
//     return out;
// }
// Tensor operator*(float s, const Tensor& a) { return a * s; }
//
// Tensor operator/(const Tensor& a, float s) {
//     auto out = Tensor::full(a.shape(), 0.0f, a.requires_grad());
//     const auto &Ad = a.data();
//     auto &Od = out.data();
//     if (s == 0.0f) {
//         for (size_t i = 0; i < Ad.size(); ++i) Od[i] = std::numeric_limits<float>::infinity();
//     } else {
//         for (size_t i = 0; i < Ad.size(); ++i) Od[i] = Ad[i] / s;
//     }
//     return out;
// }
// Tensor operator/(float s, const Tensor& a) {
//     auto out = Tensor::full(a.shape(), 0.0f, a.requires_grad());
//     const auto &Ad = a.data();
//     auto &Od = out.data();
//     for (size_t i = 0; i < Ad.size(); ++i) {
//         if (Ad[i] == 0.0f) Od[i] = std::numeric_limits<float>::infinity();
//         else Od[i] = s / Ad[i];
//     }
//     return out;
// }
//
// Tensor operator-(const Tensor& a) {
//     return a * (-1.0f);
// }
