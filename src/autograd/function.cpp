#include "cppgrad/autograd/function.hpp"
#include "cppgrad/tensor/tensor.hpp"
#include "cppgrad/utils/broadcastUtils.hpp"
#include <stdexcept>
#include <numeric>
#include <limits>

#include "dispatcher/kernelRegistry.h"

namespace cppgrad {

/*
 Helpers:
 - pad_shape_right: pad a shape on the left with 1s so it has n dims
 - compute_strides: compute C-order stride vector for a given shape
 - compute_broadcast_shape: compute output (broadcasted) shape for two inputs
 - reduce_grad: given grad_output (flattened, out_shape), reduce/accumulate into grad_in (shape in_shape)
   by mapping each out index to an in index (broadcasting rules) and summing contributions.
*/

// reduce grad_output (which is in out_sh) to grad_in (which is in in_sh)
// returns grad_in vector (initialized zeros) of size product(in_sh)
static std::vector<float> reduce_grad_to_input(const std::vector<float>& grad_out,
                                               const std::vector<size_t>& out_sh,
                                               const std::vector<size_t>& in_sh) {
    int n = (int)out_sh.size();
    if ((int)in_sh.size() > n) throw std::runtime_error("reduce_grad_to_input: input rank > output rank");

    // pad input shape to n
    auto in_pad = pad_shape_right(in_sh, n);

    // precompute strides
    auto stride_out = compute_strides(out_sh);
    auto stride_in = compute_strides(in_pad);

    size_t total = 1;
    for (auto s : out_sh) total *= s;
    size_t in_total = 1;
    for (auto s : in_pad) in_total *= s;

    std::vector<float> grad_in(in_total, 0.0f);

    // Map each out position to in index (taking broadcast dims into account)
    for (size_t pos = 0; pos < total; ++pos) {
        size_t in_idx = 0;
        for (int d = 0; d < n; ++d) {
            size_t coord = (pos / stride_out[d]) % out_sh[d];
            // if this input dimension is broadcasted (size == 1) -> index 0 for that dim
            if (in_pad[d] == 1) {
                // contributes to same single element => in_idx not changed for this dim
            } else {
                in_idx += coord * stride_in[d];
            }
        }
        grad_in[in_idx] += grad_out[pos];
    }

    // If input was padded (had leading 1s) remove the padding in the returned vector's interpretation:
    // NOTE: grad_in vector length corresponds to padded shape; caller should interpret accordingly.
    // We keep the padded representation since TensorImpl::grad() uses unpadded shape; the caller will work with in_sh.
    // To map to original unpadded layout we must compress dims if needed.
    // But since in_sh corresponds to shape before padding, we need to produce size matching in_sh product.
    // We'll collapse the leading padded dims to produce a vector matching original in_sh.
    if ((int)in_sh.size() < n) {
        // Need to squeeze leading dimensions (which must be size 1)
        int offset = n - (int)in_sh.size();
        // If in_sh.size()==0 (scalar) then return single-element sum
        if (in_sh.empty()) {
            float acc = 0.0f;
            for (float v : grad_in) acc += v;
            return std::vector<float>{acc};
        } else {
            // Build result vector of size product(in_sh)
            size_t target = 1;
            for (auto s : in_sh) target *= s;
            std::vector<float> squeezed(target, 0.0f);
            // iterate over padded indices and map to squeezed index
            // We'll compute coordinates for padded shape and then map to squeezed linear index
            auto in_padded_sh = in_pad; // padded shape
            auto stride_squeezed = compute_strides(in_sh);
            // iterate over all positions in padded shape and add to squeezed
            size_t padded_total = in_total;
            for (size_t pos = 0; pos < padded_total; ++pos) {
                // compute multi-index for padded
                size_t tmp = pos;
                size_t squeezed_idx = 0;
                for (int d = 0; d < n; ++d) {
                    size_t coord = tmp / stride_in[d];
                    tmp = tmp % stride_in[d];
                    if (d >= offset) {
                        // this dimension maps to squeezed dimension index (d-offset)
                        int sd = d - offset;
                        squeezed_idx += coord * stride_squeezed[sd];
                    } else {
                        // padded leading dim (size 1), ignored in squeezed mapping
                    }
                }
                squeezed[squeezed_idx] += grad_in[pos];
            }
            return squeezed;
        }
    }

    return grad_in;
}

/* ---------- Function implementations ---------- */

void AddFunction::apply(const std::vector<float>& grad_output) {
    this->mark_visited();
    if (inputs.size() < 2) throw std::runtime_error("AddFunction requires two inputs");

    auto a_impl = inputs[0];
    auto b_impl = inputs[1];

    // keep device consistency for now
    if (a_impl->device() != b_impl->device())
        throw std::runtime_error("AddFunction::apply: mixed-device inputs not supported");
    DeviceType dev = a_impl->device();

    auto out_sh = compute_broadcast_shape(a_impl->shape(), b_impl->shape());

    Tensor grad_out_tensor(out_sh, grad_output, false, dev);
    Tensor grad_a_tensor(a_impl->shape(), 0.0f, false, dev);
    Tensor grad_b_tensor(b_impl->shape(), 0.0f, false, dev);

    auto bk = KernelRegistry::instance().getBackwardKernelOrNull(OpType::Add, dev);
    if (!bk) throw std::runtime_error("No backward kernel registered for Add (device or CPU)");

    // kernel is responsible for writing grad_a_tensor and grad_b_tensor
    bk(Tensor(a_impl->shape(), a_impl->data(), false, dev),
       Tensor(b_impl->shape(), b_impl->data(), false, dev),
       grad_out_tensor,
       grad_a_tensor,
       grad_b_tensor);

    // Accumulate into inputs and recurse
    if (a_impl->requires_grad()) {
        auto &g0 = a_impl->grad();
        const auto &g_a_data = grad_a_tensor.data();
        if (g0.empty()) g0 = std::vector<float>(g_a_data.size(), 0.0f);
        for (size_t i = 0; i < g_a_data.size(); ++i) g0[i] += g_a_data[i];
        if (a_impl->grad_fn()) a_impl->grad_fn()->apply(g_a_data);
    }

    if (b_impl->requires_grad()) {
        auto &g1 = b_impl->grad();
        const auto &g_b_data = grad_b_tensor.data();
        if (g1.empty()) g1 = std::vector<float>(g_b_data.size(), 0.0f);
        for (size_t i = 0; i < g_b_data.size(); ++i) g1[i] += g_b_data[i];
        if (b_impl->grad_fn()) b_impl->grad_fn()->apply(g_b_data);
    }
}

void MulFunction::apply(const std::vector<float>& grad_output) {
    this->mark_visited();
    if (inputs.size() < 2) throw std::runtime_error("MulFunction requires two inputs");

    auto a_impl = inputs[0];
    auto b_impl = inputs[1];

    if (a_impl->device() != b_impl->device())
        throw std::runtime_error("MulFunction::apply: mixed-device inputs not supported");
    DeviceType dev = a_impl->device();

    auto out_sh = compute_broadcast_shape(a_impl->shape(), b_impl->shape());

    Tensor grad_out_tensor(out_sh, grad_output, false, dev);
    Tensor grad_a_tensor(a_impl->shape(), 0.0f, false, dev);
    Tensor grad_b_tensor(b_impl->shape(), 0.0f, false, dev);

    auto bk = KernelRegistry::instance().getBackwardKernelOrNull(OpType::Mul, dev);
    if (!bk) throw std::runtime_error("No backward kernel registered for Mul (device or CPU)");

    bk(Tensor(a_impl->shape(), a_impl->data(), false, dev),
       Tensor(b_impl->shape(), b_impl->data(), false, dev),
       grad_out_tensor,
       grad_a_tensor,
       grad_b_tensor);

    if (a_impl->requires_grad()) {
        auto &g0 = a_impl->grad();
        const auto &g_a_data = grad_a_tensor.data();
        if (g0.empty()) g0 = std::vector<float>(g_a_data.size(), 0.0f);
        for (size_t i = 0; i < g_a_data.size(); ++i) g0[i] += g_a_data[i];
        if (a_impl->grad_fn()) a_impl->grad_fn()->apply(g_a_data);
    }

    if (b_impl->requires_grad()) {
        auto &g1 = b_impl->grad();
        const auto &g_b_data = grad_b_tensor.data();
        if (g1.empty()) g1 = std::vector<float>(g_b_data.size(), 0.0f);
        for (size_t i = 0; i < g_b_data.size(); ++i) g1[i] += g_b_data[i];
        if (b_impl->grad_fn()) b_impl->grad_fn()->apply(g_b_data);
    }
}

void SubFunction::apply(const std::vector<float>& grad_output) {
    this->mark_visited();
    if (inputs.size() < 2) throw std::runtime_error("SubFunction requires two inputs");

    auto a_impl = inputs[0];
    auto b_impl = inputs[1];

    if (a_impl->device() != b_impl->device())
        throw std::runtime_error("SubFunction::apply: mixed-device inputs not supported");
    DeviceType dev = a_impl->device();

    auto out_sh = compute_broadcast_shape(a_impl->shape(), b_impl->shape());

    Tensor grad_out_tensor(out_sh, grad_output, false, dev);
    Tensor grad_a_tensor(a_impl->shape(), 0.0f, false, dev);
    Tensor grad_b_tensor(b_impl->shape(), 0.0f, false, dev);

    auto bk = KernelRegistry::instance().getBackwardKernelOrNull(OpType::Sub, dev);
    if (!bk) throw std::runtime_error("No backward kernel registered for Sub (device or CPU)");

    bk(Tensor(a_impl->shape(), a_impl->data(), false, dev),
       Tensor(b_impl->shape(), b_impl->data(), false, dev),
       grad_out_tensor,
       grad_a_tensor,
       grad_b_tensor);

    if (a_impl->requires_grad()) {
        auto &g0 = a_impl->grad();
        const auto &g_a_data = grad_a_tensor.data();
        if (g0.empty()) g0 = std::vector<float>(g_a_data.size(), 0.0f);
        for (size_t i = 0; i < g_a_data.size(); ++i) g0[i] += g_a_data[i];
        if (a_impl->grad_fn()) a_impl->grad_fn()->apply(g_a_data);
    }

    if (b_impl->requires_grad()) {
        auto &g1 = b_impl->grad();
        const auto &g_b_data = grad_b_tensor.data();
        if (g1.empty()) g1 = std::vector<float>(g_b_data.size(), 0.0f);
        for (size_t i = 0; i < g_b_data.size(); ++i) g1[i] += g_b_data[i];
        if (b_impl->grad_fn()) b_impl->grad_fn()->apply(g_b_data);
    }
}

void DivFunction::apply(const std::vector<float>& grad_output) {
    this->mark_visited();
    if (inputs.size() < 2) throw std::runtime_error("DivFunction requires two inputs");

    auto a_impl = inputs[0];
    auto b_impl = inputs[1];

    if (a_impl->device() != b_impl->device())
        throw std::runtime_error("DivFunction::apply: mixed-device inputs not supported");
    DeviceType dev = a_impl->device();

    auto out_sh = compute_broadcast_shape(a_impl->shape(), b_impl->shape());

    Tensor grad_out_tensor(out_sh, grad_output, false, dev);
    Tensor grad_a_tensor(a_impl->shape(), 0.0f, false, dev);
    Tensor grad_b_tensor(b_impl->shape(), 0.0f, false, dev);

    auto bk = KernelRegistry::instance().getBackwardKernelOrNull(OpType::Div, dev);
    if (!bk) throw std::runtime_error("No backward kernel registered for Div (device or CPU)");

    bk(Tensor(a_impl->shape(), a_impl->data(), false, dev),
       Tensor(b_impl->shape(), b_impl->data(), false, dev),
       grad_out_tensor,
       grad_a_tensor,
       grad_b_tensor);

    if (a_impl->requires_grad()) {
        auto &g0 = a_impl->grad();
        const auto &g_a_data = grad_a_tensor.data();
        if (g0.empty()) g0 = std::vector<float>(g_a_data.size(), 0.0f);
        for (size_t i = 0; i < g_a_data.size(); ++i) g0[i] += g_a_data[i];
        if (a_impl->grad_fn()) a_impl->grad_fn()->apply(g_a_data);
    }

    if (b_impl->requires_grad()) {
        auto &g1 = b_impl->grad();
        const auto &g_b_data = grad_b_tensor.data();
        if (g1.empty()) g1 = std::vector<float>(g_b_data.size(), 0.0f);
        for (size_t i = 0; i < g_b_data.size(); ++i) g1[i] += g_b_data[i];
        if (b_impl->grad_fn()) b_impl->grad_fn()->apply(g_b_data);
    }
}


} // namespace cppgrad

    // //----------------Clone---------------------------
    // void CloneFunction::apply(const af::array &grad_output) {
    //     this->mark_visited();
    //     inputs[0]->grad() = grad_output.copy();
    // }
    //
    // std::string CloneFunction::name() const {
    //     return "Clone";
    // }
    //
    // //----------------Matmul---------------------------
    // void MatMulFunction::apply(const af::array& grad_output) {
    //     this->mark_visited();
    //     // inputs[0] = a, inputs[1] = b
    //     const af::array& a = inputs[0]->data();   // shape: (M × K)
    //     const af::array& b = inputs[1]->data();   // shape: (K × N)
    //
    //     // ∂L/∂a = grad_output @ bᵀ  ==> shape: (M × N) @ (N × K) = (M × K)
    //     if (inputs[0]->requires_grad()) {
    //         af::array grad_a = af::matmul(grad_output, af::transpose(b));
    //         inputs[0]->grad() += grad_a;
    //         if (inputs[0]->grad_fn())
    //             inputs[0]->grad_fn()->apply(grad_a);
    //     }
    //
    //     // ∂L/∂b = aᵀ @ grad_output  ==> shape: (K × M) @ (M × N) = (K × N)
    //     if (inputs[1]->requires_grad()) {
    //         af::array grad_b = af::matmul(af::transpose(a), grad_output);
    //         inputs[1]->grad() += grad_b;
    //         if (inputs[1]->grad_fn())
    //             inputs[1]->grad_fn()->apply(grad_b);
    //     }
    // }
    //
    // std::string MatMulFunction::name() const {
    //     return "MatMul";
    // }
    //
    // //----------------Neg---------------------------
    // void NegFunction::apply(const af::array& grad_output) {
    //     this->mark_visited();
    //
    //     if (inputs[0]->requires_grad()) {
    //         inputs[0]->grad() += -grad_output;
    //
    //         if (inputs[0]->grad_fn()) {
    //             inputs[0]->grad_fn()->apply(-grad_output);
    //         }
    //     }
    // }
    //
    // std::string NegFunction::name() const {
    //     return "Neg";
    // }
    //
    // //----------------Exp---------------------------
    // void ExpFunction::apply(const af::array& grad_output) {
    //     this->mark_visited();
    //
    //     const af::array& a = inputs[0]->data();
    //     af::array exp_a = af::exp(a);
    //
    //     if (inputs[0]->requires_grad()) {
    //         af::array grad_input = exp_a * grad_output;
    //         inputs[0]->grad() += grad_input;
    //
    //         if (inputs[0]->grad_fn()) {
    //             inputs[0]->grad_fn()->apply(grad_input);
    //         }
    //     }
    // }
    //
    // std::string ExpFunction::name() const {
    //     return "Exp";
    // }
    //
    // //----------------Log---------------------------
    // void LogFunction::apply(const af::array& grad_output) {
    //     this->mark_visited();
    //
    //     const af::array& a = inputs[0]->data();
    //
    //     if (inputs[0]->requires_grad()) {
    //         af::array grad_input = grad_output / a;
    //         inputs[0]->grad() += grad_input;
    //
    //         if (inputs[0]->grad_fn()) {
    //             inputs[0]->grad_fn()->apply(grad_input);
    //         }
    //     }
    // }
    //
    // std::string LogFunction::name() const {
    //     return "Log";
    // }
    //
    // //----------------Pow---------------------------
    // void PowFunction::apply(const af::array& grad_output) {
    //     this->mark_visited();
    //
    //     const af::array& base = inputs[0]->data();
    //     const af::array& exponent = inputs[1]->data();
    //     af::array output = af::pow(base, exponent);
    //
    //     if (inputs[0]->requires_grad()) {
    //         af::array grad_base = exponent * af::pow(base, exponent - 1) * grad_output;
    //         inputs[0]->grad() += grad_base;
    //
    //         if (inputs[0]->grad_fn()) {
    //             inputs[0]->grad_fn()->apply(grad_base);
    //         }
    //     }
    //
    //     if (inputs[1]->requires_grad()) {
    //         af::array grad_exp = output * af::log(base) * grad_output;
    //         inputs[1]->grad() += grad_exp;
    //
    //         if (inputs[1]->grad_fn()) {
    //             inputs[1]->grad_fn()->apply(grad_exp);
    //         }
    //     }
    // }
    //
    // std::string PowFunction::name() const {
    //     return "Pow";
    // }
    //
    // //----------------Sum---------------------------
    //
    // SumFunction::SumFunction(const af::dim4& input_shape, int dim, bool keepdim)
    // : input_shape_(input_shape), dim_(dim), keepdim_(keepdim) {}
    //
    // void SumFunction::apply(const af::array& grad_output) {
    //     this->mark_visited();
    //
    //     const auto& input = inputs[0];
    //     const af::array& input_data = input->data();
    //
    //     if (!input->requires_grad()) return;
    //
    //     af::array grad_input;
    //
    //     if (dim_ == -1) {
    //         // Gradient of sum over all elements: fill with ones and multiply by grad_output scalar
    //         grad_input = af::constant(1.0f, input_shape_) * grad_output;
    //     } else {
    //         // Sum over specific dim
    //         // If keepdim == false, we must expand grad_output shape before broadcasting
    //         af::array grad = grad_output;
    //         if (!keepdim_) {
    //             // Insert singleton dimension back for broadcasting
    //             std::vector<dim_t> dims = {
    //                 input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3]
    //             };
    //             dims[dim_] = 1;  // insert singleton
    //             grad = af::moddims(grad_output, af::dim4(dims[0], dims[1], dims[2], dims[3]));
    //         }
    //
    //         // Broadcast grad to match input shape
    //         grad_input = af::tile(grad, get_tile_repeats(input_shape_, grad.dims()));
    //     }
    //
    //     input->grad() += grad_input;
    //
    //     if (input->grad_fn()) {
    //         input->grad_fn()->apply(grad_input);
    //     }
    // }
    //
    // std::string SumFunction::name() const {
    //     return "Sum";
    // }
    //
    // //----------------Mean---------------------------
    // MeanFunction::MeanFunction(const af::dim4& input_shape, int dim, bool keepdim)
    //     : input_shape_(input_shape), dim_(dim), keepdim_(keepdim) {}
    //
    //
    // void MeanFunction::apply(const af::array& grad_output) {
    //     this->mark_visited();
    //
    //     const auto& input = inputs[0];
    //     const af::array& input_data = input->data();
    //
    //     if (!input->requires_grad()) return;
    //
    //     af::array grad_input;
    //
    //     if (dim_ == -1) {
    //         // Mean over all elements: gradient is 1/N broadcasted to input shape
    //         dim_t N = input_shape_.elements();
    //         grad_input = af::constant(1.0f / static_cast<float>(N), input_shape_) * grad_output;
    //     } else {
    //         // Mean over specific dim
    //         dim_t N = input_shape_[dim_];
    //         af::array grad = grad_output;
    //
    //         if (!keepdim_) {
    //             // Insert singleton dimension back for broadcasting
    //             std::vector<dim_t> dims = {
    //                 input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3]
    //             };
    //             dims[dim_] = 1;
    //             grad = af::moddims(grad_output, af::dim4(dims[0], dims[1], dims[2], dims[3]));
    //         }
    //
    //         // Scale the gradient by 1/N
    //         grad = grad / static_cast<float>(N);
    //
    //         // Broadcast grad to match input shape
    //         grad_input = af::tile(grad, get_tile_dims(input_shape_, dim_));
    //     }
    //
    //     input->grad() += grad_input;
    //
    //     if (input->grad_fn()) {
    //         input->grad_fn()->apply(grad_input);
    //     }
    // }
    //
    // std::string MeanFunction::name() const {
    //     return "Mean";
    // }
    //
    // //----------------Max---------------------------
    //
    // MaxFunction::MaxFunction(const af::array& input_data, int dim, bool keepdim)
    // : input_data_(input_data), dim_(dim), keepdim_(keepdim), input_shape_(input_data.dims()) {}
    //
    //
    // void MaxFunction::apply(const af::array& grad_output) {
    //     this->mark_visited();
    //
    //     const auto& input = inputs[0];
    //     if (!input->requires_grad()) return;
    //
    //     af::array grad_mask;
    //     if (dim_ == -1) {
    //         // Global max: compare with scalar
    //         float max_val = af::max<float>(af::flat(input_data_));
    //         grad_mask = (input_data_ == max_val);
    //     } else {
    //         // Max along dimension: build broadcasted mask
    //         af::array max_vals = af::max(input_data_, dim_);
    //
    //         if (!keepdim_) {
    //             // Insert singleton for broadcasting
    //             std::vector<dim_t> dims = {
    //                 input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3]
    //             };
    //             dims[dim_] = 1;
    //             max_vals = af::moddims(max_vals, af::dim4(dims[0], dims[1], dims[2], dims[3]));
    //         }
    //
    //         grad_mask = (input_data_ == af::tile(max_vals, get_tile_dims(input_shape_, dim_)));
    //     }
    //
    //     af::array grad = grad_output;
    //
    //     if (!keepdim_ && dim_ != -1) {
    //         // Insert singleton for broadcasting
    //         std::vector<dim_t> dims = {
    //             input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3]
    //         };
    //         dims[dim_] = 1;
    //         grad = af::moddims(grad_output, af::dim4(dims[0], dims[1], dims[2], dims[3]));
    //     }
    //
    //     // Broadcast grad to input shape
    //     if (dim_ != -1) {
    //         grad = af::tile(grad, get_tile_dims(input_shape_, dim_));
    //     }
    //
    //     // Apply mask: gradient only to positions that had the max value
    //     af::array grad_input = grad * grad_mask.as(f32);
    //
    //     input->grad() += grad_input;
    //
    //     if (input->grad_fn()) {
    //         input->grad_fn()->apply(grad_input);
    //     }
    // }
    //
    // std::string MaxFunction::name() const {
    //     return "Max";
    // }

