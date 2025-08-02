#include "autograd/function.hpp"
#include "tensor/tensorimpl.hpp"

namespace cppgrad {

    //----------------Add---------------------------
    void AddFunction::apply(const af::array &grad_output) {
        this->mark_visited();
        if (inputs[0]->requires_grad()) {
            inputs[0]->grad() += grad_output;

            if (inputs[0]->grad_fn()) {
                inputs[0]->grad_fn()->apply(grad_output);
            }
        }

        if (inputs[1]->requires_grad()) {
            inputs[1]->grad() += grad_output;

            if (inputs[1]->grad_fn()) {
                inputs[1]->grad_fn()->apply(grad_output);
            }
        }
    }

    std::string AddFunction::name() const {
        return "Add";
    }

    //----------------Sub---------------------------
    void SubFunction::apply(const af::array& grad_output) {
        this->mark_visited();

        if (inputs[0]->requires_grad()) {
            inputs[0]->grad() += grad_output;

            if (inputs[0]->grad_fn()) {
                inputs[0]->grad_fn()->apply(grad_output);
            }
        }

        if (inputs[1]->requires_grad()) {
            inputs[1]->grad() += -grad_output;

            if (inputs[1]->grad_fn()) {
                inputs[1]->grad_fn()->apply(-grad_output);
            }
        }
    }

    std::string SubFunction::name() const {
        return "Sub";
    }

    //----------------Mul---------------------------
    void MulFunction::apply(const af::array& grad_output) {
        this->mark_visited();
        // for z = a * b, ∂z/∂a = b, ∂z/∂b = a
        auto a = inputs[0]->data();
        auto b = inputs[1]->data();

        // ∂L/∂a = grad_out * b
        if (inputs[0]->requires_grad()) {
            af::array grad_a = grad_output * b;
            inputs[0]->grad() += grad_a;
            if (inputs[0]->grad_fn())
                inputs[0]->grad_fn()->apply(grad_a);
        }

        if (inputs[1]->requires_grad()) {
            // ∂L/∂b = grad_out * a
            af::array grad_b = grad_output * a;
            inputs[1]->grad() += grad_b;
            if (inputs[1]->grad_fn())
                inputs[1]->grad_fn()->apply(grad_b);
        }
    }

    std::string MulFunction::name() const {
        return "Mul";
    }

    //----------------Div---------------------------

    void DivFunction::apply(const af::array& grad_output) {
        this->mark_visited();

        const af::array& a = inputs[0]->data();  // numerator
        const af::array& b = inputs[1]->data();  // denominator

        if (inputs[0]->requires_grad()) {
            af::array grad_a = grad_output / b;  // ∂(a / b) / ∂a = 1 / b
            inputs[0]->grad() += grad_a;

            if (inputs[0]->grad_fn()) {
                inputs[0]->grad_fn()->apply(grad_a);
            }
        }

        if (inputs[1]->requires_grad()) {
            af::array grad_b = -grad_output * a / (b * b);  // ∂(a / b) / ∂b = -a / b²
            inputs[1]->grad() += grad_b;

            if (inputs[1]->grad_fn()) {
                inputs[1]->grad_fn()->apply(grad_b);
            }
        }
    }

    std::string DivFunction::name() const {
        return "Div";
    }

    //----------------Clone---------------------------
    void CloneFunction::apply(const af::array &grad_output) {
        this->mark_visited();
        inputs[0]->grad() = grad_output.copy();
    }

    std::string CloneFunction::name() const {
        return "Clone";
    }

    //----------------Matmul---------------------------
    void MatMulFunction::apply(const af::array& grad_output) {
        this->mark_visited();
        // inputs[0] = a, inputs[1] = b
        const af::array& a = inputs[0]->data();   // shape: (M × K)
        const af::array& b = inputs[1]->data();   // shape: (K × N)

        // ∂L/∂a = grad_output @ bᵀ  ==> shape: (M × N) @ (N × K) = (M × K)
        if (inputs[0]->requires_grad()) {
            af::array grad_a = af::matmul(grad_output, af::transpose(b));
            inputs[0]->grad() += grad_a;
            if (inputs[0]->grad_fn())
                inputs[0]->grad_fn()->apply(grad_a);
        }

        // ∂L/∂b = aᵀ @ grad_output  ==> shape: (K × M) @ (M × N) = (K × N)
        if (inputs[1]->requires_grad()) {
            af::array grad_b = af::matmul(af::transpose(a), grad_output);
            inputs[1]->grad() += grad_b;
            if (inputs[1]->grad_fn())
                inputs[1]->grad_fn()->apply(grad_b);
        }
    }

    std::string MatMulFunction::name() const {
        return "MatMul";
    }

    //----------------Neg---------------------------
    void NegFunction::apply(const af::array& grad_output) {
        this->mark_visited();

        if (inputs[0]->requires_grad()) {
            inputs[0]->grad() += -grad_output;

            if (inputs[0]->grad_fn()) {
                inputs[0]->grad_fn()->apply(-grad_output);
            }
        }
    }

    std::string NegFunction::name() const {
        return "Neg";
    }

    //----------------Exp---------------------------
    void ExpFunction::apply(const af::array& grad_output) {
        this->mark_visited();

        const af::array& a = inputs[0]->data();
        af::array exp_a = af::exp(a);

        if (inputs[0]->requires_grad()) {
            af::array grad_input = exp_a * grad_output;
            inputs[0]->grad() += grad_input;

            if (inputs[0]->grad_fn()) {
                inputs[0]->grad_fn()->apply(grad_input);
            }
        }
    }

    std::string ExpFunction::name() const {
        return "Exp";
    }

    //----------------Log---------------------------
    void LogFunction::apply(const af::array& grad_output) {
        this->mark_visited();

        const af::array& a = inputs[0]->data();

        if (inputs[0]->requires_grad()) {
            af::array grad_input = grad_output / a;
            inputs[0]->grad() += grad_input;

            if (inputs[0]->grad_fn()) {
                inputs[0]->grad_fn()->apply(grad_input);
            }
        }
    }

    std::string LogFunction::name() const {
        return "Log";
    }

    //----------------Pow---------------------------
    void PowFunction::apply(const af::array& grad_output) {
        this->mark_visited();

        const af::array& base = inputs[0]->data();
        const af::array& exponent = inputs[1]->data();
        af::array output = af::pow(base, exponent);

        if (inputs[0]->requires_grad()) {
            af::array grad_base = exponent * af::pow(base, exponent - 1) * grad_output;
            inputs[0]->grad() += grad_base;

            if (inputs[0]->grad_fn()) {
                inputs[0]->grad_fn()->apply(grad_base);
            }
        }

        if (inputs[1]->requires_grad()) {
            af::array grad_exp = output * af::log(base) * grad_output;
            inputs[1]->grad() += grad_exp;

            if (inputs[1]->grad_fn()) {
                inputs[1]->grad_fn()->apply(grad_exp);
            }
        }
    }

    std::string PowFunction::name() const {
        return "Pow";
    }

    //----------------Sum---------------------------

    SumFunction::SumFunction(const af::dim4& input_shape, int dim, bool keepdim)
    : input_shape_(input_shape), dim_(dim), keepdim_(keepdim) {}

    void SumFunction::apply(const af::array& grad_output) {
        this->mark_visited();

        const auto& input = inputs[0];
        const af::array& input_data = input->data();

        if (!input->requires_grad()) return;

        af::array grad_input;

        if (dim_ == -1) {
            // Gradient of sum over all elements: fill with ones and multiply by grad_output scalar
            grad_input = af::constant(1.0f, input_shape_) * grad_output;
        } else {
            // Sum over specific dim
            // If keepdim == false, we must expand grad_output shape before broadcasting
            af::array grad = grad_output;
            if (!keepdim_) {
                // Insert singleton dimension back for broadcasting
                std::vector<dim_t> dims = {
                    input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3]
                };
                dims[dim_] = 1;  // insert singleton
                grad = af::moddims(grad_output, af::dim4(dims[0], dims[1], dims[2], dims[3]));
            }

            // Broadcast grad to match input shape
            grad_input = af::tile(grad, get_tile_repeats(input_shape_, grad.dims()));
        }

        input->grad() += grad_input;

        if (input->grad_fn()) {
            input->grad_fn()->apply(grad_input);
        }
    }

    std::string SumFunction::name() const {
        return "Sum";
    }

    //----------------Mean---------------------------
    MeanFunction::MeanFunction(const af::dim4& input_shape, int dim, bool keepdim)
        : input_shape_(input_shape), dim_(dim), keepdim_(keepdim) {}


    void MeanFunction::apply(const af::array& grad_output) {
        this->mark_visited();

        const auto& input = inputs[0];
        const af::array& input_data = input->data();

        if (!input->requires_grad()) return;

        af::array grad_input;

        if (dim_ == -1) {
            // Mean over all elements: gradient is 1/N broadcasted to input shape
            dim_t N = input_shape_.elements();
            grad_input = af::constant(1.0f / static_cast<float>(N), input_shape_) * grad_output;
        } else {
            // Mean over specific dim
            dim_t N = input_shape_[dim_];
            af::array grad = grad_output;

            if (!keepdim_) {
                // Insert singleton dimension back for broadcasting
                std::vector<dim_t> dims = {
                    input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3]
                };
                dims[dim_] = 1;
                grad = af::moddims(grad_output, af::dim4(dims[0], dims[1], dims[2], dims[3]));
            }

            // Scale the gradient by 1/N
            grad = grad / static_cast<float>(N);

            // Broadcast grad to match input shape
            grad_input = af::tile(grad, get_tile_dims(input_shape_, dim_));
        }

        input->grad() += grad_input;

        if (input->grad_fn()) {
            input->grad_fn()->apply(grad_input);
        }
    }

    std::string MeanFunction::name() const {
        return "Mean";
    }

    //----------------Max---------------------------

    MaxFunction::MaxFunction(const af::array& input_data, int dim, bool keepdim)
    : input_data_(input_data), dim_(dim), keepdim_(keepdim), input_shape_(input_data.dims()) {}


    void MaxFunction::apply(const af::array& grad_output) {
        this->mark_visited();

        const auto& input = inputs[0];
        if (!input->requires_grad()) return;

        af::array grad_mask;
        if (dim_ == -1) {
            // Global max: compare with scalar
            float max_val = af::max<float>(af::flat(input_data_));
            grad_mask = (input_data_ == max_val);
        } else {
            // Max along dimension: build broadcasted mask
            af::array max_vals = af::max(input_data_, dim_);

            if (!keepdim_) {
                // Insert singleton for broadcasting
                std::vector<dim_t> dims = {
                    input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3]
                };
                dims[dim_] = 1;
                max_vals = af::moddims(max_vals, af::dim4(dims[0], dims[1], dims[2], dims[3]));
            }

            grad_mask = (input_data_ == af::tile(max_vals, get_tile_dims(input_shape_, dim_)));
        }

        af::array grad = grad_output;

        if (!keepdim_ && dim_ != -1) {
            // Insert singleton for broadcasting
            std::vector<dim_t> dims = {
                input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3]
            };
            dims[dim_] = 1;
            grad = af::moddims(grad_output, af::dim4(dims[0], dims[1], dims[2], dims[3]));
        }

        // Broadcast grad to input shape
        if (dim_ != -1) {
            grad = af::tile(grad, get_tile_dims(input_shape_, dim_));
        }

        // Apply mask: gradient only to positions that had the max value
        af::array grad_input = grad * grad_mask.as(f32);

        input->grad() += grad_input;

        if (input->grad_fn()) {
            input->grad_fn()->apply(grad_input);
        }
    }

    std::string MaxFunction::name() const {
        return "Max";
    }

}