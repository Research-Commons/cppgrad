#pragma once
#include <vector>
#include <arrayfire.h>

namespace cppgrad {

    class Tensor;

    //Like a node class of pytorch
    class Function {
    public:
        virtual ~Function() = default;

        std::vector<std::shared_ptr<TensorImpl>> inputs;

        virtual void apply(const af::array& grad_output) = 0;
    };


    class AddFunction : public Function {
        void apply(const af::array& grad_output) override {

            if (inputs[0]->requires_grad_) {
                inputs[0]->grad_ += grad_output;

                if (inputs[0]->grad_fn) {
                    inputs[0]->grad_fn->apply(grad_output);
                }
            }

            if (inputs[1]->requires_grad_) {
                inputs[1]->grad_ += grad_output;

                if (inputs[1]->grad_fn) {
                    inputs[1]->grad_fn->apply(grad_output);
                }
            }
        }
    };

    class MulFunction : public Function {
        void apply(const af::array& grad_output) override {
            // for z = a * b, ∂z/∂a = b, ∂z/∂b = a
            auto a = inputs[0]->data_;
            auto b = inputs[1]->data_;

            // ∂L/∂a = grad_out * b
            if (inputs[0]->requires_grad_) {
                af::array grad_a = grad_output * b;
                inputs[0]->grad_ += grad_a;
                if (inputs[0]->grad_fn)
                    inputs[0]->grad_fn->apply(grad_a);
            }

            if (inputs[1]->requires_grad_) {
                // ∂L/∂b = grad_out * a
                af::array grad_b = grad_output * a;
                inputs[1]->grad_ += grad_b;
                if (inputs[1]->grad_fn)
                    inputs[1]->grad_fn->apply(grad_b);
            }
        }
    };

} // namespace cppgrad