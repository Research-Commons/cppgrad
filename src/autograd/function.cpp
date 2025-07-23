#include "autograd/function.hpp"
#include "tensor/tensorimpl.hpp"

namespace cppgrad {

    void AddFunction::apply(const af::array &grad_output) {
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


    void MulFunction::apply(const af::array& grad_output) {
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

    void CloneFunction::apply(const af::array &grad_output) {
        inputs[0]->grad() = grad_output.copy();
    }


}