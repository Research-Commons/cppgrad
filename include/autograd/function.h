#pragma once
#include <vector>
#include <arrayfire.h>

namespace cppgrad {

    class Tensor;

    //Like a node class of pytorch
    class Function {
    public:
        virtual ~Function() = default;

        std::vector<Tensor*> inputs;

        virtual void apply(const af::array& grad_output) = 0;
    };


    class AddFunction : public Function {
        void apply(const af::array& grad_output) override {
            inputs[0]->backward(grad_output);
            inputs[1]->backward(grad_output);
        }
    };

    class MulFunction : public Function {
        void apply(const af::array& grad_output) override {
            Tensor *A = inputs[0];
            Tensor *B = inputs[1];
            A->backward(grad_output * B->data());
            B->backward(grad_output * A->data());
        }
    };

} // namespace cppgrad