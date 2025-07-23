#pragma once
#include <vector>
#include <arrayfire.h>
#include <memory>

namespace cppgrad {

    class TensorImpl;

    //Like a node class of pytorch
    class Function {
    public:
        virtual ~Function() = default;

        std::vector<std::shared_ptr<TensorImpl>> inputs;

        virtual void apply(const af::array& grad_output) = 0;
    };

    class AddFunction : public Function {
        void apply(const af::array& grad_output) override;
    };

    class MulFunction : public Function {
        void apply(const af::array& grad_output) override;
    };

    class CloneFunction : public Function {
        void apply(const af::array& grad_output) override;
    };

} // namespace cppgrad