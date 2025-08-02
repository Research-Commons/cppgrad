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
        virtual std::string name() const = 0;

        void mark_visited() { visited_ = true; }
        bool is_visited() const { return visited_; }

    private:
        bool visited_ = false;
    };

    class AddFunction : public Function {
        void apply(const af::array& grad_output) override;
        std::string name() const override;
    };

    class SubFunction : public Function {
        void apply(const af::array& grad_output) override;
        std::string name() const override;
    };

    class MulFunction : public Function {
        void apply(const af::array& grad_output) override;
        std::string name() const override;
    };

    class DivFunction : public Function {
        void apply(const af::array& grad_output) override;
        std::string name() const override;
    };

    class CloneFunction : public Function {
        void apply(const af::array& grad_output) override;
        std::string name() const override;
    };

    class MatMulFunction : public Function {
        void apply(const af::array& grad_output) override;
        std::string name() const override;
    };

    class NegFunction : public Function {
        void apply(const af::array& grad_output) override;
        std::string name() const override;
    };

    class ExpFunction : public Function {
        void apply(const af::array& grad_output) override;
        std::string name() const override;
    };

    class LogFunction : public Function {
        void apply(const af::array& grad_output) override;
        std::string name() const override;
    };

    class PowFunction : public Function {
        void apply(const af::array& grad_output) override;
        std::string name() const override;
    };

    class SumFunction : public Function {
        void apply(const af::array& grad_output) override;
        std::string name() const override;

        public:
            SumFunction(const af::dim4& input_shape, int dim, bool keepdim);

        private:
            af::dim4 input_shape_;
            int dim_;
            bool keepdim_;

            af::dim4 get_tile_repeats(const af::dim4& target, const af::dim4& smaller) {
                return af::dim4(
                    target[0] / std::max((dim_t)1, smaller[0]),
                    target[1] / std::max((dim_t)1, smaller[1]),
                    target[2] / std::max((dim_t)1, smaller[2]),
                    target[3] / std::max((dim_t)1, smaller[3])
                );
            }
    };

    class MeanFunction : public Function {
        void apply(const af::array& grad_output) override;
        std::string name() const override;

        public:
            MeanFunction(const af::dim4& input_shape, int dim, bool keepdim);

        private:
            af::dim4 input_shape_;
            int dim_;
            bool keepdim_;

            af::dim4 get_tile_dims(const af::dim4& input_dims, int dim) const {
                af::dim4 tile_dims(1, 1, 1, 1);
                tile_dims[dim] = input_dims[dim];
                return tile_dims;
            }
    };

    class MaxFunction : public Function {
        void apply(const af::array& grad_output) override;
        std::string name() const override;

        public:
            MaxFunction(const af::array& input_data, int dim, bool keepdim);

        private:
            af::array input_data_;
            int dim_;
            bool keepdim_;
            af::dim4 input_shape_;

            af::dim4 get_tile_dims(const af::dim4& input_dims, int dim) const {
                af::dim4 tile_dims(1, 1, 1, 1);
                tile_dims[dim] = input_dims[dim];
                return tile_dims;
            }
    };

} // namespace cppgrad