#pragma once
#include <vector>
#include <arrayfire.h>
#include <memory>

namespace cppgrad {

    /**
     * @file function.hpp
     * @brief Defines the autograd Function hierarchy for backward computation.
     *
     * Each subclass of `Function` represents a specific operation in the computation graph
     * and implements how to backpropagate through it by overriding the `apply()` method.
     *
     * This design allows dynamic computation graph construction (define-by-run),
     * similar to PyTorch's autograd system. During the forward pass, a `Function`
     * instance is attached to the output tensor if `requires_grad=true`, and
     * during the backward pass, these `Function` nodes are traversed to compute gradients.
     *
     * The base `Function` class manages input tensor references and visitation state
     * (to avoid duplicate traversal). Derived classes implement custom gradient logic.
     *
     * Categories of supported operations:
     * - Elementwise operations: Add, Sub, Mul, Div
     * - Unary operations: Neg, Exp, Log, Pow, Clone
     * - Matrix operations: MatMul
     * - Reductions: Sum, Mean, Max
     *
     * Each `Function` subclass is expected to:
     *   - Store any info needed for backward computation (e.g., input shape, dim).
     *   - Implement `apply()` for computing gradients.
     *   - Provide a `name()` for graph visualization/debugging.
    */

    class TensorImpl;

    /// Base class for all backward functions in the autograd graph.
    /// Analogous to PyTorch's `Node` class.
    class Function {
    public:
        virtual ~Function() = default;

        /// Pointers to input tensors used in the forward pass.
        std::vector<std::shared_ptr<TensorImpl>> inputs;

        /// Compute gradient w.r.t. inputs, given gradient of the output.
        virtual void apply(const af::array& grad_output) = 0;

        /// Human-readable name of the function (used for graph display/debug).
        virtual std::string name() const = 0;

        /// Used during graph traversal to avoid revisiting the same node.
        void mark_visited() { visited_ = true; }
        bool is_visited() const { return visited_; }

    private:
        bool visited_ = false;
    };

    // --- Elementwise Operations ---

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

    // --- Unary Operations ---

    class CloneFunction : public Function {
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

    // --- Matrix Operations ---

    class MatMulFunction : public Function {
        void apply(const af::array& grad_output) override;
        std::string name() const override;
    };

    // --- Reduction Operations ---

    class SumFunction : public Function {
        void apply(const af::array& grad_output) override;
        std::string name() const override;

    public:
        SumFunction(const af::dim4& input_shape, int dim, bool keepdim);

    private:
        af::dim4 input_shape_;  // Shape of input before sum
        int dim_;               // Reduction dimension
        bool keepdim_;          // Whether output kept reduced dim

        /// Compute how many times to tile the reduced gradient
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
        af::dim4 input_shape_;  // Original input shape
        int dim_;               // Dimension reduced
        bool keepdim_;          // Whether reduced dim is kept

        /// Create tiling pattern to broadcast grad_output back to input shape
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
        af::array input_data_;  // Needed to identify max positions
        int dim_;               // Axis along which max was computed
        bool keepdim_;          // Whether reduced dim is kept
        af::dim4 input_shape_;  // Original shape of input

        /// Tiling pattern for broadcasting grad_output
        af::dim4 get_tile_dims(const af::dim4& input_dims, int dim) const {
            af::dim4 tile_dims(1, 1, 1, 1);
            tile_dims[dim] = input_dims[dim];
            return tile_dims;
        }
    };

} // namespace cppgrad
