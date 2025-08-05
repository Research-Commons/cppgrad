#pragma once

#include <sstream>
#include <memory>

namespace cppgrad {

    /**
     * @file visualizer.hpp
     * @brief Autograd graph visualization utilities for cppgrad.
     *
     * The `Visualizer` class provides tooling to export the autograd graph to a
     * [Graphviz DOT](https://graphviz.org/doc/info/lang.html) file for debugging
     * and understanding gradient flows through tensor operations.
     *
     * Key Features:
     * - `save_dot()` saves the graph of the computation that produced a given `Tensor`.
     * - Internal traversal starts from the output Tensor and walks backward via `grad_fn`.
     * - Nodes in the graph represent `Function` objects (operations), and edges represent tensor data flow.
     *
     * Typical Usage:
     * ```cpp
     * Tensor a = Tensor::ones({2, 2}, true);
     * Tensor b = Tensor::ones({2, 2}, true);
     * Tensor c = a + b;
     * c.backward();
     * Visualizer::save_dot(c, "graph");  // Outputs graph.dot file
     * ```
     *
     * Design Notes:
     * - All logic is static; no instance of `Visualizer` is required.
     * - Internally uses `TensorImpl` to access graph structure.
     *
     * This tool is useful for inspecting and debugging autograd internals,
     * especially when developing new `Function` nodes or diagnosing gradient issues.
    */

    class TensorImpl;
    class Tensor;

    class Visualizer {
        public:
            static void save_dot(const Tensor& output, const std::string& base_filename);

        private:
            static std::string export_graphviz(std::shared_ptr<TensorImpl> root);
    };

}