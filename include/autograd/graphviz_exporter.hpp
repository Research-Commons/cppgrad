#pragma once

#include <sstream>
#include <memory>

namespace cppgrad {

    class TensorImpl;
    class Tensor;

    class Visualizer {
        public:
            static void save_dot(const Tensor& output, const std::string& base_filename);

        private:
            static std::string export_graphviz(std::shared_ptr<TensorImpl> root);
    };

}