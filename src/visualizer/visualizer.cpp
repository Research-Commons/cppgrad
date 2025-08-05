#include "visualizer/visualizer.hpp"
#include "tensor/tensor.hpp"
#include "autograd/function.hpp"

#include <fstream>
#include <unordered_set>
#include <queue>
#include <cstdlib>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <iostream>

#ifdef _WIN32
    #include <windows.h>
#elif __APPLE__
    #include <TargetConditionals.h>
#endif

namespace cppgrad {
    std::string Visualizer::export_graphviz(std::shared_ptr<TensorImpl> root) {
        std::ostringstream out;
        out << "digraph ComputationGraph {\n";
        out << "  rankdir=LR;\n";

        std::unordered_set<TensorImpl*> visited;
        std::queue<std::shared_ptr<TensorImpl>> queue;

        queue.push(root);

        while (!queue.empty()) {
            auto current = queue.front();
            queue.pop();

            if (!current || visited.contains(current.get())) continue;
            visited.insert(current.get());

            std::string tensor_id = "tensor_" + std::to_string(reinterpret_cast<uintptr_t>(current.get()));
            std::ostringstream label;
            label << "Tensor\\nshape=";

            auto dims = current->data().dims();
            for (int i = 0; i < 4 && dims[i] > 1; ++i)
                label << dims[i] << ((i < 3 && dims[i + 1] > 1) ? "x" : "");

            // Show tensor values (if small enough)
            if (current->data().elements() <= 4) {
                label << "\\nval=";
                for (int i = 0; i < current->data().elements(); ++i) {
                    // Convert to scalar float or double
                    float value = current->data()(i).scalar<float>();
                    label << value;
                    if (i != current->data().elements() - 1) label << ",";
                }
            }

            std::string color = current->requires_grad() ? "red" : "black";
            out << "  " << tensor_id << " [label=\"" << label.str() << "\", shape=box, color=" << color << "];\n";

            if (current->grad_fn()) {
                auto fn = current->grad_fn();
                std::string fn_id = "fn_" + std::to_string(reinterpret_cast<uintptr_t>(fn.get()));

                std::string fillcolor = "lightgray";
                if (fn->is_visited())  // You must set this flag during backprop
                    fillcolor = "orange";

                out << "  " << fn_id << " [label=\"" << fn->name()
                    << "\", shape=ellipse, style=filled, fillcolor=" << fillcolor << "];\n";
                out << "  " << fn_id << " -> " << tensor_id << ";\n";

                for (auto& input : fn->inputs) {
                    if (!input) continue;

                    std::string input_id = "tensor_" + std::to_string(reinterpret_cast<uintptr_t>(input.get()));
                    std::ostringstream input_label;
                    input_label << "Tensor";

                    std::string input_color = input->requires_grad() ? "red" : "black";

                    out << "  " << input_id << " [label=\"" << input_label.str() << "\", shape=box, color=" << input_color << "];\n";
                    out << "  " << input_id << " -> " << fn_id << ";\n";

                    queue.push(input);
                }
            }
        }

        out << "}\n";
        return out.str();
    }

    void Visualizer::save_dot(const Tensor& output, const std::string& base_filename) {
        // Generate timestamp in format YYYYMMDD_HHMM
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::tm local_tm{};
#ifdef _WIN32
        localtime_s(&local_tm, &now_time);
#else
        localtime_r(&now_time, &local_tm);
#endif

        std::ostringstream timestamp_stream;
        timestamp_stream << std::put_time(&local_tm, "%Y%m%d_%H%M");
        std::string timestamp = timestamp_stream.str();

        // Build filenames
        const std::string directory = "examples/resources/";
        std::string full_base = base_filename + "_" + timestamp;
        std::string dot_file = directory + full_base + ".dot";
        std::string png_file = directory + full_base + ".png";

        // Print save location
        std::cout << "VISUALIZER : Saving computation graph to: Build_loc/" << png_file << std::endl;

        // Ensure directory exists
        std::filesystem::create_directories(directory);  // cross-platform

        // Write the DOT content
        std::ofstream file(dot_file);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + dot_file);
        }

        file << export_graphviz(output.impl());
        file.close();

        // Convert to PNG using dot
        std::string cmd = "dot -Tpng " + dot_file + " -o " + png_file;
        if (std::system(cmd.c_str()) != 0) {
            throw std::runtime_error("Graphviz 'dot' command failed. Is Graphviz installed?");
        }

        // Open image with default viewer
#ifdef _WIN32
        std::string open_cmd = "start \"\" \"" + png_file + "\"";
#elif __APPLE__
        std::string open_cmd = "open \"" + png_file + "\"";
#else
        std::string open_cmd = "xdg-open \"" + png_file + "\" &";
#endif
        std::system(open_cmd.c_str());
    }

}
