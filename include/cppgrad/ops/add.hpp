#pragma once
#include <vector>
#include <stdexcept>

namespace cppgrad {

    class Tensor;

    // Tensor operator+(const Tensor& a, const Tensor& b);
    // Tensor operator+(const Tensor& lhs, float scalar);
    // Tensor operator+(float scalar, const Tensor& rhs);

    // Helper: compute broadcasted output shape for two input shapes
    static std::vector<size_t> computeBroadcastShape(const std::vector<size_t>& a, const std::vector<size_t>& b) {
        size_t na = a.size(), nb = b.size();
        size_t n = std::max(na, nb);
        std::vector<size_t> out(n);
        for (size_t i = 0; i < n; ++i) {
            // Align from the end; if one shape is shorter, treat missing dims as 1
            size_t dimA = (i < n-na ? 1 : a[i-(n-na)]);
            size_t dimB = (i < n-nb ? 1 : b[i-(n-nb)]);
            if (dimA != dimB && dimA != 1 && dimB != 1) {
                throw std::runtime_error("Shapes not broadcastable");
            }
            out[i] = (dimA > dimB ? dimA : dimB);
        }
        return out;
    }

}