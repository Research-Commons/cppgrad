#pragma once
#include <vector>
#include <cstddef>
#include <stdexcept>

namespace cppgrad {

    /**
     * Pad a shape on the left with 1s to length n.
     * Example: pad_shape_right([2,3], 4) => [1,1,2,3]
     */
    inline std::vector<size_t> pad_shape_right(const std::vector<size_t>& shape, int n) {
        if (n < 0) throw std::runtime_error("pad_shape_right: negative n");
        std::vector<size_t> res((size_t)n, 1);
        int offset = n - static_cast<int>(shape.size());
        for (size_t i = 0; i < shape.size(); ++i) res[offset + i] = shape[i];
        return res;
    }

    /**
     * Compute C-order strides for a shape.
     * Example: shape [2,3,4] => strides [12,4,1]
     */
    inline std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
        int n = static_cast<int>(shape.size());
        std::vector<size_t> stride((size_t)n, 0);
        if (n == 0) return stride;
        stride[(size_t)n - 1] = 1;
        for (int i = n - 2; i >= 0; --i) {
            stride[(size_t)i] = stride[(size_t)i + 1] * shape[(size_t)i + 1];
        }
        return stride;
    }

    /**
     * Compute the broadcasted shape of two shapes (right-aligned).
     * Example: [2,3] and [3] => [2,3]
     */
    inline std::vector<size_t> compute_broadcast_shape(const std::vector<size_t>& a_sh,
                                                       const std::vector<size_t>& b_sh) {
        int n = static_cast<int>(std::max(a_sh.size(), b_sh.size()));
        auto a_pad = pad_shape_right(a_sh, n);
        auto b_pad = pad_shape_right(b_sh, n);
        std::vector<size_t> out_sh((size_t)n);
        for (int i = 0; i < n; ++i) out_sh[(size_t)i] = std::max(a_pad[(size_t)i], b_pad[(size_t)i]);
        return out_sh;
    }

    /** product of dims */
    inline size_t numel(const std::vector<size_t>& shape) {
        size_t total = 1;
        for (auto s : shape) total *= s;
        return total;
    }

    /**
     * Squeeze a padded buffer (with padded_shape) back to the unpadded shape (unpadded_shape).
     * - padded must be in row-major (C-order) and length numel(padded_shape).
     * - unpadded_shape is the original shape before left-padding with ones.
     *
     * This collapses leading padded dims (which must be 1 in the original semantics).
     * If unpadded_shape is empty (scalar), the result is a single-element vector containing the sum.
     */
    inline std::vector<float> squeeze_padded_to_unpadded(const std::vector<float>& padded,
                                                         const std::vector<size_t>& padded_shape,
                                                         const std::vector<size_t>& unpadded_shape) {
        int n = static_cast<int>(padded_shape.size());
        if ((int)unpadded_shape.size() > n) throw std::runtime_error("squeeze: target rank > padded rank");

        size_t padded_total = numel(padded_shape);
        if (padded_total == 0) return {};

        if (unpadded_shape.empty()) {
            // scalar: sum all elements
            float acc = 0.0f;
            for (float v : padded) acc += v;
            return std::vector<float>{acc};
        }

        // compute offset = number of padded leading dims
        int offset = n - static_cast<int>(unpadded_shape.size());
        // stride for padded and for squeezed shapes
        auto stride_padded = compute_strides(padded_shape);
        auto stride_squeezed = compute_strides(unpadded_shape);

        size_t squeezed_total = numel(unpadded_shape);
        std::vector<float> squeezed(squeezed_total, 0.0f);

        // For each position in padded buffer, map to squeezed index and add
        for (size_t pos = 0; pos < padded_total; ++pos) {
            size_t tmp = pos;
            size_t squeezed_idx = 0;
            for (int d = 0; d < n; ++d) {
                size_t coord = tmp / stride_padded[(size_t)d];
                tmp = tmp % stride_padded[(size_t)d];
                if (d >= offset) {
                    int sd = d - offset;
                    squeezed_idx += coord * stride_squeezed[(size_t)sd];
                } else {
                    // leading padded dimension; ignored in squeezed mapping
                }
            }
            squeezed[squeezed_idx] += padded[pos];
        }
        return squeezed;
    }
} // namespace cppgrad