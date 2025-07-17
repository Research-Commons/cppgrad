#include "tensor.hpp"
#include <stdexcept>

namespace cppgrad {

    //if tensor has more than 4 dimensions, ignore the rest for now
    af::dim4 to_dim4(const std::vector<size_t>& shape) {
        af::dim4 dims(1, 1, 1, 1);
        for (size_t i = 0; i < std::min(static_cast<size_t>(4), shape.size()); ++i)
            dims[i] = shape[i];
        return dims;
    }

    Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<float>& values, bool requires_grad) {
        this->requires_grad_ = requires_grad;

        if (shape.empty()) {
            throw std::invalid_argument("Shape must not be empty");
        }

        af::dim4 dims = to_dim4(shape);

        size_t expected_numel = dims.elements();  // total elements
        if (values.size() != expected_numel) {
            throw std::invalid_argument("Number of values does not match shape");
        }

        data_ = af::array(static_cast<unsigned>(values.size()), values.data());
        data_ = af::moddims(data_, dims);
    }

    Tensor::Tensor(const af::array& arr, bool requires_grad) : data_(arr) {
        this->requires_grad_ = requires_grad;
    }

    Tensor Tensor::zeros(const std::vector<size_t>& shape, bool requires_grad) {
        af::dim4 dims = to_dim4(shape);
        return Tensor(af::constant(0.0f, dims), requires_grad);
    }

    Tensor Tensor::ones(const std::vector<size_t>& shape, bool requires_grad) {
        af::dim4 dims = to_dim4(shape);
        return Tensor(af::constant(1.0f, dims), requires_grad);
    }

    Tensor Tensor::randn(const std::vector<size_t>& shape, bool requires_grad) {
        af::dim4 dims = to_dim4(shape);
        return Tensor(af::randn(dims), requires_grad);
    }

    Tensor Tensor::full(const std::vector<size_t>& shape, float value, bool requires_grad) {
        af::dim4 dims = to_dim4(shape);
        return Tensor(af::constant(value, dims), requires_grad);
    }

    std::vector<size_t> Tensor::shape() const {
        af::dim4 dims = data_.dims();
        std::vector<size_t> out;
        for (int i = 0; i < 4 && dims[i] > 1; ++i)
            out.push_back(dims[i]);
        return out;
    }

    size_t Tensor::numel() const {
        return data_.elements();
    }

    size_t Tensor::ndim() const {
        return data_.numdims();
    }

    void Tensor::print() const {
        af_print(data_);
    }

} // namespace cppgrad