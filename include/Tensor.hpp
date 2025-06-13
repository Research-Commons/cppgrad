#pragma once
#include <vector>
#include <memory>
#include <string>
#include <optional>

namespace cppgrad {

class Tensor;

using TensorRef = std::shared_ptr<Tensor>;

class Tensor {
public:
    std::vector<int> shape;
    std::vector<int> strides;
    std::vector<float> data;
    std::optional<TensorRef> grad;
    bool requires_grad = false;
    std::string op_type;
    std::vector<TensorRef> parents;

    Tensor(std::vector<int> shape, bool requires_grad = false);

    float& operator[](int index);
    const float& operator[](int index) const;

    int numel() const;
    void zero_grad();
    TensorRef clone() const;

    void backward(); // call on scalar
    void backward(float initial_grad);

    static TensorRef create(std::vector<int> shape, bool requires_grad = false);
};

}  // namespace cppgrad
