#pragma once
#include "tensor/tensor.hpp"
#include <vector>

namespace cppgrad {

class Module {
public:
    virtual TensorRef forward(const TensorRef& input) = 0;
    virtual std::vector<TensorRef> parameters() = 0;
};

class Linear : public Module {
    TensorRef W, b;
public:
    Linear(int in_features, int out_features);
    TensorRef forward(const TensorRef& input) override;
    std::vector<TensorRef> parameters() override;
};

class Sequential : public Module {
    std::vector<std::shared_ptr<Module>> layers;
public:
    Sequential(std::initializer_list<std::shared_ptr<Module>> l);
    TensorRef forward(const TensorRef& input) override;
    std::vector<TensorRef> parameters() override;
};

}
