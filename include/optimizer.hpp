// #pragma once
// #include "tensor/tensor.hpp"
// #include <vector>
//
// namespace cppgrad {
//
// class Optimizer {
// public:
//     virtual void step() = 0;
//     virtual void zero_grad() = 0;
// };
//
// class SGD : public Optimizer {
//     std::vector<TensorRef> params;
//     float lr;
// public:
//     SGD(std::vector<TensorRef> parameters, float lr);
//     void step() override;
//     void zero_grad() override;
// };
//
// }
