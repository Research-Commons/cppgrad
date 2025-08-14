#pragma once
#include <functional>
#include <map>
#include <stdexcept>
#include <utility>

#include "cppgrad/tensor/tensor.hpp"
#include "cppgrad/enums/dispatcherEnum.h"

namespace cppgrad {

class KernelRegistry {
public:
    // existing forward kernel signature
    using KernelFunc = std::function<void(const Tensor&, const Tensor&, Tensor&)>;

    // backward kernel signature:
    // grad_out is broadcasted-output-shaped Tensor; grad_a/grad_b are allocated output-shaped Tensors for input grads.
    using BackwardKernelFunc = std::function<void(const Tensor& a,
                                                  const Tensor& b,
                                                  const Tensor& grad_out,
                                                  Tensor& grad_a,
                                                  Tensor& grad_b)>;

    static KernelRegistry& instance() {
        static KernelRegistry inst;
        return inst;
    }

    // Register a forward kernel for an operation and device
    void registerKernel(OpType op, DeviceType dev, KernelFunc fn) {
        forward_registry_[std::make_pair(op, dev)] = fn;
    }

    // Retrieve a forward kernel; throws if not found
    KernelFunc getKernel(OpType op, DeviceType dev) {
        auto it = forward_registry_.find(std::make_pair(op, dev));
        if (it == forward_registry_.end()) {
            throw std::runtime_error("No forward kernel registered for this op/device");
        }
        return it->second;
    }

    // Register a backward kernel
    void registerBackwardKernel(OpType op, DeviceType dev, BackwardKernelFunc fn) {
        backward_registry_[std::make_pair(op, dev)] = fn;
    }

    // Retrieve a backward kernel; returns nullptr when not found (so caller can fallback)
    BackwardKernelFunc getBackwardKernelOrNull(OpType op, DeviceType dev) {
        auto it = backward_registry_.find(std::make_pair(op, dev));
        if (it == backward_registry_.end()) return nullptr;
        return it->second;
    }

private:
    KernelRegistry() = default;
    std::map<std::pair<OpType,DeviceType>, KernelFunc> forward_registry_;
    std::map<std::pair<OpType,DeviceType>, BackwardKernelFunc> backward_registry_;
};

} // namespace cppgrad