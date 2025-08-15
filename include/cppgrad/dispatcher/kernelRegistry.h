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
    // forward kernel signature (same as before)
    using KernelFunc = std::function<void(const Tensor&, const Tensor&, Tensor&)>;

    // backward kernel signature (same as before)
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
        forward_registry_[std::make_pair(op, dev)] = std::move(fn);
    }

    // Retrieve a forward kernel; first try (op,dev), then fall back to (op, CPU).
    // Throws only if no kernel at all is registered for this op.
    KernelFunc getKernel(OpType op, DeviceType dev) {
        auto key = std::make_pair(op, dev);
        auto it = forward_registry_.find(key);
        if (it != forward_registry_.end()) return it->second;

        // fallback to CPU
        auto cpu_key = std::make_pair(op, DeviceType::CPU);
        auto it_cpu = forward_registry_.find(cpu_key);
        if (it_cpu != forward_registry_.end()) return it_cpu->second;

        throw std::runtime_error("No forward kernel registered for this op on device and no CPU fallback");
    }

    // Register a backward kernel
    void registerBackwardKernel(OpType op, DeviceType dev, BackwardKernelFunc fn) {
        backward_registry_[std::make_pair(op, dev)] = std::move(fn);
    }

    // Retrieve a backward kernel; try (op,dev) then fall back to (op, CPU).
    // Returns nullptr only if neither device nor CPU kernel exists.
    BackwardKernelFunc getBackwardKernelOrNull(OpType op, DeviceType dev) {
        auto key = std::make_pair(op, dev);
        auto it = backward_registry_.find(key);
        if (it != backward_registry_.end()) return it->second;

        // fallback to CPU
        auto cpu_key = std::make_pair(op, DeviceType::CPU);
        auto it_cpu = backward_registry_.find(cpu_key);
        if (it_cpu != backward_registry_.end()) return it_cpu->second;

        // no backward kernel anywhere
        return nullptr;
    }

private:
    KernelRegistry() = default;
    std::map<std::pair<OpType,DeviceType>, KernelFunc> forward_registry_;
    std::map<std::pair<OpType,DeviceType>, BackwardKernelFunc> backward_registry_;
};

} // namespace cppgrad