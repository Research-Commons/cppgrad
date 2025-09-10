#pragma once
#include <functional>
#include <map>
#include <stdexcept>
#include <utility>

#include "cppgrad/tensor/tensor.hpp"
#include "cppgrad/enums/dispatcherEnum.h"
#include "cppgrad/backend/isa/isaDetect.hpp"

namespace cppgrad {

class KernelRegistry {
public:
    using KernelFunc = std::function<void(const Tensor&, const Tensor&, Tensor&)>;
    using BackwardKernelFunc = std::function<void(
        const Tensor&, const Tensor&, const Tensor&, Tensor&, Tensor&)>;

    static KernelRegistry& instance() { static KernelRegistry inst; return inst; }

    void registerKernel(OpType op, DeviceType dev, CpuIsa isa, KernelFunc fn) {
        forward_[{op, dev, isa}] = std::move(fn);
    }
    void registerKernel(OpType op, DeviceType dev, KernelFunc fn) {
        registerKernel(op, dev, CpuIsa::GENERIC, std::move(fn));
    }

    void registerBackwardKernel(OpType op, DeviceType dev, CpuIsa isa, BackwardKernelFunc fn) {
        backward_[{op, dev, isa}] = std::move(fn);
    }
    void registerBackwardKernel(OpType op, DeviceType dev, BackwardKernelFunc fn) {
        registerBackwardKernel(op, dev, CpuIsa::GENERIC, std::move(fn));
    }

    // Try exact (op,dev,best_isa) then degrade to AVX2 then GENERIC
    KernelFunc getKernel(OpType op, DeviceType dev) {
        if (dev == DeviceType::CPU) {
            auto best = detect_best_cpu_isa();
            for (CpuIsa isa : {best, CpuIsa::AVX2, CpuIsa::GENERIC}) {
                auto it = forward_.find({op, dev, isa});
                if (it != forward_.end()) return it->second;
            }
        }
        // Non-CPU (e.g., CUDA) → exact, then CPU fallback (keeping your behavior)
        auto exact = forward_.find({op, dev, CpuIsa::GENERIC});
        if (exact != forward_.end()) return exact->second;
        auto cpu_fallback = forward_.find({op, DeviceType::CPU, CpuIsa::GENERIC});
        if (cpu_fallback != forward_.end()) return cpu_fallback->second;
        throw std::runtime_error("No forward kernel registered for this op/device");
    }

    BackwardKernelFunc getBackwardKernel(OpType op, DeviceType dev) {
        if (dev == DeviceType::CPU) {
            auto best = detect_best_cpu_isa();
            for (CpuIsa isa : {best, CpuIsa::AVX2, CpuIsa::GENERIC}) {
                auto it = backward_.find({op, dev, isa});
                if (it != backward_.end()) return it->second;
            }
        }
        auto exact = backward_.find({op, dev, CpuIsa::GENERIC});
        if (exact != backward_.end()) return exact->second;
        auto cpu_fallback = backward_.find({op, DeviceType::CPU, CpuIsa::GENERIC});
        if (cpu_fallback != backward_.end()) return cpu_fallback->second;
        throw std::runtime_error("No backward kernel registered for this op/device");
    }

private:
    KernelRegistry() = default;
    std::map<DispatchKey, KernelFunc> forward_;
    std::map<DispatchKey, BackwardKernelFunc> backward_;
};

} // namespace cppgrad