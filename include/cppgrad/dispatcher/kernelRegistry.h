#pragma once

#include <functional>
#include <map>
#include <stdexcept>


#include "cppgrad/tensor/tensor.hpp"
#include "cppgrad/enums/dispatcherEnum.h"

namespace cppgrad {
    class KernelRegistry {
    public:
        using KernelFunc = std::function<void(const Tensor&, const Tensor&, Tensor&)>;

        static KernelRegistry& instance() {
            static KernelRegistry inst;
            return inst;
        }
        // Register a kernel for an operation and device
        void registerKernel(OpType op, DeviceType dev, KernelFunc fn) {
            registry_[std::make_pair(op, dev)] = fn;
        }
        // Retrieve the kernel; throw if not found
        KernelFunc getKernel(OpType op, DeviceType dev) {
            auto it = registry_.find(std::make_pair(op, dev));
            if (it == registry_.end()) {
                throw std::runtime_error("No kernel registered for this op/device");
            }
            return it->second;
        }
    private:
        KernelRegistry() = default;
        std::map<std::pair<OpType,DeviceType>, KernelFunc> registry_;
    };
}