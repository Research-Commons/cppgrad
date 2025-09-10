#pragma once

enum class DeviceType { CPU , CUDA};
enum class OpType    { Add, Mul , Sub, Div};

//CPU instruction-set tiers
enum class CpuIsa { GENERIC, AVX2, AVX512 };

struct DispatchKey {
    OpType op;
    DeviceType dev;
    CpuIsa isa;

    // for std::map
    bool operator<(const DispatchKey& o) const {
        if (op != o.op) return op < o.op;
        if (dev != o.dev) return dev < o.dev;
        return isa < o.isa;
    }
};