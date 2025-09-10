#pragma once

#include "cppgrad/enums/dispatcherEnum.h"

#include <string>

namespace cppgrad {
#if defined(_MSC_VER)
#include <immintrin.h>
#include <intrin.h>
#else
#include <cpuid.h>
#endif

    inline bool has_avx2() {
#if defined(_MSC_VER)
        int info[4];
        __cpuid(info, 0);
        if (info[0] < 7) return false;
        __cpuidex(info, 7, 0);
        return (info[1] & (1 << 5)) != 0; // EBX bit5 AVX2
#else
        unsigned int eax, ebx, ecx, edx;
        if (!__get_cpuid_max(0, 0) || __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 0) return false;
        return (ebx & (1u << 5)) != 0;
#endif
    }

    inline bool has_avx512f() {
#if defined(_MSC_VER)
        int info[4];
        __cpuid(info, 0);
        if (info[0] < 7) return false;
        __cpuidex(info, 7, 0);
        return (info[1] & (1 << 16)) != 0; // EBX bit16 AVX512F
#else
        unsigned int eax, ebx, ecx, edx;
        if (!__get_cpuid_max(0, 0) || __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 0) return false;
        return (ebx & (1u << 16)) != 0; // AVX-512F
#endif
    }

    inline CpuIsa detect_best_cpu_isa() {
        // Optional: allow override via env
        if (const char* env = std::getenv("CPPGRAD_CPU_ISA")) {
            if (std::string(env) == "avx512") return CpuIsa::AVX512;
            if (std::string(env) == "avx2")   return CpuIsa::AVX2;
            return CpuIsa::GENERIC;
        }
        if (has_avx512f()) return CpuIsa::AVX512;
        if (has_avx2())    return CpuIsa::AVX2;
        return CpuIsa::GENERIC;
    }
}