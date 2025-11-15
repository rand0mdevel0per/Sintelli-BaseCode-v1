/**
 * @file atomic_utils.cuh
 * @brief Unified atomic operations for CUDA and host code
 * 
 * Provides consistent atomic operations across different compilers
 * and platforms. CUDA device code uses built-in atomicAdd,
 * while host code uses compiler-specific intrinsics.
 */

#ifndef ATOMIC_UTILS_CUH
#define ATOMIC_UTILS_CUH

#include <cuda_runtime.h>

#ifdef _MSC_VER
#include <intrin0.inl.h>
#endif

#define ll long long
#define ull unsigned long long

// Host-side atomicAdd implementation (only when NOT in CUDA device code)
#ifndef __CUDA_ARCH__

// Helper function for host atomic add
inline ull atomic_add_ull_host(ull *ptr, ull value) {
#if defined(__GNUC__) || defined(__clang__)
    return __sync_fetch_and_add(ptr, value);
#elif defined(_MSC_VER)
    #ifdef _WIN64
    return _InterlockedExchangeAdd64(reinterpret_cast<__int64 *>(ptr), static_cast<__int64>(value));
    #else
    return _InterlockedExchangeAdd64((__int64 *) ptr, (__int64) value);
    #endif
#else
    #warning "Using fallback atomic implementation"
    return __sync_fetch_and_add(ptr, value);
#endif
}

// Host-side wrapper - only define if not in device code and if not already defined
#ifndef atomicAdd
inline ull atomicAdd(ull *ptr, ull value) {
    return atomic_add_ull_host(ptr, value);
}
#endif

#endif // __CUDA_ARCH__

// Device-side atomic subtract for unsigned long long
// This is safe to define as CUDA doesn't provide built-in atomicSub for ull
__device__ __forceinline__ unsigned long long atomicSubULL(unsigned long long* address, unsigned long long val) {
#ifdef __CUDA_ARCH__
    unsigned long long old = atomicCAS(address, 0ULL, 0ULL);
    unsigned long long assumed;
    do {
        assumed = old;
        unsigned long long newval = assumed - val;
        old = atomicCAS(address, assumed, newval);
    } while (old != assumed);
    return old;
#else
    return 0;
#endif
}

#endif // ATOMIC_UTILS_CUH
