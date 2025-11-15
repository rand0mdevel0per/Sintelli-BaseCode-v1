/**
 * @file atomic_utils.cuh  
 * @brief Unified atomic operations for CUDA and host code
 * 
 * Provides atomic operations utilities. CUDA device code uses built-in
 * atomicAdd from device_atomic_functions.h. Host code can use helper functions.
 */

#ifndef ATOMIC_UTILS_CUH
#define ATOMIC_UTILS_CUH

#include <cuda_runtime.h>

#define ll long long
#define ull unsigned long long

// Device-side atomic subtract for unsigned long long
// Safe to define as CUDA doesn't provide built-in atomicSub for ull
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

// Host-side helper functions (use these explicitly when needed on host)
#ifndef __CUDA_ARCH__

#ifdef _MSC_VER
#include <intrin0.inl.h>
#endif

inline ull host_atomic_add_ull(ull *ptr, ull value) {
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

#endif // __CUDA_ARCH__

#endif // ATOMIC_UTILS_CUH
