//
// Created by ASUS on 9/29/2025.
//

#ifndef CUDA_DEVICE_QUEUE_CUH
#define CUDA_DEVICE_QUEUE_CUH

#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include <intrin0.inl.h>

#define ll long long
#define ull unsigned ll

ull atomic_add_ull_builtin(ull *ptr, ull value) {
#if defined(__GNUC__) || defined(__clang__)
    // GCC/Clang
    return __sync_fetch_and_add(ptr, value);
#elif defined(_MSC_VER)
    // MSVC
#ifdef _WIN64
    return _InterlockedExchangeAdd64(reinterpret_cast<__int64 *>(ptr), static_cast<__int64>(value));
#else
    // WIN32
    return _InterlockedExchangeAdd64((__int64 *) ptr, (__int64) value);
#endif
#else
#warning "Using fallback atomic implementation"
    return __sync_fetch_and_add(ptr, value);
#endif
}
#ifndef __CUDA_ARCH__
ull atomicAdd_(ull *ptr, ull value) {
    return atomic_add_ull_builtin(ptr, static_cast<__int64>(value));
}

#define atomicAdd atomicAdd_
#endif

template<typename T, int CAPACITY>
struct DeviceQueue {
    T data[CAPACITY];
    ull head;
    ull tail;

    __device__ void init() {
        head = 0ULL;
        tail = 0ULL;
    }

    __host__ void h_init() {
        head = 0ULL;
        tail = 0ULL;
    }

    __device__ bool push(const T &item) {
        const ull old_tail = atomicAdd(&tail, 1ULL);
        const ull current_head = atomicAdd((ull *) &head, 0ULL);
        if (old_tail - current_head >= CAPACITY) {
            atomicAdd(&tail, -1LL);
            return false;
        }
        int pos = static_cast<int>(old_tail % CAPACITY);
        data[pos] = item;

        return true;
    }

    __host__ bool h_push(const T &item) {
        if (tail - head >= CAPACITY) return false;

        int pos = (int) (tail % CAPACITY);

        data[pos] = item;

        tail++;

        return true;
    }

    __host__ bool h_pop(T &result) {
        if (tail <= head) return false;

        int pos = (int) (head % CAPACITY);

        result = data[pos];

        head++;

        return true;
    }

    __host__ bool h_front(T &result) const {
        if (tail <= head) return false;

        int pos = (int) (head % CAPACITY);

        result = data[pos];

        return true;
    }

    __host__ bool h_empty() const {
        return head >= tail;
    }

    __host__ int h_size() const {
        return (int) (tail - head);
    }

    __host__ bool h_full() const {
        return (tail - head) >= CAPACITY;
    }

    __device__ bool pop(T &result) {
        ull old_head = atomicAdd(&head, 1ULL);
        ull current_tail = atomicAdd(&tail, 0ULL);
        if (old_head >= current_tail) {
            atomicAdd(&head, -1ULL);
            return false;
        }
        int pos = static_cast<int>(old_head % CAPACITY);
        result = data[pos];

        return true;
    }

    [[nodiscard]] __device__ bool empty() const {
        ull current_head = atomicAdd(const_cast<ull *>(&head), 0ULL);
        ull current_tail = atomicAdd(const_cast<ull *>(&tail), 0ULL);
        return current_head >= current_tail;
    }

    [[nodiscard]] __device__ bool full() const {
        ull current_head = atomicAdd(const_cast<ull *>(&head), 0ULL);
        ull current_tail = atomicAdd(const_cast<ull *>(&tail), 0ULL);
        return (current_tail - current_head) >= CAPACITY;
    }

    [[nodiscard]] __device__ int size() const {
        ull current_head = atomicAdd(const_cast<ull *>(&head), 0ULL);
        ull current_tail = atomicAdd(const_cast<ull *>(&tail), 0ULL);
        return (int) (current_tail - current_head);
    }

    // 使用环形缓冲区索引获取队首元素但不移除
    [[nodiscard]] __device__ T front() const {
        // 检查队列是否为空
        ull current_head = atomicAdd(const_cast<ull *>(&head), 0ULL);
        ull current_tail = atomicAdd(const_cast<ull *>(&tail), 0ULL);
        if (current_head >= current_tail) {
            // 队列空，返回默认构造的T类型对象
            return T{};
        }

        // 使用环形缓冲区索引获取队首元素但不移除
        int pos = static_cast<int>(current_head % CAPACITY);
        return data[pos];
    }
};
#endif // CUDA_DEVICE_QUEUE_CUH
