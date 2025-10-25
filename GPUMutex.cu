//
// Created by ASUS on 10/18/2025.
//

#ifndef SRC_GPUMUTEX_CUH
#define SRC_GPUMUTEX_CUH

#include <cuda_runtime.h>
#include <device_atomic_functions.h>

class GPUMutex {
private:
    int* lock_state;  // 0=未锁, 1=已锁
public:
    GPUMutex() {lock_state = 0;}

    __device__ void init(int* state) {
        lock_state = state;
    }

    __device__ void lock() {
        while (atomicCAS(lock_state, 0, 1) != 0) {
            // 自旋等待，但加个限制避免死锁
            for (int i = 0; i < 256; i++) {
                if (atomicCAS(lock_state, 0, 1) == 0) return;
            }
            // 100次失败后放弃这次操作
            return;
        }
    }

    __device__ void unlock() {
        atomicExch(lock_state, 0);
    }

    __device__ bool try_lock() {
        return atomicCAS(lock_state, 0, 1) == 0;
    }
};

class GPUMutexGuard {
private:
    GPUMutex* mutex;
    bool owns_lock;
public:
    GPUMutexGuard() = delete;
    GPUMutexGuard(const GPUMutexGuard& mutex_guard) {
        mutex = mutex_guard.mutex;
        owns_lock = mutex->try_lock();
    };
    GPUMutexGuard(GPUMutex* mutex_original) {
        mutex = mutex_original;
        mutex->lock();
        owns_lock = true;
    }
    ~GPUMutexGuard() {
        if (owns_lock) {
            mutex->unlock();
        }
    }
    void unlock() {
        if (owns_lock) {
            mutex->unlock();
            owns_lock = false;
        }
    }
    bool try_lock() {
        if (!owns_lock) {
            owns_lock = mutex->try_lock();
        }
        return owns_lock;
    }
    void lock() {
        if (!owns_lock) {
            mutex->lock();
            owns_lock = true;
        }
    }
};

#endif //SRC_GPUMUTEX_CUH