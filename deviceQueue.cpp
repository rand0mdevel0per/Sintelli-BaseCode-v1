//
// Created by ASUS on 9/29/2025.
//

#ifndef CUDA_DEVICE_QUEUE_CUH
#define CUDA_DEVICE_QUEUE_CUH

#include <cuda_runtime.h>

template<typename T, int CAPACITY>
struct DeviceQueue {
    T data[CAPACITY];
    unsigned long long head;
    unsigned long long tail;

    __device__ void init() {
        head = 0ULL;
        tail = 0ULL;
    }

    __device__ bool push(const T &item) {
        // 原子地获取并递增tail
        unsigned long long old_tail = atomicAdd(&tail, 1ULL);

        // 检查队列是否已满
        unsigned long long current_head = atomicAdd(&head, 0ULL);
        if (old_tail - current_head >= CAPACITY) {
            // 队列满，回退tail
            atomicAdd(&tail, -1LL);
            return false;
        }

        // 使用环形缓冲区索引
        int pos = (int) (old_tail % CAPACITY);
        data[pos] = item;

        return true;
    }

    // 主机端版本的push方法，用于主机/设备数据传输

    __host__ bool host_push(const T &item) {
        if (tail - head >= CAPACITY) return false;

        int pos = (int) (tail % CAPACITY);

        data[pos] = item;

        tail++;

        return true;
    }


    // 主机端版本的pop方法

    __host__ bool host_pop(T &result) {
        if (tail <= head) return false;

        int pos = (int) (head % CAPACITY);

        result = data[pos];

        head++;

        return true;
    }


    // 主机端版本的front方法

    __host__ bool host_front(T &result) const {
        if (tail <= head) return false;

        int pos = (int) (head % CAPACITY);

        result = data[pos];

        return true;
    }


    // 主机端版本的empty方法

    __host__ bool host_empty() const {
        return head >= tail;
    }


    // 主机端版本的size方法

    __host__ int host_size() const {
        return (int) (tail - head);
    }


    // 主机端版本的full方法

    __host__ bool host_full() const {
        return (tail - head) >= CAPACITY;
    }


    // 主机端版本的init方法

    __host__ void host_init() {
        head = 0ULL;

        tail = 0ULL;
    }

    __device__ __host__ bool pop(T &result) {
        // 原子地获取并递增head
        unsigned long long old_head = atomicAdd(&head, 1ULL);

        // 检查队列是否为空
        unsigned long long current_tail = atomicAdd(&tail, 0ULL);
        if (old_head >= current_tail) {
            // 队列空，回退head
            atomicAdd(&head, -1LL);
            return false;
        }

        // 使用环形缓冲区索引
        int pos = (int) (old_head % CAPACITY);
        result = data[pos];

        return true;
    }

    [[nodiscard]] __device__ __host__ bool empty() const {
        unsigned long long current_head = atomicAdd((unsigned long long *) &head, 0ULL);
        unsigned long long current_tail = atomicAdd((unsigned long long *) &tail, 0ULL);
        return current_head >= current_tail;
    }

    [[nodiscard]] __device__ bool full() const {
        unsigned long long current_head = atomicAdd((unsigned long long *) &head, 0ULL);
        unsigned long long current_tail = atomicAdd((unsigned long long *) &tail, 0ULL);
        return (current_tail - current_head) >= CAPACITY;
    }

    [[nodiscard]] __device__ int size() const {
        unsigned long long current_head = atomicAdd((unsigned long long *) &head, 0ULL);
        unsigned long long current_tail = atomicAdd((unsigned long long *) &tail, 0ULL);
        return (int) (current_tail - current_head);
    }

    [[nodiscard]] __device__ T front() const {
        // 检查队列是否为空
        unsigned long long current_head = atomicAdd((unsigned long long *) &head, 0ULL);
        unsigned long long current_tail = atomicAdd((unsigned long long *) &tail, 0ULL);
        if (current_head >= current_tail) {
            // 队列空，返回默认构造的T类型对象
            return T{};
        }

        // 使用环形缓冲区索引获取队首元素但不移除
        int pos = (int) (current_head % CAPACITY);
        return data[pos];
    }
};

// 使用示例：
// __device__ DeviceQueue<Message, 1024> message_queue;
//
// __global__ void producer_kernel() {
//     Message msg;
//     // ... 填充msg
//     if (!message_queue.push(msg)) {
//         // 处理队列满的情况
//     }
// }
//
// __global__ void consumer_kernel() {
//     Message msg;
//     if (message_queue.pop(msg)) {
//         // 处理msg
//     }
// }

#endif // CUDA_DEVICE_QUEUE_CUH
