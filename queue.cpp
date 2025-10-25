//
// Created by ASUS on 10/3/2025.
////
// Created by ASUS on 9/29/2025.
//

#ifndef QUEUE_HPP
#define QUEUE_HPP
#define ll long long
#define ull unsigned ll

#if defined (_MSC_VER)
#include <intrin0.inl.h>
#endif



// 统一的ULL原子加实现
unsigned long long atomic_add_ull_builtin(unsigned long long* ptr, unsigned long long value) {
    // 尝试不同的编译器内置函数
    #if defined(__GNUC__) || defined(__clang__)
        // GCC/Clang 版本
        return __sync_fetch_and_add(ptr, value);
    #elif defined(_MSC_VER)
        // MSVC 版本 - 即使NVCC也可能支持这些
    #ifdef _WIN64
        return _InterlockedExchangeAdd64((__int64*)ptr, (__int64)value);
    #else
        // 32位Windows
        return _InterlockedExchangeAdd64((__int64*)ptr, (__int64)value);
    #endif

    #else
        // 回退方案
    #warning "Using fallback atomic implementation"
        return __sync_fetch_and_add(ptr, value);
    #endif
}

ull atomicAdd(ull* ptr, ull value) {
    return atomic_add_ull_builtin(ptr, static_cast<__int64>(value));
}

template<typename T, int CAPACITY>
struct Queue {
    T data[CAPACITY];
    ull head;
    ull tail;

    void init() {
        head = 0ULL;
        tail = 0ULL;
    }

    bool push(const T& item) {
        // 原子地获取并递增tail
        ull old_tail = atomicAdd(&tail, 1ULL);

        // 检查队列是否已满
        ull current_head = atomicAdd(&head, 0ULL);
        if (old_tail - current_head >= CAPACITY) {
            // 队列满，回退tail
            atomicAdd(&tail, -1LL);
            return false;
        }

        // 使用环形缓冲区索引
        int pos = (int)(old_tail % CAPACITY);
        data[pos] = item;

        return true;
    }

    bool pop(T& result) {
        // 原子地获取并递增head
        ull old_head = atomicAdd(&head, 1ULL);

        // 检查队列是否为空
        ull current_tail = atomicAdd(&tail, 0ULL);
        if (old_head >= current_tail) {
            // 队列空，回退head
            atomicAdd(&head, -1LL);
            return false;
        }

        // 使用环形缓冲区索引
        int pos = (int)(old_head % CAPACITY);
        result = data[pos];

        return true;
    }

    [[nodiscard]] bool empty() const {
        ull current_head = atomicAdd((ull*)&head, 0ULL);
        ull current_tail = atomicAdd((ull*)&tail, 0ULL);
        return current_head >= current_tail;
    }

    [[nodiscard]] bool full() const {
        ull current_head = atomicAdd((ull*)&head, 0ULL);
        ull current_tail = atomicAdd((ull*)&tail, 0ULL);
        return (current_tail - current_head) >= CAPACITY;
    }

    [[nodiscard]] int size() const {
        ull current_head = atomicAdd((ull*)&head, 0ULL);
        ull current_tail = atomicAdd((ull*)&tail, 0ULL);
        return (int)(current_tail - current_head);
    }
};

// 使用示例：
// DeviceQueue<Message, 1024> message_queue;
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

#endif // QUEUE_HPP