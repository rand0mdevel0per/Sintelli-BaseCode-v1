/**
 * @file gpu_containers.cuh
 * @brief GPU-compatible dynamic containers (替代STL的vector和string)
 */

#ifndef GPU_CONTAINERS_CUH
#define GPU_CONTAINERS_CUH

#include <cuda_runtime.h>
#include <cstring>

// ===== GPU String 喵 =====
/**
 * @brief GPU兼容的字符串类
 * 使用小字符串优化(SSO)：<=31字节用栈，>31字节用堆
 */
class GPUString {
private:
    static constexpr size_t SSO_SIZE = 31;

    union {
        char sso_buffer[SSO_SIZE + 1];  // 小字符串优化缓冲区
        struct {
            char* ptr;                   // 堆分配指针
            size_t capacity;             // 容量
        } heap;
    } data;

    size_t length;
    bool is_sso;  // 是否使用小字符串优化

public:
    // 默认构造喵
    __host__ __device__ GPUString() : length(0), is_sso(true) {
        data.sso_buffer[0] = '\0';
    }

    // C字符串构造喵
    __host__ __device__ GPUString(const char* str) {
        length = 0;
        if (str) {
            while (str[length] != '\0') length++;
        }

        if (length <= SSO_SIZE) {
            // 使用SSO喵
            is_sso = true;
            if (str) {
                memcpy(data.sso_buffer, str, length);
            }
            data.sso_buffer[length] = '\0';
        } else {
            // 需要堆分配喵
            is_sso = false;
            data.heap.capacity = length + 1;
            #ifdef __CUDA_ARCH__
            // GPU上使用malloc
            data.heap.ptr = (char*)malloc(data.heap.capacity);
            #else
            // CPU上使用cudaMallocManaged
            cudaMallocManaged(&data.heap.ptr, data.heap.capacity);
            #endif
            if (str && data.heap.ptr) {
                memcpy(data.heap.ptr, str, length);
                data.heap.ptr[length] = '\0';
            }
        }
    }

    // 拷贝构造喵
    __host__ __device__ GPUString(const GPUString& other) : length(other.length), is_sso(other.is_sso) {
        if (is_sso) {
            memcpy(data.sso_buffer, other.data.sso_buffer, SSO_SIZE + 1);
        } else {
            data.heap.capacity = other.data.heap.capacity;
            #ifdef __CUDA_ARCH__
            data.heap.ptr = (char*)malloc(data.heap.capacity);
            #else
            cudaMallocManaged(&data.heap.ptr, data.heap.capacity);
            #endif
            if (data.heap.ptr && other.data.heap.ptr) {
                memcpy(data.heap.ptr, other.data.heap.ptr, length + 1);
            }
        }
    }

    // 移动构造喵
    __host__ __device__ GPUString(GPUString&& other) noexcept : length(other.length), is_sso(other.is_sso) {
        if (is_sso) {
            memcpy(data.sso_buffer, other.data.sso_buffer, SSO_SIZE + 1);
        } else {
            data.heap = other.data.heap;
            other.data.heap.ptr = nullptr;
            other.length = 0;
            other.is_sso = true;
        }
    }

    // 析构喵
    __host__ __device__ ~GPUString() {
        if (!is_sso && data.heap.ptr) {
            #ifdef __CUDA_ARCH__
            free(data.heap.ptr);
            #else
            cudaFree(data.heap.ptr);
            #endif
        }
    }

    // 赋值运算符喵
    __host__ __device__ GPUString& operator=(const GPUString& other) {
        if (this != &other) {
            this->~GPUString();
            new (this) GPUString(other);
        }
        return *this;
    }

    // 获取C字符串喵
    __host__ __device__ const char* c_str() const {
        return is_sso ? data.sso_buffer : data.heap.ptr;
    }

    __host__ __device__ size_t size() const { return length; }
    __host__ __device__ bool empty() const { return length == 0; }

    // 比较运算符喵
    __host__ __device__ bool operator==(const GPUString& other) const {
        if (length != other.length) return false;
        const char* str1 = c_str();
        const char* str2 = other.c_str();
        for (size_t i = 0; i < length; i++) {
            if (str1[i] != str2[i]) return false;
        }
        return true;
    }

    // 下标访问喵
    __host__ __device__ char operator[](size_t idx) const {
        return is_sso ? data.sso_buffer[idx] : data.heap.ptr[idx];
    }
};

// ===== GPU Vector 喵 =====
/**
 * @brief GPU兼容的动态数组
 * @tparam T 元素类型
 */
template<typename T>
class GPUVector {
private:
    T* data_ptr;
    size_t size_;
    size_t capacity_;

    __host__ __device__ void reallocate(size_t new_capacity) {
        T* new_ptr;
        #ifdef __CUDA_ARCH__
        new_ptr = (T*)malloc(new_capacity * sizeof(T));
        #else
        cudaMallocManaged(&new_ptr, new_capacity * sizeof(T));
        #endif

        if (new_ptr && data_ptr) {
            // 移动已有元素喵
            for (size_t i = 0; i < size_; i++) {
                new (&new_ptr[i]) T(static_cast<T&&>(data_ptr[i]));
                data_ptr[i].~T();
            }

            #ifdef __CUDA_ARCH__
            free(data_ptr);
            #else
            cudaFree(data_ptr);
            #endif
        }

        data_ptr = new_ptr;
        capacity_ = new_capacity;
    }

public:
    // 默认构造喵
    __host__ __device__ GPUVector() : data_ptr(nullptr), size_(0), capacity_(0) {}

    // 预分配容量喵
    __host__ __device__ explicit GPUVector(size_t initial_capacity) : size_(0) {
        capacity_ = initial_capacity;
        #ifdef __CUDA_ARCH__
        data_ptr = (T*)malloc(capacity_ * sizeof(T));
        #else
        cudaMallocManaged(&data_ptr, capacity_ * sizeof(T));
        #endif
    }

    // 析构喵
    __host__ __device__ ~GPUVector() {
        if (data_ptr) {
            // 调用所有元素的析构函数喵
            for (size_t i = 0; i < size_; i++) {
                data_ptr[i].~T();
            }
            #ifdef __CUDA_ARCH__
            free(data_ptr);
            #else
            cudaFree(data_ptr);
            #endif
        }
    }

    // 拷贝构造喵
    __host__ __device__ GPUVector(const GPUVector& other) : size_(other.size_), capacity_(other.capacity_) {
        if (capacity_ > 0) {
            #ifdef __CUDA_ARCH__
            data_ptr = (T*)malloc(capacity_ * sizeof(T));
            #else
            cudaMallocManaged(&data_ptr, capacity_ * sizeof(T));
            #endif

            if (data_ptr) {
                for (size_t i = 0; i < size_; i++) {
                    new (&data_ptr[i]) T(other.data_ptr[i]);
                }
            }
        } else {
            data_ptr = nullptr;
        }
    }

    // 移动构造喵
    __host__ __device__ GPUVector(GPUVector&& other) noexcept
        : data_ptr(other.data_ptr), size_(other.size_), capacity_(other.capacity_) {
        other.data_ptr = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    // Push back喵
    __host__ __device__ void push_back(const T& value) {
        if (size_ >= capacity_) {
            size_t new_capacity = capacity_ == 0 ? 8 : capacity_ * 2;
            reallocate(new_capacity);
        }

        if (data_ptr) {
            new (&data_ptr[size_]) T(value);
            size_++;
        }
    }

    // Emplace back喵（避免拷贝）
    template<typename... Args>
    __host__ __device__ void emplace_back(Args&&... args) {
        if (size_ >= capacity_) {
            size_t new_capacity = capacity_ == 0 ? 8 : capacity_ * 2;
            reallocate(new_capacity);
        }

        if (data_ptr) {
            new (&data_ptr[size_]) T(static_cast<Args&&>(args)...);
            size_++;
        }
    }

    // Pop back喵
    __host__ __device__ void pop_back() {
        if (size_ > 0) {
            size_--;
            data_ptr[size_].~T();
        }
    }

    // 访问喵
    __host__ __device__ T& operator[](size_t idx) { return data_ptr[idx]; }
    __host__ __device__ const T& operator[](size_t idx) const { return data_ptr[idx]; }

    __host__ __device__ T& back() { return data_ptr[size_ - 1]; }
    __host__ __device__ const T& back() const { return data_ptr[size_ - 1]; }

    // 信息喵
    __host__ __device__ size_t size() const { return size_; }
    __host__ __device__ size_t capacity() const { return capacity_; }
    __host__ __device__ bool empty() const { return size_ == 0; }

    // 清空喵（保留容量）
    __host__ __device__ void clear() {
        for (size_t i = 0; i < size_; i++) {
            data_ptr[i].~T();
        }
        size_ = 0;
    }

    // 预留容量喵
    __host__ __device__ void reserve(size_t new_capacity) {
        if (new_capacity > capacity_) {
            reallocate(new_capacity);
        }
    }

    // 迭代器支持喵
    __host__ __device__ T* begin() { return data_ptr; }
    __host__ __device__ T* end() { return data_ptr + size_; }
    __host__ __device__ const T* begin() const { return data_ptr; }
    __host__ __device__ const T* end() const { return data_ptr + size_; }
};

// ===== 使用示例喵 =====
/*
// 在Neuron类中使用：
class Neuron {
private:
    GPUVector<ExtKFE_Slot> ext_kfe_slots;  // 替代std::vector
    GPUVector<GPUString> query_cache;       // 字符串数组

    __device__ void example() {
        // 使用vector喵
        ExtKFE_Slot slot;
        ext_kfe_slots.push_back(slot);

        // 使用string喵
        GPUString str1("Hello");
        GPUString str2 = str1;
        if (str1 == str2) {
            // ...
        }

        // 遍历喵
        for (size_t i = 0; i < ext_kfe_slots.size(); i++) {
            auto& slot = ext_kfe_slots[i];
            // 处理slot...
        }
    }
};
*/

#endif // GPU_CONTAINERS_CUH