#ifndef SRC_CONVERTER_H
#define SRC_CONVERTER_H

#include <cstddef>
#include <cstdint>
#include <new>

// 简单内存管理包装
class DoubleBuffer {
private:
    double* data_;
    size_t size_;

public:
    DoubleBuffer(size_t size = 0) : data_(nullptr), size_(0) {
        if (size > 0) {
            resize(size);
        }
    }

    ~DoubleBuffer() {
        if (data_) {
            delete[] data_;
        }
    }

    bool resize(size_t new_size) {
        if (new_size == size_) return true;

        double* new_data = new (std::nothrow) double[new_size];
        if (!new_data && new_size > 0) return false;

        if (data_) {
            // 拷贝现有数据
            size_t copy_size = (new_size < size_) ? new_size : size_;
            for (size_t i = 0; i < copy_size; ++i) {
                new_data[i] = data_[i];
            }
            delete[] data_;
        }

        data_ = new_data;
        size_ = new_size;
        return true;
    }

    double* data() { return data_; }
    const double* data() const { return data_; }
    size_t size() const { return size_; }

    double& operator[](size_t index) { return data_[index]; }
    const double& operator[](size_t index) const { return data_[index]; }

    // 禁用拷贝
    DoubleBuffer(const DoubleBuffer&) = delete;
    DoubleBuffer& operator=(const DoubleBuffer&) = delete;

    // 允许移动
    DoubleBuffer(DoubleBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    DoubleBuffer& operator=(DoubleBuffer&& other) noexcept {
        if (this != &other) {
            if (data_) delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
};

class ByteBuffer {
private:
    uint8_t* data_;
    size_t size_;

public:
    ByteBuffer(size_t size = 0) : data_(nullptr), size_(0) {
        if (size > 0) {
            resize(size);
        }
    }

    ~ByteBuffer() {
        if (data_) {
            delete[] data_;
        }
    }

    bool resize(size_t new_size) {
        if (new_size == size_) return true;

        uint8_t* new_data = new (std::nothrow) uint8_t[new_size];
        if (!new_data && new_size > 0) return false;

        if (data_) {
            delete[] data_;
        }

        data_ = new_data;
        size_ = new_size;
        return true;
    }

    uint8_t* data() { return data_; }
    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }

    uint8_t& operator[](size_t index) { return data_[index]; }
    const uint8_t& operator[](size_t index) const { return data_[index]; }

    // 禁用拷贝
    ByteBuffer(const ByteBuffer&) = delete;
    ByteBuffer& operator=(const ByteBuffer&) = delete;

    // 允许移动
    ByteBuffer(ByteBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    ByteBuffer& operator=(ByteBuffer&& other) noexcept {
        if (this != &other) {
            if (data_) delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
};

// 简单字符串包装（避免std::string）
class SimpleString {
private:
    char* data_;
    size_t length_;

public:
    SimpleString() : data_(nullptr), length_(0) {}

    SimpleString(const char* str) : data_(nullptr), length_(0) {
        if (str) {
            assign(str);
        }
    }

    ~SimpleString() {
        clear();
    }

    bool assign(const char* str) {
        clear();
        if (!str) return true;

        length_ = 0;
        while (str[length_] != '\0') length_++;

        if (length_ > 0) {
            data_ = new (std::nothrow) char[length_ + 1];
            if (!data_) {
                length_ = 0;
                return false;
            }

            for (size_t i = 0; i < length_; ++i) {
                data_[i] = str[i];
            }
            data_[length_] = '\0';
        }
        return true;
    }

    void clear() {
        if (data_) {
            delete[] data_;
            data_ = nullptr;
        }
        length_ = 0;
    }

    const char* c_str() const { return data_ ? data_ : ""; }
    size_t length() const { return length_; }
    bool empty() const { return length_ == 0; }

    // 禁用拷贝
    SimpleString(const SimpleString&) = delete;
    SimpleString& operator=(const SimpleString&) = delete;

    // 允许移动
    SimpleString(SimpleString&& other) noexcept
        : data_(other.data_), length_(other.length_) {
        other.data_ = nullptr;
        other.length_ = 0;
    }

    SimpleString& operator=(SimpleString&& other) noexcept {
        if (this != &other) {
            clear();
            data_ = other.data_;
            length_ = other.length_;
            other.data_ = nullptr;
            other.length_ = 0;
        }
        return *this;
    }
};

// Base64编码表
extern const char base64_chars[];

// 函数声明
bool base64_encode(const uint8_t* data, size_t len, SimpleString& result);
bool pool_matrix(const double* matrix, size_t rows, size_t cols,
                size_t pool_size, DoubleBuffer& pooled);
bool extract_features(const double* matrix, size_t rows, size_t cols,
                     DoubleBuffer& features);
bool features_to_utf8(const DoubleBuffer& features, int quantize_bits,
                     SimpleString& result);
bool matrix_to_utf8_full(const double* matrix, size_t rows, size_t cols,
                        SimpleString& result);

// 前向比对类
class MatrixComparator {
private:
    DoubleBuffer last_matrix_;
    size_t matrix_size_;
    double change_threshold_;

public:
    MatrixComparator(size_t size = 256 * 256, double threshold = 1e-4);
    bool check_change(const double* current_matrix, double* out_mse = nullptr);
    void force_update(const double* matrix);
    void reset();
    double get_threshold() const;
    void set_threshold(double threshold);
};

// 多帧叠加类
class FrameAccumulator {
private:
    DoubleBuffer accumulated_;
    size_t matrix_size_;
    int frame_count_;

public:
    FrameAccumulator(size_t size = 256 * 256);
    bool add_frame(const double* matrix);
    bool get_accumulated_features(SimpleString& result);
    void reset();
    int get_frame_count() const;
};

#endif // SRC_CONVERTER_H