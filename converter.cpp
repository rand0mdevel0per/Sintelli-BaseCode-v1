//
// Created by ASUS on 10/3/2025.
//

#include "converter.h"
#include <cmath>
#include <new>

// Base64编码表
const char base64_chars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

bool base64_encode(const uint8_t* data, size_t len, SimpleString& result) {
    if (!data && len > 0) return false;

    // 计算输出长度
    size_t output_len = ((len + 2) / 3) * 4;
    char* output = new (std::nothrow) char[output_len + 1];
    if (!output) return false;

    size_t i = 0, j = 0;
    size_t out_index = 0;
    uint8_t char_array_3[3], char_array_4[4];

    while (len-- > 0) {
        char_array_3[i++] = *(data++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (i = 0; i < 4; i++) {
                output[out_index++] = base64_chars[char_array_4[i]];
            }
            i = 0;
        }
    }

    if (i > 0) {
        for (j = i; j < 3; j++) {
            char_array_3[j] = 0;
        }

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

        for (j = 0; j < i + 1; j++) {
            output[out_index++] = base64_chars[char_array_4[j]];
        }

        while (i++ < 3) {
            output[out_index++] = '=';
        }
    }

    output[out_index] = '\0';
    result = SimpleString(output);
    delete[] output;
    return true;
}

bool pool_matrix(const double* matrix, size_t rows, size_t cols,
                size_t pool_size, DoubleBuffer& pooled) {
    if (!matrix || rows == 0 || cols == 0 || pool_size == 0) return false;

    size_t out_rows = rows / pool_size;
    size_t out_cols = cols / pool_size;

    if (!pooled.resize(out_rows * out_cols)) return false;

    for (size_t i = 0; i < out_rows; ++i) {
        for (size_t j = 0; j < out_cols; ++j) {
            double sum = 0.0;
            size_t count = 0;

            for (size_t pi = 0; pi < pool_size; ++pi) {
                for (size_t pj = 0; pj < pool_size; ++pj) {
                    size_t row_idx = i * pool_size + pi;
                    size_t col_idx = j * pool_size + pj;

                    if (row_idx < rows && col_idx < cols) {
                        sum += matrix[row_idx * cols + col_idx];
                        count++;
                    }
                }
            }

            pooled[i * out_cols + j] = (count > 0) ? (sum / count) : 0.0;
        }
    }

    return true;
}

bool extract_features(const double* matrix, size_t rows, size_t cols,
                     DoubleBuffer& features) {
    DoubleBuffer level1;
    if (!pool_matrix(matrix, rows, cols, 8, level1)) return false;

    // 第一层应该是32x32
    if (level1.size() != 32 * 32) return false;

    return pool_matrix(level1.data(), 32, 32, 8, features);
}

bool features_to_utf8(const DoubleBuffer& features, int quantize_bits,
                     SimpleString& result) {
    if (features.size() == 0) return false;

    // 找到最小最大值
    double min_val = features[0];
    double max_val = features[0];
    for (size_t i = 1; i < features.size(); ++i) {
        if (features[i] < min_val) min_val = features[i];
        if (features[i] > max_val) max_val = features[i];
    }

    double range = max_val - min_val;
    if (range < 1e-10) {
        range = 1.0;
    }

    // 量化
    int max_val_int = (1 << quantize_bits) - 1;
    ByteBuffer quantized(features.size());

    for (size_t i = 0; i < features.size(); ++i) {
        double normalized = (features[i] - min_val) / range;
        int quantized_val = static_cast<int>(normalized * max_val_int);

        // clamp
        if (quantized_val < 0) quantized_val = 0;
        if (quantized_val > max_val_int) quantized_val = max_val_int;

        quantized[i] = static_cast<uint8_t>(quantized_val);
    }

    return base64_encode(quantized.data(), quantized.size(), result);
}

bool matrix_to_utf8_full(const double* matrix, size_t rows, size_t cols,
                        SimpleString& result) {
    if (!matrix) return false;

    size_t size = rows * cols;

    // 检查有效性
    bool has_valid_data = false;
    for (size_t i = 0; i < size; ++i) {
        if (std::isnan(matrix[i]) || std::isinf(matrix[i])) {
            return false;
        }
        if (std::abs(matrix[i]) > 1e-10) {
            has_valid_data = true;
        }
    }

    if (!has_valid_data) return false;

    // 转换为字节
    ByteBuffer bytes(size * sizeof(double));
    const uint8_t* byte_ptr = reinterpret_cast<const uint8_t*>(matrix);

    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = byte_ptr[i];
    }

    return base64_encode(bytes.data(), bytes.size(), result);
}

// MatrixComparator实现
MatrixComparator::MatrixComparator(size_t size, double threshold)
    : matrix_size_(size), change_threshold_(threshold) {
    last_matrix_.resize(size);
    reset();
}

bool MatrixComparator::check_change(const double* current_matrix, double* out_mse) {
    if (!current_matrix) return false;

    // 检查有效性
    for (size_t i = 0; i < matrix_size_; ++i) {
        if (std::isnan(current_matrix[i]) || std::isinf(current_matrix[i])) {
            return false;
        }
    }

    // 计算MSE
    double mse = 0.0;
    for (size_t i = 0; i < matrix_size_; ++i) {
        double diff = current_matrix[i] - last_matrix_[i];
        mse += diff * diff;
    }
    mse /= matrix_size_;

    if (mse < change_threshold_) {
        return false;
    }

    // 更新
    for (size_t i = 0; i < matrix_size_; ++i) {
        last_matrix_[i] = current_matrix[i];
    }

    if (out_mse) {
        *out_mse = mse;
    }
    return true;
}

void MatrixComparator::force_update(const double* matrix) {
    if (!matrix) return;
    for (size_t i = 0; i < matrix_size_; ++i) {
        last_matrix_[i] = matrix[i];
    }
}

void MatrixComparator::reset() {
    for (size_t i = 0; i < matrix_size_; ++i) {
        last_matrix_[i] = 0.0;
    }
}

double MatrixComparator::get_threshold() const {
    return change_threshold_;
}

void MatrixComparator::set_threshold(double threshold) {
    change_threshold_ = threshold;
}

// FrameAccumulator实现
FrameAccumulator::FrameAccumulator(size_t size)
    : matrix_size_(size), frame_count_(0) {
    accumulated_.resize(size);
    reset();
}

bool FrameAccumulator::add_frame(const double* matrix) {
    if (!matrix) return false;

    for (size_t i = 0; i < matrix_size_; ++i) {
        accumulated_[i] += matrix[i];
    }
    frame_count_++;
    return true;
}

bool FrameAccumulator::get_accumulated_features(SimpleString& result) {
    if (frame_count_ == 0) return false;

    DoubleBuffer averaged(matrix_size_);
    for (size_t i = 0; i < matrix_size_; ++i) {
        averaged[i] = accumulated_[i] / frame_count_;
    }

    DoubleBuffer features;
    if (!extract_features(averaged.data(), 256, 256, features)) {
        return false;
    }

    return features_to_utf8(features, 8, result);
}

void FrameAccumulator::reset() {
    for (size_t i = 0; i < matrix_size_; ++i) {
        accumulated_[i] = 0.0;
    }
    frame_count_ = 0;
}

int FrameAccumulator::get_frame_count() const {
    return frame_count_;
}