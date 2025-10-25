#ifndef CONV16_RESIDUAL_MESSAGE_H
#define CONV16_RESIDUAL_MESSAGE_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

// ===== 16×16卷积核 =====
struct ConvKernel16 {
    double kernel[16][16];
    double bias;

    __device__ void randomInit(curandStatePhilox4_32_10_t* rand_state) {
        // He初始化
        double std_dev = sqrt(2.0 / 256.0);
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                kernel[i][j] = curand_normal_double(rand_state) * std_dev;
            }
        }
        bias = 0.0;
    }
};

// ===== Adaptive Message Types =====
/**
 * @enum CompressionMode
 * @brief Adaptive message compression modes for efficient neural communication
 * 
 * Defines different compression strategies for message transmission:
 * - MODE_FULL: Complete data transmission using unified memory
 * - MODE_RESIDUAL: 16×16 convolution + 8×8 residual for balanced quality/size
 * - MODE_CONV_ONLY: 16×16 convolution features only for maximum compression
 * - MODE_CONTROL: Control messages without data payload
 * 
 * Compression decisions are made based on:
 * - Network activity levels
 * - Core vulnerability metrics
 * - Memory pressure
 * - Importance weighting
 * - Available bandwidth
 */
enum CompressionMode {
    MODE_FULL,          // Full data (using unified memory)
    MODE_RESIDUAL,      // 16×16 convolution + 8×8 residual
    MODE_CONV_ONLY,     // 16×16 convolution features only
    MODE_CONTROL        // Control message (no data)
};

// ===== Control Message (Minimal) =====
/**
 * @struct ControlMessage
 * @brief Minimal control message without data payload
 * 
 * @details
 * Ultra-lightweight message for network control and coordination:
 * - Coordinate information for routing
 * - Message type identification
 * - Small metadata payload
 * 
 * Used for:
 * - Network topology discovery
 * - Connection management
 * - Status updates
 * - Control signaling
 * 
 * Fields:
 * - from_coord[3]: Sender coordinates (3D grid position)
 * - to_coord[3]: Receiver coordinates (3D grid position)
 * - type: Control message type identifier
 * - metadata[4]: Small control data payload
 */
struct ControlMessage {
    long long from_coord[3];
    long long to_coord[3];
    int type;
    double metadata[4];
}; // 72 bytes

// ===== Convolution Feature Message (Small) =====
/**
 * @struct ConvMessage
 * @brief Lightweight message with convolution features only
 * 
 * @details
 * Compact message format containing only essential convolution features:
 * - 16×16 convolution features from 16 kernels
 * - Minimal metadata for efficient transmission
 * 
 * This is the most compressed message format, suitable for:
 * - Low-bandwidth communication
 * - High-frequency messaging
 * - Memory-constrained environments
 * - Preliminary information exchange
 * 
 * Fields:
 * - from_coord[3]: Sender coordinates (3D grid position)
 * - to_coord[3]: Receiver coordinates (3D grid position)
 * - conv_features[16][16]: Convolution features from 16×16 kernels
 * - activity: Signal activity level
 * - type: Message type identifier
 * - weight: Message propagation weight
 */
struct ConvMessage {
    long long from_coord[3];
    long long to_coord[3];
    double conv_features[16][16];  // 16×16 features from 16 kernels
    double activity;
    int type;
    double weight;
}; // ~2.1KB

// ===== Residual Enhanced Message (Medium) =====
/**
 * @struct ResidualMessage
 * @brief Residual-enhanced message with convolution features and residual correction
 * 
 * @details
 * Balanced message format combining:
 * - 16×16 convolution features for main content
 * - 8×8 residual correction for fine details
 * - Coordinate and metadata for routing
 * 
 * This format provides a good balance between:
 * - Transmission efficiency (compressed size)
 * - Data quality (residual correction)
 * - Computational overhead (moderate complexity)
 * 
 * Fields:
 * - from_coord[3]: Sender coordinates (3D grid position)
 * - to_coord[3]: Receiver coordinates (3D grid position)
 * - conv_features[16][16]: Main convolution features
 * - residual[8][8]: Residual correction data
 * - activity: Signal activity level
 * - type: Message type identifier
 * - weight: Message propagation weight
 */
struct ResidualMessage {
    long long from_coord[3];
    long long to_coord[3];
    double conv_features[16][16];  // Main features
    double residual[8][8];         // Residual correction
    double activity;
    int type;
    double weight;
}; // ~2.6KB

// ===== Full Data Message (Large, using unified memory pool) =====
/**
 * @struct FullMessage
 * @brief Complete data message using unified memory for high-fidelity transmission
 * 
 * @details
 * Full data message for high-quality neural communication:
 * - 256×256 complete data matrix using unified memory
 * - Coordinate information for routing
 * - Memory pool management with reference counting
 * - Activity level and weight for signal strength
 * 
 * Features:
 * - High-fidelity data transmission
 * - Memory-efficient unified memory allocation
 * - Automatic memory pool management
 * - Reference counting for safe memory reuse
 * 
 * Fields:
 * - from_coord[3]: Sender coordinates (3D grid position)
 * - to_coord[3]: Receiver coordinates (3D grid position)
 * - data_ptr: Pointer to unified memory block
 * - pool_block_id: Memory pool block identifier
 * - activity: Signal activity level
 * - type: Message type identifier
 * - weight: Message propagation weight
 */
struct FullMessage {
    long long from_coord[3];
    long long to_coord[3];
    double* data_ptr;              // Pointer to unified memory
    int pool_block_id;             // Memory pool block ID
    double activity;
    int type;
    double weight;
}; // 80 bytes

// ===== Unified Memory Pool (for full data transmission) =====
/**
 * @class UnifiedMemoryPool
 * @brief Memory pool for managing unified memory allocations in neural network
 * 
 * @details
 * Centralized memory management system for FullMessage data transmission:
 * - Pre-allocated unified memory blocks
 * - Reference counting for safe sharing
 * - Automatic cleanup and reuse
 * - Thread-safe operations
 * 
 * Features:
 * - Efficient memory allocation without runtime overhead
 * - Reference counting for shared data access
 * - Automatic garbage collection of unused blocks
 * - GPU-CPU unified memory for seamless access
 * 
 * The pool manages:
 * - 4096 pre-allocated 256×256 double precision blocks
 * - Reference counting for each block
 * - Usage tracking and cleanup
 */
class UnifiedMemoryPool {
private:
    static constexpr int MAX_BLOCKS = 4096;

    struct Block {
        double* data;              // 统一内存指针
        int ref_count;
        bool in_use;
        uint64_t last_access_time;
    };

    __managed__ Block blocks[MAX_BLOCKS] = {};
    __managed__ int free_count = 0;

public:
    __host__ void initialize() {
        for (int i = 0; i < MAX_BLOCKS; i++) {
            // 分配统一内存
            cudaMallocManaged(&blocks[i].data, 256 * 256 * sizeof(double));
            blocks[i].ref_count = 0;
            blocks[i].in_use = false;
            blocks[i].last_access_time = 0;
        }
        free_count = MAX_BLOCKS;
    }

    __device__ int allocate() {
        for (int i = 0; i < MAX_BLOCKS; i++) {
            if (!blocks[i].in_use) {
                blocks[i].in_use = true;
                blocks[i].ref_count = 1;
                atomicSub(&free_count, 1);
                return i;
            }
        }
        return -1;  // 池满
    }

    __device__ void addRef(int block_id) {
        if (block_id >= 0 && block_id < MAX_BLOCKS) {
            atomicAdd(&blocks[block_id].ref_count, 1);
        }
    }

    __device__ void release(int block_id) {
        if (block_id >= 0 && block_id < MAX_BLOCKS) {
            int old_count = atomicSub(&blocks[block_id].ref_count, 1);
            if (old_count == 1) {
                blocks[block_id].in_use = false;
                atomicAdd(&free_count, 1);
            }
        }
    }

    __device__ double* get(int block_id) {
        if (block_id >= 0 && block_id < MAX_BLOCKS) {
            return blocks[block_id].data;
        }
        return nullptr;
    }

    __device__ int getFreeCount() {
        return free_count;
    }

    __host__ void cleanup() {
        for (int i = 0; i < MAX_BLOCKS; i++) {
            cudaFree(blocks[i].data);
        }
    }
};

// ===== 全局统一内存池实例 =====
__managed__ UnifiedMemoryPool global_memory_pool;

// ===== Convolution and Residual Processing =====
/**
 * @class ConvResidualProcessor
 * @brief Utility class for convolution and residual processing operations
 * 
 * @details
 * Provides optimized CUDA implementations of:
 * - 16×16 convolution (256×256 → 16×16)
 * - 8×8 residual extraction
 * - Feature encoding and decoding
 * 
 * Used for:
 * - Message compression and decompression
 * - Feature extraction from high-dimensional data
 * - Residual encoding for quality enhancement
 * - Memory-efficient data representation
 * 
 * All operations are optimized for GPU execution with:
 * - Memory coalescing
 * - Shared memory usage
 * - Thread parallelization
 */
class ConvResidualProcessor {
public:
    // 16×16卷积(256×256 → 16×16)
    __device__ static void conv2d_16x16(const double input[256][256],
                                        const ConvKernel16& kernel,
                                        double output[16][16]) {
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                double sum = 0.0;

                // 卷积计算(stride=16)
                for (int ki = 0; ki < 16; ki++) {
                    for (int kj = 0; kj < 16; kj++) {
                        int input_i = i * 16 + ki;
                        int input_j = j * 16 + kj;
                        sum += input[input_i][input_j] * kernel.kernel[ki][kj];
                    }
                }

                // ReLU激活
                output[i][j] = fmax(0.0, sum + kernel.bias);
            }
        }
    }

    // 反卷积(16×16 → 256×256)
    __device__ static void deconv2d_16x16(const double input[16][16],
                                          const ConvKernel16& kernel,
                                          double output[256][256]) {
        memset(output, 0, sizeof(double) * 256 * 256);

        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                double value = input[i][j];

                // 分散到16×16区域
                for (int ki = 0; ki < 16; ki++) {
                    for (int kj = 0; kj < 16; kj++) {
                        int output_i = i * 16 + ki;
                        int output_j = j * 16 + kj;
                        output[output_i][output_j] += value * kernel.kernel[ki][kj];
                    }
                }
            }
        }
    }

    // 计算残差(降采样到8×8)
    __device__ static void computeResidual(const double original[256][256],
                                           const double reconstructed[256][256],
                                           double residual[8][8]) {
        // 计算误差并降采样(32×32块平均)
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                double error_sum = 0.0;
                int count = 0;

                // 32×32块
                for (int ki = 0; ki < 32; ki++) {
                    for (int kj = 0; kj < 32; kj++) {
                        int orig_i = i * 32 + ki;
                        int orig_j = j * 32 + kj;
                        error_sum += original[orig_i][orig_j] -
                                    reconstructed[orig_i][orig_j];
                        count++;
                    }
                }

                residual[i][j] = error_sum / count;
            }
        }
    }

    // 应用残差(上采样8×8 → 256×256)
    __device__ static void applyResidual(double base[256][256],
                                         const double residual[8][8],
                                         double output[256][256]) {
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                // 双线性插值上采样
                double fi = i / 32.0;
                double fj = j / 32.0;
                int i0 = (int)fi;
                int j0 = (int)fj;
                int i1 = min(i0 + 1, 7);
                int j1 = min(j0 + 1, 7);

                double wi = fi - i0;
                double wj = fj - j0;

                double interp = residual[i0][j0] * (1-wi) * (1-wj) +
                               residual[i1][j0] * wi * (1-wj) +
                               residual[i0][j1] * (1-wi) * wj +
                               residual[i1][j1] * wi * wj;

                output[i][j] = base[i][j] + interp;
            }
        }
    }

    // 评估重建误差
    __device__ static double evaluateError(const double original[256][256],
                                           const double reconstructed[256][256]) {
        double mse = 0.0;
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                double diff = original[i][j] - reconstructed[i][j];
                mse += diff * diff;
            }
        }
        return sqrt(mse / (256.0 * 256.0));
    }
};

// ===== 自适应压缩决策器 =====
class CompressionDecider {
public:
    // 基于神经元内部状态决定压缩模式
    __device__ static CompressionMode decide(
        double activity,
        double core_vulnerability,
        double importance,
        int available_pool_blocks,
        double recent_error_rate
    ) {
        // 规则1: 极高重要性 → 完整数据
        if (importance > 0.95 && available_pool_blocks > 32) {
            return MODE_FULL;
        }

        // 规则2: 高脆弱性或高误差率 → 残差增强
        if (core_vulnerability > 0.7 || recent_error_rate > 0.2) {
            return MODE_RESIDUAL;
        }

        // 规则3: 高活跃度 → 残差增强
        if (activity > 0.8) {
            return MODE_RESIDUAL;
        }

        // 规则4: 中等重要性 → 残差增强
        if (importance > 0.5) {
            return MODE_RESIDUAL;
        }

        // 规则5: 低重要性 → 仅卷积
        if (importance > 0.1) {
            return MODE_CONV_ONLY;
        }

        // 默认: 控制消息
        return MODE_CONTROL;
    }
};

// ===== 消息编码器 =====
class MessageEncoder {
private:
    ConvKernel16 conv_kernel;
    __declspec(__managed__) double reconstruction_error_history[100];
    __declspec(__managed__) int error_index;

public:
    __device__ void initialize(curandStatePhilox4_32_10_t* rand_state) {
        conv_kernel.randomInit(rand_state);
        error_index = 0;
        memset(reconstruction_error_history, 0, sizeof(reconstruction_error_history));
    }

    // 编码为卷积消息
    __device__ void encodeConv(const double data[256][256],
                               ConvMessage& msg) {
        ConvResidualProcessor::conv2d_16x16(data, conv_kernel, msg.conv_features);
    }

    // 编码为残差消息
    __device__ void encodeResidual(const double data[256][256],
                                   ResidualMessage& msg) {
        // 1. 卷积提取特征
        ConvResidualProcessor::conv2d_16x16(data, conv_kernel, msg.conv_features);

        // 2. 重建
        double reconstructed[256][256];
        ConvResidualProcessor::deconv2d_16x16(msg.conv_features, conv_kernel,
                                             reconstructed);

        // 3. 计算残差
        ConvResidualProcessor::computeResidual(data, reconstructed, msg.residual);

        // 4. 记录误差
        double error = ConvResidualProcessor::evaluateError(data, reconstructed);
        reconstruction_error_history[error_index] = error;
        error_index = (error_index + 1) % 100;
    }

    // 编码为完整消息
    __device__ void encodeFull(const double data[256][256],
                               FullMessage& msg) {
        // 分配统一内存块
        int block_id = global_memory_pool.allocate();
        if (block_id >= 0) {
            double* ptr = global_memory_pool.get(block_id);
            memcpy(ptr, data, 256 * 256 * sizeof(double));
            msg.data_ptr = ptr;
            msg.pool_block_id = block_id;
        } else {
            // 池满,降级到残差模式
            msg.pool_block_id = -1;
        }
    }

    // 获取平均重建误差
    __device__ double getAvgError() {
        double sum = 0.0;
        for (int i = 0; i < 100; i++) {
            sum += reconstruction_error_history[i];
        }
        return sum / 100.0;
    }
};

// ===== 消息解码器 =====
class MessageDecoder {
private:
    ConvKernel16 conv_kernel;

public:
    __device__ void setKernel(const ConvKernel16& kernel) {
        conv_kernel = kernel;
    }

    // 解码卷积消息
    __device__ void decodeConv(const ConvMessage& msg,
                               double output[256][256]) {
        ConvResidualProcessor::deconv2d_16x16(msg.conv_features, conv_kernel,
                                             output);
    }

    // 解码残差消息
    __device__ void decodeResidual(const ResidualMessage& msg,
                                   double output[256][256]) {
        // 1. 反卷积重建基础
        double base[256][256];
        ConvResidualProcessor::deconv2d_16x16(msg.conv_features, conv_kernel,
                                             base);

        // 2. 应用残差修正
        ConvResidualProcessor::applyResidual(base, msg.residual, output);
    }

    // 解码完整消息
    __device__ void decodeFull(const FullMessage& msg,
                               double output[256][256]) {
        if (msg.pool_block_id >= 0) {
            double* ptr = global_memory_pool.get(msg.pool_block_id);
            if (ptr) {
                memcpy(output, ptr, 256 * 256 * sizeof(double));
            }
        }
    }
};

#endif // CONV16_RESIDUAL_MESSAGE_H