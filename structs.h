/**
 * @file structs.h
 * @brief Core data structures for neural network simulation
 * 
 * @details
 * Defines core data structures used in neural network simulation:
 * - Message types and structures
 * - KFE (Knowledge Feature Encoding) short-term memory slots
 * - Neuron input/output structures
 * - Connection information
 * - Feature vectors and compression modes
 *
 * @version 1.0
 * @date 2025-10-03
 */

#ifndef SRC_STRUCTS_H
#define SRC_STRUCTS_H

#define ll long long
#define ull unsigned ll

#include "conv16_res_msg.cuh"
#include "hasher.h"
#include <string>

/**
 * @struct KFE_STM_Slot
 * @brief Knowledge Feature Encoding Short-Term Memory Slot
 * 
 * @details
 * Stores compressed knowledge fragments for rapid access during inference.
 * Each slot contains utility metrics, importance scores, and memory content.
 * KFE provides contextual knowledge for enhanced reasoning and decision making.
 * 
 * Fields:
 * - Ulocal: Local utility of the knowledge fragment
 * - Rcycles: Reference cycle count for temporal tracking
 * - Icore: Core importance score
 * - V: Volatility metric
 * - Vmem: Memory content matrix (256x256)
 * - conv16: Compressed 16x16 representation
 *
 * Stores short-term knowledge fragments for neurons, containing:
 * - Local utility count
 * - Last access cycle
 * - Core influence factor
 * - Knowledge fragment vector
 * - Slot validity flag
 * - 16x16 convolution features
 * 
 * @note Uses SHA256 hash as unique identifier
 */
struct KFE_STM_Slot {
    double Ulocal; // Local utility count
    int Rcycles; // Last access cycle
    double Icore; // Core influence factor
    double Vmem[256][256]; // Knowledge fragment vector
    double V; // Slot validity flag
    double conv16[16][16]; // 16x16 convolution features
    std::string hash() const {
        return sha256_hash<KFE_STM_Slot>(*this);
    }
    __device__ void conv() {
        ConvResidualProcessor::conv2d_16x16(Vmem, ConvKernel16(), conv16);
    }
};

/**
 * @struct Logic
 * @brief Logic data structure
 * 
 * @details
 * Stores logic content, containing:
 * - Cycle count
 * - Fixed-size character array (替代std::string)
 * - Importance score
 * 
 * @note Uses fixed-size array to avoid dynamic memory allocation
 */
struct Logic {
    ull Rcycles;
    wchar_t content[1024]; // Fixed-size character array替代 std::string
    double importance;
    std::string hash() {
        return sha256_hash<Logic>(*this);
    }
};

/**
 * @struct InputMessage
 * @brief Input message structure
 * 
 * @details
 * Supports multimodal input:
 * - Text input
 * - Image input (Base64 encoded)
 * 
 * @note Image data uses Base64 encoding to support network transmission
 */
struct InputMessage {
    bool has_img;
    bool has_text;
    std::string text;
    std::string base64_image; // Base64 encoded image data
};

struct ExtKFE_Slot {
    std::string hash;
    double conv16[16][16]; // 16x16 convolution features
    double importance; // Importance score
    double last_access_time; // Last access time
};

/**
 * @enum MessageType
 * @brief Message type enumeration
 * 
 * Defines different types of messages used in neuron communication:
 * - NEURON_DATA: Neural data transmission between connected neurons
 * - FIND_NEURON: Connection discovery messages for network topology
 * - REPLY_NEURON_FIND: Responses to connection discovery requests
 */
enum MessageType {
    NEURON_DATA,        // Neuron data message
    FIND_NEURON,        // Find neuron connection
    REPLY_NEURON_FIND   // Reply to neuron connection request
};

/**
 * @struct Message
 * @brief Core message structure for neuron communication
 * 
 * @details
 * Contains essential message information:
 * - Type, coordinates, activity level, weight
 * - Adaptive message content (Full/Convolutional/Residual)
 * - Compression mode for efficient transmission
 * 
 * Features:
 * - Adaptive compression based on content importance
 * - Multi-dimensional coordinate system (3D neural grid)
 * - Weighted message propagation
 * - Support for different message types (data/find/reply)
 * 
 * Fields:
 * - type: Message type (NEURON_DATA/FIND_NEURON/REPLY_NEURON_FIND)
 * - from_coord: Sender coordinates (3D grid position)
 * - to_coord: Receiver coordinates (3D grid position)
 * - last_proxy_coord: Last proxy coordinates for routing
 * - activity: Message activity level
 * - weight: Message propagation weight
 * - remains: Remaining hops for FIND_NEURON messages
 * - adaptive_msg: Union of different message formats
 * - compression_mode: Compression strategy (FULL/RESIDUAL/CONV_ONLY)
 */
struct Message {
    ll last_proxy_coord[3]; // Last proxy coordinate
    ll from_coord[3]; // Sender coordinate
    ll to_coord[3]; // Receiver coordinate
    //double value[256][256];    // Data matrix
    double activity; // Activity level
    MessageType type; // Message type
    ll remains; // Remaining hops (for FIND_NEURON)
    double weight; // Weight
    union {
        FullMessage full_msg;
        ConvMessage conv_msg;
        ResidualMessage res_msg;
    } adaptive_msg;

    CompressionMode compression_mode;
};

/**
 * @struct NeuronUpdate
 * @brief 神经元更新结构
 * 
 * @details
 * 包含消息和方向信息，用于批量更新
 */
struct NeuronUpdate {
    Message msg;
    int direction;
};

/**
 * @struct NeuronInput
 * @brief Neuron input data structure
 * 
 * @details
 * Input data received by neurons containing:
 * - 256×256 data matrix for high-resolution information
 * - Activity level and weight for signal strength
 * - Source coordinates for spatial context
 * 
 * @note Uses fixed-size arrays to ensure memory alignment and GPU efficiency
 * 
 * This structure represents the core input format for neural processing:
 * - array[256][256]: High-dimensional input data matrix
 * - activity: Signal strength indicator (0.0-1.0)
 * - weight: Importance weight for the input
 * - from_coord[3]: 3D spatial coordinates of the sender
 */
struct NeuronInput {
    double array[256][256]; // 输入数据
    double activity; // 活跃度
    double weight; // 权重
    ll from_coord[3]; // 来源坐标
};

/**
 * @struct ConnectionInfo
 * @brief 神经元连接信息
 * 
 * @details
 * 描述神经元间的连接关系：
 * - 目标神经元坐标
 * - 使用的端口号
 * - 连接方向（输入/输出）
 */
struct ConnectionInfo {
    ll coord[3]; // 连接的神经元坐标
    int port; // 使用的端口
    bool inout; // true=输入, false=输出
};

struct NeuronData {
    // ===== 端口系统(4个逻辑端口) =====
    std::queue<NeuronInput> port_in[4]{};
    std::queue<NeuronInput> port_out[4]{};
    ll port_counts[4]{}; // 每个端口的连接数

    // ===== 连接信息 =====
    ConnectionInfo input_conns[2048]{};
    ConnectionInfo output_conns[2048]{};
    int input_conn_count{};
    int output_conn_count{};

    // ===== 端口变换矩阵 =====
    double input_multiplex_array[256][256][4]{}; // 输入端口变换
    double output_multiplex_array[256][256][4]{}; // 输出端口变换

    // ===== GEMM/DRC推理状态 =====
    double P_Matrix[256][256]{}; // 意图矩阵(当前状态)
    double P_stable[256][256]{}; // 稳定预测(认知基线)
    double W_predict[256][256]{}; // 自回归权重
    double M_KFE[256][256]{}; // KFE知识上下文
    double Deviation[256][256]{}; // 预测误差
    double PS_aggregate[256][256]{}; // 邻居共识
};

#endif //SRC_STRUCTS_H