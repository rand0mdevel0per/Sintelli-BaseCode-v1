/**
 * @file structs.h
 * @brief 神经网络数据结构定义
 * 
 * @details
 * 定义神经网络模拟中使用的核心数据结构：
 * - 消息类型和结构
 * - KFE短期记忆槽位
 * - 神经元输入输出结构
 * - 连接信息
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
 * @brief 知识特征编码短期记忆槽位
 * 
 * @details
 * 存储神经元的短期知识片段，包含：
 * - 本地效用计数
 * - 最后访问周期
 * - 核心影响因子
 * - 知识片段向量
 * - 槽位有效性标志
 * - 16×16卷积特征
 * 
 * @note 使用SHA256哈希作为唯一标识符
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
    void conv() {
        ConvResidualProcessor::conv2d_16x16(Vmem, ConvKernel16(), conv16);
    }
};

/**
 * @struct Logic
 * @brief 逻辑数据结构
 * 
 * @details
 * 存储逻辑内容，包含：
 * - 周期计数
 * - 固定大小字符数组（替代std::string）
 * - 重要性评分
 * 
 * @note 使用固定大小数组避免动态内存分配
 */
struct Logic {
    ull Rcycles;
    wchar_t content[1024]; // 固定大小的字符数组替代 std::string
    double importance;
    std::string hash() {
        return sha256_hash<Logic>(*this);
    }
};

/**
 * @struct InputMessage
 * @brief 输入消息结构
 * 
 * @details
 * 支持多模态输入：
 * - 文本输入
 * - 图像输入（Base64编码）
 * 
 * @note 图像数据使用Base64编码以支持网络传输
 */
struct InputMessage {
    bool has_img;
    bool has_text;
    std::string text;
    std::string base64_image; // Base64编码的图像数据
};

struct ExtKFE_Slot {
    std::string hash;
    double conv16[16][16]; // 16x16 convolution features
    double importance; // Importance score
    double last_access_time; // Last access time
};

/**
 * @enum MessageType
 * @brief 消息类型枚举
 * 
 * @details
 * 定义神经元间通信的消息类型：
 * - NEURON_DATA: 神经元数据传递
 * - FIND_NEURON: 寻找连接目标
 * - REPLY_NEURON_FIND: 回复连接请求
 */
enum MessageType {
    NEURON_DATA, // 神经元数据传递
    FIND_NEURON, // 寻找连接目标
    REPLY_NEURON_FIND, // 回复连接请求
};

/**
 * @struct Message
 * @brief 神经元间通信的消息结构
 * 
 * @details
 * 包含完整的消息信息：
 * - 3D坐标信息（发送者、接收者、代理）
 * - 消息类型和活跃度
 * - 联合体存储不同压缩模式的数据
 * - 压缩模式标识
 * 
 * @note 使用联合体节省内存，支持多种压缩格式
 */
struct Message {
    ll last_proxy_coord[3]; // 最后一个代理坐标
    ll from_coord[3]; // 发送者坐标
    ll to_coord[3]; // 接收者坐标
    //double value[256][256];    // 数据矩阵
    double activity; // 活跃度
    MessageType type; // 消息类型
    ll remains; // 剩余跳数(用于FIND_NEURON)
    double weight; // 权重
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
 * @brief 神经元输入结构
 * 
 * @details
 * 神经元接收的输入数据：
 * - 256×256数据矩阵
 * - 活跃度和权重
 * - 来源坐标
 * 
 * @note 使用固定大小数组确保内存对齐
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