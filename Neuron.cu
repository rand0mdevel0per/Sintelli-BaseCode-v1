#pragma once
/**
 * @file Neuron.cu
 * @brief Implementation of a single neuron in a 3D neural network.
 *
 * This file contains the core logic for a neuron, including:
 * - 3D spatial positioning and neighbor connections.
 * - Adaptive message compression and routing.
 * - Short-term memory (KFE) system.
 * - Mixed convolution and GEMM inference.
 * - Multi-port input-output system.
 *
 * @author iFlow Development Team
 * @version 1.0
 * @date 2025-10-03
 * @copyright Copyright (c) 2025 iFlow Project Group
 */

#ifndef SRC_NEURON_H
#define SRC_NEURON_H

#include <iostream>
#include "deviceQueue.cpp"
#include "matrixMultiplex.cpp"
#include <curand_kernel.h>
#include "cern.cuh"
#include "isw.hpp"
#include "conv16_res_msg.cuh"
#include <cmath>    // for sqrt, exp, etc
#include <sm_20_intrinsics.h>
#include <vector>
#include "hasher.h"
#include "structs.h"
#include "sim.cu"
#include "GPUMutex.cu"
#include <cuda_fp16.hpp>
#include "gpu_containers.cuh"
#include "nlohmann/json.hpp"

#define ll long long
#define ull unsigned ll

struct NeuronStats {
    bool training;
    double activity;
    ll port_counts[4];
    double core_vul;
    double importance;

    nlohmann::json to_json() {
        return nlohmann::json{
                {"training", training},
                {"activity", activity},
                {"port_counts",port_counts},
                {"core_vul",core_vul},
                {"importance",importance}
        };
    }
};

/**
 * @brief Neuron class representing a single computational unit in the neural network.
 *
 * This class is designed to:
 * - Receive and process input signals.
 * - Perform inference calculations.
 * - Maintain short-term memory.
 * - Communicate with other neurons.
 * - Adaptively compress and route messages.
 *
 * @note Optimized for GPU execution using CUDA.
 */
class Neuron {
public:
    /**
     * @brief Delete default constructor - neurons must be explicitly initialized.
     * @note Neurons require complete initialization parameters and cannot be default constructed.
     * This ensures proper setup of all neuron components and connections.
     */
    Neuron() = delete;

    /**
     * @brief Neuron constructor.
     * 
     * @param[in] queues Array of 6 device queue pointers for message passing in 6 directions
     * @param[in] coord 3D coordinate array defining neuron position in space
     * @param[in] seed Random number seed for initializing random state
     * @param[in] queue_ptr Main device queue pointer for receiving messages
     * @param[in] storage_queue KFE storage queue pointer
     * @param[in] query_queue KFE query queue pointer
     * @param[in] result_queue KFE result queue pointer
     * 
     * @throws No explicit exceptions thrown, but relies on CUDA runtime error checking
     * 
     * @note Constructor initializes all matrices, queues, and state variables to default values
     * Sets up connections, random state, and prepares neuron for operation
     */
    __device__ Neuron(DeviceQueue<Message, 32> *queues[6], ll coord[3], ull seed, DeviceQueue<Message, 32> *queue_ptr,
                      DeviceQueue<KFE_STM_Slot, 32> *storage_queue, DeviceQueue<GPUString, 32> *query_queue,
                      DeviceQueue<KFE_STM_Slot, 32> *result_queue) {
        encoder = MessageEncoder();
        decoder = MessageDecoder();
        importance = 0;
        for (int i = 0; i < 6; i++) {
            neighbour_queues[i] = queues[i]; // Save pointer
        }
        queue = queue_ptr;

        // Initialize KFE storage queue
        this->kfe_storage_queue = storage_queue;
        this->kfe_query_queue = query_queue;
        this->kfe_result_queue = result_queue;

        // Initialize basic state
        activity = 0.0;
        input_conn_count = 0;
        output_conn_count = 0;
        cycle_counter = 0;
        core_vulnerability = 0.0;
        STM_aggregate_utility = 0.0;
        history_index = 0;

        // Initialize random number generator
        curand_init(seed, 0, 0, &rand_state);

        // Save local coordinates
        memcpy(local_coord, coord, 3 * sizeof(ll));

        // Clear KFE slots
        for (auto &i: kfe_local) {
            i = {};
        }

        // Initialize port queues and counts
        for (int i = 0; i < 4; i++) {
            port_in[i] = DeviceQueue<NeuronInput, 1024>();
            port_out[i] = DeviceQueue<NeuronInput, 1024>();
            port_counts[i] = 0;
        }

        // Clear connection information
        for (int i = 0; i < 2048; i++) {
            input_conns[i] = {};
            output_conns[i] = {};
        }

        // Initialize matrices
        initializeMatrices();
    }

    __device__ NeuronData save() {
        NeuronData data{};
        while (!port_in->empty()) {
            NeuronInput cache{};
            port_in->pop(cache);
            data.port_in->push(cache);
        }
        while (!port_out->empty()) {
            NeuronInput cache{};
            port_in->pop(cache);
            data.port_in->push(cache);
        }
        memcpy(data.port_counts, port_counts, sizeof(port_counts));

        memcpy(data.input_conns, input_conns, sizeof(input_conns));
        memcpy(data.output_conns, output_conns, sizeof(output_conns));
        data.input_conn_count = input_conn_count;
        data.output_conn_count = output_conn_count;

        memcpy(data.input_multiplex_array, input_multiplex_array, sizeof(input_multiplex_array));
        memcpy(data.output_multiplex_array, output_multiplex_array, sizeof(output_multiplex_array));

        memcpy(data.P_Matrix, P_Matrix, sizeof(P_Matrix));
        memcpy(data.P_stable, P_stable, sizeof(P_stable));
        memcpy(data.W_predict, W_predict, sizeof(W_predict));
        memcpy(data.M_KFE, M_KFE, sizeof(M_KFE));
        memcpy(data.Deviation, Deviation, sizeof(Deviation));
        memcpy(data.PS_aggregate, PS_aggregate, sizeof(PS_aggregate));

        return data;
    }

    __host__ bool load(NeuronData data) {
        try {
            while (!data.port_in->empty()) {
                NeuronInput cache = data.port_in->front();
                data.port_in->pop();
                port_in->push(cache);
            }
            while (!data.port_out->empty()) {
                NeuronInput cache = data.port_out->front();
                data.port_out->pop();
                port_out->push(cache);
            }
            memcpy(port_counts, data.port_counts, sizeof(port_counts));

            memcpy(input_conns, data.input_conns, sizeof(input_conns));
            memcpy(output_conns, data.output_conns, sizeof(output_conns));
            input_conn_count = data.input_conn_count;
            output_conn_count = data.output_conn_count;

            memcpy(input_multiplex_array, data.input_multiplex_array, sizeof(input_multiplex_array));
            memcpy(output_multiplex_array, data.output_multiplex_array, sizeof(output_multiplex_array));

            memcpy(P_Matrix, data.P_Matrix, sizeof(P_Matrix));
            memcpy(P_stable, data.P_stable, sizeof(P_stable));
            memcpy(W_predict, data.W_predict, sizeof(W_predict));
            memcpy(M_KFE, data.M_KFE, sizeof(M_KFE));
            memcpy(Deviation, data.Deviation, sizeof(Deviation));
            memcpy(PS_aggregate, data.PS_aggregate, sizeof(PS_aggregate));
        } catch (...) {
            return false;
        }
        return true;
    }

    [[nodiscard]] double get_noise() const { return noise; }
    [[nodiscard]] double get_learn_rt() const { return learn; }

    void set_noise(double new_ns) {
        noise = new_ns;
    }

    void set_learn_rt(double new_rt) {
        learn = new_rt;
    }

    /**
     * @brief Generate positive random number in range [0,1)
     * @return double Uniformly distributed random number
     */
    __device__ double generatePositiveRandom() {
        return curand_uniform_double(&rand_state);
    }

    __device__ double generatePositiveNormalRandom() {
        double val = curand_normal_double(&rand_state) * 0.2 + 0.5;
        return fmax(val, 0.0);
    }

    __device__ double randomInRange(double min, double max) {
        return curand_uniform_double(&rand_state) * (max - min) + min;
    }

    __device__ ull randomULLInRange(ull min, ull max) {
        return curand(&rand_state) % (max - min) + min;
    }

    [[nodiscard]] double get_activity() const {
        return activity;
    }

    NeuronStats get_stats() {
        auto stats = NeuronStats{
            training, activity, port_counts[0],
            port_counts[1],
            port_counts[2],
            port_counts[3],
            core_vulnerability,
            importance
        };
        return stats;
    }

    bool inject(NeuronInput inp, int port) {
        try {
            port_in[port].push(inp);
            return true;
        } catch (...) {
            return false;
        }
    }

    NeuronInput detach(int port) {
        NeuronInput ni_cache{};
        port_out[port].pop(ni_cache);
        return ni_cache;
    }

    bool is_active() {
        for (auto &port: port_in) {
            if (!port.empty()) {
                return true;
            }
        }
        return false;
    }

    // ===== Single Step Execution Interface =====
    /**
     * @enum StepMode
     * @brief Neuron single step execution phase enumeration
     *
     * Defines the sequential phases of neuron computation in a single step.
     * Each phase represents a distinct computational task in the neuron lifecycle.
     */
    enum StepMode {
        STEP_MESSAGE_PROCESSING, // Message processing phase
        STEP_INPUT_PROCESSING, // Input processing phase
        STEP_INFERENCE, // Inference computation phase
        STEP_OUTPUT_BROADCAST, // Output broadcast phase
        STEP_MAINTENANCE // Maintenance tasks phase
    };

    /**
     * @brief Execute single step neuron computation.
     *
     * @details
     * Executes neuron computation in the following sequential phases:
     * 1. Message Processing - Handle incoming messages from other neurons
     * 2. Input Processing - Process data from input ports
     * 3. Inference - Perform GEMM/DRC inference calculations
     * 4. Output Broadcast - Send results to connected neurons
     * 5. Maintenance - Update internal state and perform housekeeping
     *
     * @return StepMode Current execution phase
     *
     * @note Executed on CUDA device, ensuring all operations are thread-safe
     * This function is the core computational unit of each neuron.
     */
    __device__ StepMode step() {
        StepMode current_step = STEP_MESSAGE_PROCESSING;

        // 1. Process message queue
        if (queue && !queue->empty()) {
            Message msg_cache{};
            if (queue->pop(msg_cache)) {
                processMessage(msg_cache);
                current_step = STEP_INPUT_PROCESSING;
            }
        }

        // 2. Process input port data
        if (current_step == STEP_INPUT_PROCESSING) {
            bool has_input = false;
            for (int p = 0; p < 4; p++) {
                if (!port_in[p].empty()) {
                    processUpdate(p);
                    has_input = true;
                }
            }
            if (has_input) {
                current_step = STEP_OUTPUT_BROADCAST;
            } else {
                current_step = STEP_MAINTENANCE;
            }
        }

        // 4. Broadcast output
        if (current_step == STEP_OUTPUT_BROADCAST) {
            broadcastOutput();

            // 更新卷积核
            for (int p = 0; p < 4; p++) {
                updateConvKernels(p);
            }
            current_step = STEP_MAINTENANCE;
        }

        // 5. Maintenance tasks
        if (current_step == STEP_MAINTENANCE) {
            cycle_counter++;

            // KFE decay (every 10 steps)
            if (cycle_counter % 10 == 0) {
                kfeDecay();
            }

            // Neuron discovery (every 100 steps)
            static int neuron_discover_countdown = 100;
            neuron_discover_countdown--;
            if (neuron_discover_countdown <= 0) {
                if (activity > 0.3 && output_conn_count < 1024) {
                    initiateFindNeuron();
                }
                neuron_discover_countdown = 100;
            }

            // Port transformation matrix update (every 50 steps)
            if (cycle_counter % 50 == 0) {
                updateMultiplexMatrices();
            }

            // Update activity
            updateActivity();
        }

        return current_step;
    }

    /**
     * @brief Determine whether to execute full GEMM inference.
     *
     * @details
     * Decides whether to trigger full GEMM inference based on multiple factors:
     * 1. Periodic heartbeat (every 16 steps) - Regular full computation
     * 2. High external input variation (high prediction error) - Significant changes require full processing
     * 3. High internal core vulnerability - Instability requires robust computation
     * 4. High short-term memory aggregate utility - Contextual knowledge importance
     *
     * @return bool true if full GEMM inference is needed, false for lightweight micro-correction
     *
     * This gating mechanism optimizes computational efficiency by:
     * - Using lightweight micro-corrections for stable states
     * - Triggering full GEMM computations when significant changes occur
     * - Balancing accuracy with performance through adaptive triggering
     */
    __device__ bool shouldTriggerGEMM() {
        // 周期心跳
        if (cycle_counter % 16 == 0) {
            return true;
        }

        // 外部高需求
        double deviation_norm = 0.0;
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                deviation_norm += Deviation[i][j] * Deviation[i][j];
            }
        }
        deviation_norm = sqrt(deviation_norm / (256.0 * 256.0));
        if (deviation_norm > 0.5) {
            return true;
        }

        // 内部危机
        if (core_vulnerability > 0.7) {
            return true;
        }

        // 内部注意力
        if (STM_aggregate_utility > 0.6) {
            return true;
        }

        return false;
    }

    /**
     * @brief 检查神经元是否有待处理的工作
     *
     * @details
     * 通过检查消息队列和输入端口来判断是否有待处理的工作项
     *
     * @return bool true表示有待处理的工作，false表示空闲
     */
    __device__ bool hasPendingWork() const {
        return (queue && !queue->empty()) || checkPortsForInput();
    }

    // 检查端口是否有输入
    __device__ bool checkPortsForInput() const {
        for (int p = 0; p < 4; p++) {
            if (!port_in[p].empty()) {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief 获取神经元的当前状态信息
     *
     * @param[out] act 当前活跃度
     * @param[out] cycles 运行周期计数器
     * @param[out] vulnerability 核心脆弱性指标
     * @param[out] utility STM聚合效用
     * @param[out] in_conn 输入连接数
     * @param[out] out_conn 输出连接数
     *
     * @note 所有输出参数必须在设备端可访问
     */
    __device__ void getState(double &act, int &cycles, double &vulnerability, double &utility,
                             int &in_conn, int &out_conn) const {
        act = activity;
        cycles = cycle_counter;
        vulnerability = core_vulnerability;
        utility = STM_aggregate_utility;
        in_conn = input_conn_count;
        out_conn = output_conn_count;
    }

    // 设置队列指针的方法
    void setQueuePointer(DeviceQueue<Message, 32> *queue_ptr) {
        queue = queue_ptr;
    }

    // 设置邻居队列指针的方法
    void setNeighbourQueuePointers(DeviceQueue<Message, 32> *queues[6]) {
        for (int i = 0; i < 6; i++) {
            neighbour_queues[i] = queues[i];
        }
    }

    // 重置指针为nullptr（用于序列化）
    void resetPointersForSerialization() {
        queue = nullptr;
        for (int i = 0; i < 6; i++) {
            neighbour_queues[i] = nullptr;
        }
        kfe_storage_queue = nullptr;
        kfe_query_queue = nullptr;
        kfe_result_queue = nullptr;
    }

    // 获取邻居队列指针（用于验证）
    DeviceQueue<Message, 32> *getNeighbourQueue(int index) const {
        if (index >= 0 && index < 6) {
            return neighbour_queues[index];
        }
        return nullptr;
    }

    // 获取主队列指针（用于验证）
    DeviceQueue<Message, 32> *getQueue() const {
        return queue;
    }

    __device__ void adjust_weights_rl(double delta) {
        // 更新核心权重矩阵
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                W_predict[i][j] += delta * P_Matrix[i][j];
                // 裁剪防止爆炸
                W_predict[i][j] = fmax(-2.0, fmin(2.0, W_predict[i][j]));
            }
        }
    }

    __device__ void apply_trace_update(double global_score, double learning_rate, double trace) {
        // 1. 计算局部梯度
        double local_gradient = trace * global_score * activity;

        // 2. 梯度裁剪
        local_gradient = fmax(-1.0, fmin(1.0, local_gradient));

        // 3. 自适应学习率
        double adaptive_lr = learning_rate / (1.0 + cycle_counter * 0.0001) * getLearningRate(3);

        // 4. 更新权重
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                W_predict[i][j] += adaptive_lr * local_gradient * P_Matrix[i][j];

                // 权重裁剪
                W_predict[i][j] = fmax(-2.0, fmin(2.0, W_predict[i][j]));
            }
        }

        // 5. Trace衰减
        trace *= 0.95;
    }

    ull getcs() { return cycle_counter; }

    void enable_training() { training = true; }
    void disable_training() { training = false; }

    void set_size(ull size) { GRID_SIZE = size; }

private:
    // ===== 随机数和基础状态 =====
    curandStatePhilox4_32_10_t rand_state{};
    double activity;
    ll local_coord[3]{0, 0, 0};
    bool training;
    ull GRID_SIZE;
    double global_lr = 0.01;
    double lr_schedule[4] = {1.0, 0.5, 0.3, 0.1};
    // KFE外部存储队列（通过消息队列与主机通信）
    DeviceQueue<KFE_STM_Slot, 32> *kfe_storage_queue; // 存储请求队列
    DeviceQueue<GPUString, 32> *kfe_query_queue; // 查询请求队列
    DeviceQueue<KFE_STM_Slot, 32> *kfe_result_queue; // 查询结果队列
    GPUVector<ExtKFE_Slot> ext_kfe_slots;
    GPUMutex ext_kfe_mutex, kfe_mutex;

    // ===== KFE Short-Term Memory =====
    // Local KFE (Knowledge Feature Encoding) short-term memory slots
    // Each neuron maintains 16 local KFE slots for contextual knowledge
    // These slots store compressed knowledge fragments for rapid access
    KFE_STM_Slot kfe_local[16]{};

    // ===== 消息队列系统 =====
    DeviceQueue<Message, 32> *queue{};
    DeviceQueue<Message, 32> *neighbour_queues[6]{}; // 6个方向的邻居

    // ===== Port System (4 Logical Ports) =====
    // Each neuron has 4 logical ports for input/output operations
    // This allows for multi-channel communication between neurons
    DeviceQueue<NeuronInput, 1024> port_in[4]{};
    DeviceQueue<NeuronInput, 1024> port_out[4]{};
    ll port_counts[4]{}; // 每个端口的连接数

    // ===== Connection Information =====
    // Stores connection details for input and output connections
    // Each neuron can have up to 2048 input and 2048 output connections
    ConnectionInfo input_conns[2048]{};
    ConnectionInfo output_conns[2048]{};
    int input_conn_count;
    int output_conn_count;

    // ===== Port Transformation Matrices =====
    // Input/output transformation matrices for each of the 4 ports
    // Used for feature transformation and mapping between ports
    double input_multiplex_array[256][256][4]{}; // 输入端口变换
    double output_multiplex_array[256][256][4]{}; // 输出端口变换

    // ===== GEMM/DRC Inference State =====
    // Core matrices for GEMM (General Matrix Multiply) and DRC (Dynamic Recalibration Correction) inference
    // P_Matrix: Current state matrix
    // P_stable: Stable prediction baseline
    // W_predict: Autoregressive weights
    // M_KFE: KFE knowledge context
    // Deviation: Prediction error
    // PS_aggregate: Neighbor consensus
    double P_Matrix[256][256]{}; // 意图矩阵(当前状态)
    double P_stable[256][256]{}; // 稳定预测(认知基线)
    double W_predict[256][256]{}; // 自回归权重
    double M_KFE[256][256]{}; // KFE知识上下文
    double Deviation[256][256]{}; // 预测误差
    half PS_aggregate[256][256]{}; // 邻居共识
    double h_state[256];
    half time_decay[256]; // 每个通道的衰减率
    half time_first[256]; // 时间优先度

    // ===== Gating and DRC History =====
    // Variables for controlling inference execution and maintaining history
    // cycle_counter: Tracks neuron execution cycles
    // core_vulnerability: Measures internal instability
    // STM_aggregate_utility: Short-term memory aggregate utility
    // P_history: Stores recent 5 rounds of state matrices
    // history_index: Current position in history buffer
    int cycle_counter;
    double core_vulnerability;
    double STM_aggregate_utility;
    half P_history[5][256][256]{}; // 最近5轮历史
    int history_index;
    MessageEncoder encoder{};
    MessageDecoder decoder{};
    double importance;

    // ===== XOR相关(备用) =====
    /*  __DEPRECATED__
    bool core_xor_array[2048][2048]{};
    double cor_xor_clip_array[2048][2048]{};
    */

    double noise;
    double learn;

    __device__ double getLearningRate(int update_type) {
        // 根据训练步数衰减
        double decay = 1.0 / (1.0 + cycle_counter * 0.0001);
        return global_lr * lr_schedule[update_type] * decay;
    }

    // ===== 添加卷积相关成员 =====
    ConvKernel input_conv_kernels[4][8]{}; // 每个端口8个卷积核
    ConvKernel output_conv_kernels[4][8]{}; // 输出端口卷积核
    double conv_feature_maps[4][8][32][32]{}; // 特征图(256/8=32)

    __device__ void sendAdaptiveMessage(const double data[256][256],
                                        ll to_coord[3]) {
        // 1. 决定压缩模式
        CompressionMode mode = CompressionDecider::decide(
            activity,
            core_vulnerability,
            importance,
            global_memory_pool.getFreeCount(),
            encoder.getAvgError()
        );

        // 2. 根据模式编码
        if (mode == MODE_FULL) {
            FullMessage msg{};
            memcpy(msg.to_coord, to_coord, sizeof(ll) * 3);
            msg.activity = activity;
            msg.type = 0;
            msg.weight = 1.0;
            encoder.encodeFull(data, msg);
            Message message{};
            memcpy(message.from_coord, local_coord, sizeof(ll) * 3);
            memcpy(message.to_coord, to_coord, sizeof(ll) * 3);
            message.adaptive_msg.full_msg = msg;
            message.activity = activity;
            message.type = NEURON_DATA;
            message.weight = computeImportance();
            message.compression_mode = MODE_FULL;
            route(message);
        } else if (mode == MODE_RESIDUAL) {
            ResidualMessage msg{};
            memcpy(msg.to_coord, to_coord, sizeof(ll) * 3);
            msg.activity = activity;
            msg.type = 0;
            msg.weight = 1.0;
            encoder.encodeResidual(data, msg);
            Message message{};
            memcpy(message.from_coord, local_coord, sizeof(ll) * 3);
            memcpy(message.to_coord, to_coord, sizeof(ll) * 3);
            message.adaptive_msg.res_msg = msg;
            message.activity = activity;
            message.type = NEURON_DATA;
            message.weight = computeImportance();
            message.compression_mode = MODE_RESIDUAL;
            route(message);
        } else if (mode == MODE_CONV_ONLY) {
            ConvMessage msg{};
            memcpy(msg.to_coord, to_coord, sizeof(ll) * 3);
            msg.activity = activity;
            msg.type = 0;
            msg.weight = 1.0;
            encoder.encodeConv(data, msg);
            Message message{};
            memcpy(message.from_coord, local_coord, sizeof(ll) * 3);
            memcpy(message.to_coord, to_coord, sizeof(ll) * 3);
            message.adaptive_msg.conv_msg = msg;
            message.activity = activity;
            message.type = NEURON_DATA;
            message.weight = computeImportance();
            message.compression_mode = MODE_CONV_ONLY;
            route(message);
        }
    }

    __device__ void receiveAdaptiveMessage(CompressionMode mode,
                                           void *msg_ptr,
                                           double output[256][256]) {
        if (mode == MODE_FULL) {
            FullMessage *msg = (FullMessage *) msg_ptr;
            decoder.decodeFull(*msg, output);
            // 记得释放内存块
            global_memory_pool.release(msg->pool_block_id);
        } else if (mode == MODE_RESIDUAL) {
            ResidualMessage *msg = (ResidualMessage *) msg_ptr;
            decoder.decodeResidual(*msg, output);
        } else if (mode == MODE_CONV_ONLY) {
            ConvMessage *msg = (ConvMessage *) msg_ptr;
            decoder.decodeConv(*msg, output);
        }
    }

    // ===== 初始化函数 =====
    /**
     * @brief Initialize neuron matrix state
     *
     * @details
     * Initialize the following matrices:
     * - P_Matrix: Intent matrix (small random values)
     * - P_stable: Stable prediction matrix
     * - W_predict: Autoregressive weight matrix
     * - M_KFE: KFE knowledge context matrix
     * - Deviation: Prediction error matrix
     * - PS_aggregate: Neighbor consensus matrix
     * - Port transformation matrices: Initially variant of identity matrix
     * - DRC history matrices
     *
     * @note Initialized using random number generator to ensure values are within reasonable range
     */
    __device__ void initializeMatrices() {
        // 初始化P_Matrix为小随机值
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                P_Matrix[i][j] = randomInRange(-0.1, 0.1);
                P_stable[i][j] = P_Matrix[i][j];
                W_predict[i][j] = (i == j) ? 0.9 : randomInRange(-0.05, 0.05);
                M_KFE[i][j] = 0.0;
                Deviation[i][j] = 0.0;
                PS_aggregate[i][j] = 0.0;
            }
        }

        // 初始化端口变换矩阵(初始为单位阵的变体)
        for (int p = 0; p < 4; p++) {
            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    if (i == j) {
                        input_multiplex_array[i][j][p] = 1.0;
                        output_multiplex_array[i][j][p] = 1.0;
                    } else {
                        input_multiplex_array[i][j][p] = randomInRange(-0.01, 0.01);
                        output_multiplex_array[i][j][p] = randomInRange(-0.01, 0.01);
                    }
                }
            }
        }

        // 初始化DRC历史
        for (int h = 0; h < 5; h++) {
            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    P_history[h][i][j] = 0;
                }
            }
        }
    }

    // ===== 卷积操作 =====
    /**
     * @brief Perform 8×8 convolution operation (stride=8, non-overlapping)
     *
     * @param[in] input 256×256 input matrix
     * @param[in] kernel 8×8 convolution kernel
     * @param[out] output 32×32 output feature map
     *
     * @details
     * Downsample 256×256 input to 32×32 feature map through 8×8 convolution kernel:
     * - Stride of 8, no overlap
     * - Apply ReLU activation function
     * - Add bias term
     *
     * @note Ensure input and convolution kernel memory alignment for performance optimization
     */
    __device__ void conv2d_8x8(const double input[256][256],
                               const ConvKernel &kernel,
                               double output[32][32]) {
        // 256×256 → 32×32 (stride=8, no padding)
        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 32; j++) {
                double sum = 0.0;

                // 卷积计算
                for (int ki = 0; ki < 8; ki++) {
                    for (int kj = 0; kj < 8; kj++) {
                        int input_i = i * 8 + ki;
                        int input_j = j * 8 + kj;
                        sum += input[input_i][input_j] * kernel.kernel[ki][kj];
                    }
                }

                // 加bias和ReLU激活
                output[i][j] = fmax(0.0, sum + kernel.bias);
            }
        }
    }

    /**
     * @brief Perform deconvolution operation (upsampling)
     *
     * @param[in] input 32×32 input feature map
     * @param[in] kernel 8×8 convolution kernel
     * @param[out] output 256×256 output matrix
     *
     * @details
     * Upsample 32×32 feature map to 256×256 output:
     * - Use transposed convolution operation
     * - Output size is 8 times the input size
     * - Initialize output matrix to 0
     *
     * @note Deconvolution is the inverse operation of convolution, used for feature reconstruction
     */
    __device__ void deconv2d_8x8(const double input[32][32],
                                 const ConvKernel &kernel,
                                 double output[256][256]) {
        // 32×32 → 256×256
        memset(output, 0, sizeof(double) * 256 * 256);

        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 32; j++) {
                double value = input[i][j];

                // 将值分散到8×8区域
                for (int ki = 0; ki < 8; ki++) {
                    for (int kj = 0; kj < 8; kj++) {
                        int output_i = i * 8 + ki;
                        int output_j = j * 8 + kj;
                        output[output_i][output_j] += value * kernel.kernel[ki][kj];
                    }
                }
            }
        }
    }

    /**
     * @brief Extract input features using multi-kernel convolution
     *
     * @param[in] port Port index (0-3)
     * @param[in] input 256×256 input matrix
     *
     * @details
     * Extract multi-scale features of input using 8 different convolution kernels:
     * - Each convolution kernel extracts different types of features
     * - Results stored in conv_feature_maps
     * - Supports multi-level representation of features
     *
     * @note Feature extraction is a prerequisite step for inference computation
     */
    __device__ void extractConvFeatures(int port, const double input[256][256]) {
        // 使用8个卷积核提取不同特征
        for (int k = 0; k < 8; k++) {
            conv2d_8x8(input, input_conv_kernels[port][k],
                       conv_feature_maps[port][k]);
        }
    }

    // ===== 特征聚合(替代简单的matmul) =====
    __device__ void aggregateFeatures(int port, double output[256][256]) {
        // 将8个特征图反卷积并加权融合
        double temp_outputs[8][256][256];

        for (int k = 0; k < 8; k++) {
            deconv2d_8x8(conv_feature_maps[port][k],
                         input_conv_kernels[port][k],
                         temp_outputs[k]);
        }

        // 加权融合
        memset(output, 0, sizeof(double) * 256 * 256);
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                for (int k = 0; k < 8; k++) {
                    output[i][j] += temp_outputs[k][i][j] / 8.0;
                }
            }
        }
    }

    // ===== 卷积核更新(反向传播) =====
    __device__ void updateConvKernels(int port) {
        double learning_rate = getLearningRate(1);

        // 计算梯度(简化版,实际应该是完整的BP)
        for (int k = 0; k < 8; k++) {
            for (int ki = 0; ki < 8; ki++) {
                for (int kj = 0; kj < 8; kj++) {
                    // 使用特征图和误差计算梯度
                    double grad = 0.0;
                    for (int i = 0; i < 32; i++) {
                        for (int j = 0; j < 32; j++) {
                            // 简化梯度:特征值×偏差
                            grad += conv_feature_maps[port][k][i][j] *
                                    Deviation[i * 8 + ki][j * 8 + kj];
                        }
                    }

                    // 更新权重
                    input_conv_kernels[port][k].kernel[ki][kj] -=
                            learning_rate * grad / (32.0 * 32.0);
                }
            }

            // 更新bias
            double bias_grad = 0.0;
            for (int i = 0; i < 32; i++) {
                for (int j = 0; j < 32; j++) {
                    bias_grad += conv_feature_maps[port][k][i][j];
                }
            }
            input_conv_kernels[port][k].bias -=
                    learning_rate * bias_grad / (32.0 * 32.0);
        }
    }

    // ===== Message Processing =====
    /**
     * @brief Process received messages.
     *
     * @param[in] msg Message to be processed
     *
     * @details
     * Performs different operations based on message type:
     * - NEURON_DATA: Route or receive data messages
     * - FIND_NEURON: Forward or reply to connection requests
     * - REPLY_NEURON_FIND: Process connection replies
     *
     * @note Message processing is the core communication functionality of neurons
     * Handles all inter-neuron communication and network topology discovery.
     */
    __device__ void processMessage(const Message &msg) {
        if (msg.type == NEURON_DATA) {
            // 数据消息:路由或接收
            if (msg.to_coord[0] == local_coord[0] &&
                msg.to_coord[1] == local_coord[1] &&
                msg.to_coord[2] == local_coord[2]) {
                receiveMessages(msg);
            } else {
                route(msg);
            }
        } else if (msg.type == FIND_NEURON) {
            // 连接请求:转发或回复
            if (msg.remains > 1) {
                Message msg_forward = msg;
                msg_forward.remains--;
                sendMessage(msg_forward, randomULLInRange(0, 6));
            }

            // 回复连接请求
            Message msg_reply{};
            memcpy(msg_reply.last_proxy_coord, local_coord, 3 * sizeof(ll));
            memcpy(msg_reply.from_coord, local_coord, 3 * sizeof(ll));
            memcpy(msg_reply.to_coord, msg.from_coord, 3 * sizeof(ll));
            msg_reply.activity = activity;
            msg_reply.type = REPLY_NEURON_FIND;
            route(msg_reply);

            // 添加为输入连接(分配到负载最小的端口)
            if (input_conn_count < 2048) {
                ll min_port = port_counts[0];
                int min_port_index = 0;
                for (int i = 1; i < 4; i++) {
                    if (port_counts[i] < min_port) {
                        min_port = port_counts[i];
                        min_port_index = i;
                    }
                }

                input_conns[input_conn_count].port = min_port_index;
                input_conns[input_conn_count].inout = true;
                memcpy(input_conns[input_conn_count].coord, msg.from_coord, 3 * sizeof(ll));
                port_counts[min_port_index]++;
                input_conn_count++;
            }
        } else if (msg.type == REPLY_NEURON_FIND) {
            // 连接回复:路由或接受
            if (msg.to_coord[0] == local_coord[0] &&
                msg.to_coord[1] == local_coord[1] &&
                msg.to_coord[2] == local_coord[2]) {
                // 添加为输出连接
                if (output_conn_count < 2048) {
                    ll min_port = port_counts[0];
                    int min_port_index = 0;
                    for (int i = 1; i < 4; i++) {
                        if (port_counts[i] < min_port) {
                            min_port = port_counts[i];
                            min_port_index = i;
                        }
                    }

                    output_conns[output_conn_count].port = min_port_index;
                    output_conns[output_conn_count].inout = false;
                    memcpy(output_conns[output_conn_count].coord, msg.from_coord, 3 * sizeof(ll));
                    port_counts[min_port_index]++;
                    output_conn_count++;
                }
            } else {
                route(msg);
            }
        }
    }

    /**
     * @brief Message routing algorithm in 3D space
     *
     * @param[in] msg Message to be routed
     *
     * @details
     * Greedy routing strategy based on 3D coordinates:
     * - Compare target coordinates with local coordinates
     * - Select direction closest to target
     * - Support six directions: ±X, ±Y, ±Z
     *
     * @note Routing algorithm ensures efficient message transmission in 3D network
     */
    __device__ void route(Message msg) {
        if (msg.to_coord[0] > local_coord[0]) {
            sendMessage(msg, 0); // +X方向
        } else if (msg.to_coord[0] < local_coord[0]) {
            sendMessage(msg, 1); // -X方向
        } else if (msg.to_coord[1] > local_coord[1]) {
            sendMessage(msg, 2); // +Y方向
        } else if (msg.to_coord[1] < local_coord[1]) {
            sendMessage(msg, 3); // -Y方向
        } else if (msg.to_coord[2] > local_coord[2]) {
            sendMessage(msg, 4); // +Z方向
        } else if (msg.to_coord[2] < local_coord[2]) {
            sendMessage(msg, 5); // -Z方向
        }
    }

    __device__ void sendMessage(const Message &msg, int direction) {
        if (direction >= 0 && direction < 6 && neighbour_queues[direction]) {
            neighbour_queues[direction]->push(msg);
        }
    }

    // ===== 接收消息并分配到端口 =====
    __device__ void receiveMessages(Message msg) {
        for (int i = 0; i < input_conn_count; i++) {
            if (input_conns[i].coord[0] == msg.from_coord[0] &&
                input_conns[i].coord[1] == msg.from_coord[1] &&
                input_conns[i].coord[2] == msg.from_coord[2]) {
                NeuronInput cache_inp{};
                cache_inp.activity = msg.activity;
                cache_inp.weight = msg.weight;
                double value[256][256]{};
                switch (msg.compression_mode) {
                    case MODE_FULL:
                        receiveAdaptiveMessage(msg.compression_mode, &msg.adaptive_msg.full_msg, cache_inp.array);
                        break;
                    case MODE_RESIDUAL:
                        receiveAdaptiveMessage(msg.compression_mode, &msg.adaptive_msg.res_msg, cache_inp.array);
                        break;
                    case MODE_CONV_ONLY:
                        receiveAdaptiveMessage(msg.compression_mode, &msg.adaptive_msg.conv_msg, cache_inp.array);
                        break;
                    default: ;
                }
                memcpy(cache_inp.from_coord, msg.from_coord, 3 * sizeof(ll));

                port_in[input_conns[i].port].push(cache_inp);
                break;
            }
        }
    }

    __device__ void addPositionalEncoding() {
        // pos = 神经元在3D空间的"序列位置"
        ll pos = local_coord[0] * GRID_SIZE * GRID_SIZE +
                 local_coord[1] * GRID_SIZE +
                 local_coord[2];

        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                int d = i * 256 + j; // 维度索引

                // Sinusoidal位置编码
                double freq = 1.0 / pow(10000.0, 2.0 * d / 65536.0);

                if (d % 2 == 0) {
                    // 偶数维度用sin
                    P_Matrix[i][j] += 0.1 * sin(pos * freq);
                } else {
                    // 奇数维度用cos
                    P_Matrix[i][j] += 0.1 * cos(pos * freq);
                }
            }
        }
    }

    // ===== Core Inference Update =====
    /**
     * @brief Execute core inference update computation.
     *
     * @param[in] port Input port index
     *
     * @details
     * Executes the complete inference computation flow:
     * 1. Aggregate neighbor inputs from all ports
     * 2. Compute prediction error and KFE attention
     * 3. Gating decision to execute GEMM or micro-correction
     * 4. Broadcast output and update convolution kernels
     *
     * @note This is the core computational function of neurons
     * Integrates inputs, performs reasoning, and generates outputs.
     */
    __device__ void processUpdate(int port) {
        if (port_in[port].empty()) return;

        NeuronInput curr_inp;
        port_in[port].pop(curr_inp);

        // === 阶段1: 聚合所有端口的邻居输入 ===
        double weight_sum = 0.0;

        // 重置聚合矩阵
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                PS_aggregate[i][j] = 0.0;
            }
        }

        // 从4个端口收集并加权平均
        for (int p = 0; p < 4; p++) {
            if (!port_in[p].empty()) {
                NeuronInput temp_inp = port_in[p].front();

                // 先通过input_multiplex_array变换输入
                double transformed_input[256][256];
                matmul_double(&temp_inp.array[0][0], &input_multiplex_array[0][0][p],
                              &transformed_input[0][0]);

                // 提取卷积特征
                extractConvFeatures(p, transformed_input);

                double score = 0.0;
                for (int i = 0; i < 256; i++) {
                    for (int j = 0; j < 256; j++) {
                        // Q = P_Matrix (当前状态)
                        // K = all_inputs[k] (输入)
                        score += P_Matrix[i][j] * transformed_input[i][j];
                    }
                }
                score /= sqrt(256.0 * 256.0);

                // 聚合特征图
                double aggregated[256][256];
                aggregateFeatures(p, aggregated);

                double w = temp_inp.weight * temp_inp.activity;
                weight_sum += w;

                for (int i = 0; i < 256; i++) {
                    double wkv = 0.0;
                    double state = h_state[i]; // 复用SSM的状态
                    for (int j = 0; j < 256; j++) {
                        double k = PS_aggregate[i][j]; // key
                        double v = PS_aggregate[i][j]; // value

                        // 时间衰减
                        double w = -exp(time_decay[i]);

                        // WKV计算
                        wkv += exp(__half2float(time_first[i]) + k) * v;
                        state = state * exp(w) + exp(k) * v;
                        PS_aggregate[i][j] += transformed_input[i][j] * w * aggregated[i][j] * score + wkv / (
                            wkv + state);
                    }
                }
            }
        }

        // 归一化
        if (weight_sum > 1e-6) {
            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    PS_aggregate[i][j] /= weight_sum;
                }
            }
        }

        // 计算预测误差(意外性)
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                Deviation[i][j] = __half2float(PS_aggregate[i][j]) - P_stable[i][j];
            }
        }

        selectiveSSM();

        // === 阶段2: KFE注意力计算 ===
        STM_aggregate_utility = computeKFEAttention();

        // === 阶段3: 门控判断 ===
        bool trigger_gemm = false;

        // 条件1: 周期心跳
        if (cycle_counter % 16 == 0) {
            trigger_gemm = true;
        }

        // 条件2: 外部高需求(输入变化大)
        double deviation_norm = 0.0;
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                deviation_norm += Deviation[i][j] * Deviation[i][j];
            }
        }
        deviation_norm = sqrt(deviation_norm / (256.0 * 256.0));
        if (deviation_norm > 0.5) {
            trigger_gemm = true;
        }

        // 条件3: 内部危机
        if (core_vulnerability > 0.7) {
            trigger_gemm = true;
        }

        // 条件4: 内部注意力
        if (STM_aggregate_utility > 0.6) {
            trigger_gemm = true;
        }

        if (trigger_gemm) {
            executeGEMMAndDRC();
        } else {
            executeMicroCorrection();
        }


        // === 阶段4: 广播输出 ===
        broadcastOutput();

        for (int p = 0; p < 4; p++) {
            updateConvKernels(p);
        }
    }

    /**
     * @brief Compute attention weights for KFE short-term memory.
     *
     * @return double STM aggregate utility value
     *
     * @details
     * Computes correlation between current prediction error and KFE memory fragments:
     * - Iterates through 16 local KFE slots
     * - Computes dot product and applies Sigmoid activation function
     * - Updates KFE statistics for adaptive learning
     * - Supports external KFE queries when local memory is insufficient
     *
     * @note Attention mechanism determines influence of knowledge fragments on current reasoning
     * KFE (Knowledge Feature Encoding) provides contextual knowledge for enhanced inference.
     *
     * The function performs the following operations:
     * 1. Calculates attention weights for each local KFE slot
     * 2. Updates utility, importance, and volatility metrics
     * 3. Queries external KFE storage when local capacity is low
     * 4. Aggregates attention-weighted knowledge into M_KFE matrix
     */
    __device__ double computeKFEAttention() {
        double utility = 0.0;
        memset(M_KFE, 0, sizeof(M_KFE));

        double usum = 0.0;
        double isum = 0.0;
        double vsum = 0.0;

        for (int k = 0; k < 16; k++) {
            if (kfe_local[k].Icore < 0.01) continue;

            // KFE块与Deviation的点积(相关性)
            double dot_product = 0.0;
            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    dot_product += kfe_local[k].Vmem[i][j] * Deviation[i][j];
                }
            }

            usum += kfe_local[k].Ulocal;
            isum += kfe_local[k].Icore;
            vsum += kfe_local[k].V;

            // Sigmoid激活并累加到M_KFE
            double attention_weight = 1.0 / (1.0 + exp(-dot_product));
            double weighted_attention = attention_weight * kfe_local[k].Icore;

            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    M_KFE[i][j] += weighted_attention * kfe_local[k].Vmem[i][j];
                }
            }

            utility += weighted_attention;

            kfe_mutex.lock();
            // 更新KFE槽位统计
            kfe_local[k].Ulocal += 0.01 * (attention_weight - kfe_local[k].Ulocal);
            kfe_local[k].Icore += 0.01 * (0.5 - kfe_local[k].Icore);
            kfe_local[k].V -= 0.01 * (1.0 - kfe_local[k].V);
            kfe_local[k].Rcycles = cycle_counter;
            kfe_mutex.unlock();
        }

        if ((usum < 4 && isum < 6) || vsum > 12) {
            double curr_conv16[16][16] = {};
            ConvResidualProcessor::conv2d_16x16(W_predict, ConvKernel16(), curr_conv16);
            double max_sim = 0.0;
            ull max_index = 0;
            for (int k = 0; k < ext_kfe_slots.size(); k++) {
                auto &i = ext_kfe_slots[k];
                double sim = cosineSimilarity16x16(i.conv16, curr_conv16);
                if (sim > max_sim) {
                    max_sim = sim;
                    max_index = k;
                }
            }
            KFE_STM_Slot ext_kfe_pulled{};
            // 通过队列查询外部KFE
            if (kfe_query_queue && kfe_result_queue) {
                kfe_query_queue->push(ext_kfe_slots[max_index].hash.data());
                // 尝试获取查询结果
                if (kfe_result_queue->pop(ext_kfe_pulled)) {
                    // 成功获取到外部KFE槽位
                } else {
                    // 如果获取失败，使用默认值
                    ext_kfe_pulled = {};
                }
            }
            // KFE块与Deviation的点积(相关性)
            double dot_product = 0.0;
            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    dot_product += ext_kfe_pulled.Vmem[i][j] * Deviation[i][j];
                }
            }
            // Sigmoid激活并累加到M_KFE
            double attention_weight = 1.0 / (1.0 + exp(-dot_product));
            double weighted_attention = attention_weight * ext_kfe_pulled.Icore;

            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    M_KFE[i][j] += weighted_attention * ext_kfe_pulled.Vmem[i][j];
                }
            }

            utility += weighted_attention;
        }

        return utility / 16.0;
    }

    double gelu(double x) {
        return 0.5 * x * (1.0 + tanh(0.797885 * (x + 0.044715 * x * x * x)));
    }

    __device__ void selectiveSSM() {
        // 1. 输入投影
        double B[256], C[256]; // 输入门和输出门

        for (int i = 0; i < 256; i++) {
            // 根据输入决定如何更新状态
            double input_i = 0.0;
            for (int j = 0; j < 256; j++) {
                input_i += __half2float(PS_aggregate[i][j]);
            }
            input_i /= 256.0;

            // 选择性门控 (类似LSTM的forget gate)
            B[i] = gelu(input_i); // 记忆多少输入
            C[i] = gelu(-input_i); // 输出多少状态
        }

        // 2. 状态更新 (线性递归)
        for (int i = 0; i < 256; i++) {
            // Δ = B × input + decay × h_old
            double delta = B[i] * __half2float(PS_aggregate[i][0]);
            h_state[i] = 0.9 * h_state[i] + delta;
        }

        // 3. 输出投影
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                P_Matrix[i][j] += C[i] * h_state[i];
            }
        }
    }

    __device__ bool shouldUseFullAttention() {
        // 重要任务用Transformer的完整注意力
        // 一般任务用RWKV的线性注意力
        return (core_vulnerability > 0.7) ||
               (STM_aggregate_utility > 0.6);
    }

    void predictNoise(double input[256][256], double output[256][256]) {
        // 多尺度特征提取层
        double scale1[256][256]; // 细节特征
        double scale2[256][256]; // 中等尺度特征
        double scale3[256][256]; // 粗糙特征

        // 第一层：提取边缘和细节特征 (3x3卷积)
        for (int i = 1; i < 255; i++) {
            for (int j = 1; j < 255; j++) {
                // Sobel边缘检测核
                double gx = -input[i - 1][j - 1] - 2 * input[i][j - 1] - input[i + 1][j - 1] +
                            input[i - 1][j + 1] + 2 * input[i][j + 1] + input[i + 1][j + 1];
                double gy = -input[i - 1][j - 1] - 2 * input[i - 1][j] - input[i - 1][j + 1] +
                            input[i + 1][j - 1] + 2 * input[i + 1][j] + input[i + 1][j + 1];
                scale1[i][j] = sqrt(gx * gx + gy * gy) * 0.1; // 缩放因子
            }
        }

        // 第二层：提取中等尺度特征 (5x5高斯平滑)
        for (int i = 2; i < 254; i++) {
            for (int j = 2; j < 254; j++) {
                double sum = 0.0;
                // 5x5高斯核近似
                sum += 1 * input[i - 2][j - 2] + 4 * input[i - 2][j - 1] + 6 * input[i - 2][j] + 4 * input[i - 2][j + 1]
                        + 1 * input[i - 2][j + 2];
                sum += 4 * input[i - 1][j - 2] + 16 * input[i - 1][j - 1] + 24 * input[i - 1][j] + 16 * input[i - 1][
                    j + 1] + 4 * input[i - 1][j + 2];
                sum += 6 * input[i][j - 2] + 24 * input[i][j - 1] + 36 * input[i][j] + 24 * input[i][j + 1] + 6 * input[
                    i][j + 2];
                sum += 4 * input[i + 1][j - 2] + 16 * input[i + 1][j - 1] + 24 * input[i + 1][j] + 16 * input[i + 1][
                    j + 1] + 4 * input[i + 1][j + 2];
                sum += 1 * input[i + 2][j - 2] + 4 * input[i + 2][j - 1] + 6 * input[i + 2][j] + 4 * input[i + 2][j + 1]
                        + 1 * input[i + 2][j + 2];
                scale2[i][j] = sum / 256.0; // 归一化
            }
        }

        // 第三层：提取粗糙特征 (9x9平均池化)
        for (int i = 4; i < 252; i++) {
            for (int j = 4; j < 252; j++) {
                double sum = 0.0;
                for (int di = -4; di <= 4; di++) {
                    for (int dj = -4; dj <= 4; dj++) {
                        sum += input[i + di][j + dj];
                    }
                }
                scale3[i][j] = sum / 81.0; // 9x9=81
            }
        }

        // 边界填充
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                if (i < 1 || i >= 255 || j < 1 || j >= 255) {
                    scale1[i][j] = 0.0;
                }
                if (i < 2 || i >= 254 || j < 2 || j >= 254) {
                    scale2[i][j] = input[i][j];
                }
                if (i < 4 || i >= 252 || j < 4 || j >= 252) {
                    scale3[i][j] = input[i][j];
                }
            }
        }

        // 特征融合层
        for (int i = 4; i < 252; i++) {
            for (int j = 4; j < 252; j++) {
                // 加权融合多尺度特征
                double fused = 0.4 * scale1[i][j] + 0.3 * scale2[i][j] + 0.3 * scale3[i][j];

                // 使用Swish激活函数: x * sigmoid(x)
                double sigmoid_val = 1.0 / (1.0 + exp(-fused));
                output[i][j] = fused * sigmoid_val;
            }
        }

        // 注意力机制：突出重要的噪声区域
        double attention_map[256][256];
        double mean_val = 0.0;

        // 计算平均值
        for (int i = 4; i < 252; i++) {
            for (int j = 4; j < 252; j++) {
                mean_val += fabs(output[i][j]);
            }
        }
        mean_val /= (248 * 248);

        // 生成注意力图
        for (int i = 4; i < 252; i++) {
            for (int j = 4; j < 252; j++) {
                // 对比度增强
                double diff = fabs(output[i][j]) - mean_val;
                attention_map[i][j] = 1.0 + tanh(diff * 2.0); // 放大差异
            }
        }

        // 应用注意力权重
        for (int i = 4; i < 252; i++) {
            for (int j = 4; j < 252; j++) {
                output[i][j] *= attention_map[i][j];
            }
        }

        // 残差连接：保留原始输入信息
        for (int i = 4; i < 252; i++) {
            for (int j = 4; j < 252; j++) {
                output[i][j] += 0.1 * input[i][j]; // 弱连接
            }
        }

        // Final smoothing processing
        double temp[256][256];
        for (int i = 5; i < 251; i++) {
            for (int j = 5; j < 251; j++) {
                // 3x3平均滤波
                double sum = 0.0;
                for (int di = -1; di <= 1; di++) {
                    for (int dj = -1; dj <= 1; dj++) {
                        sum += output[i + di][j + dj];
                    }
                }
                temp[i][j] = sum / 9.0;
            }
        }

        // Copy results
        for (int i = 5; i < 251; i++) {
            for (int j = 5; j < 251; j++) {
                output[i][j] = temp[i][j];
            }
        }

        // Boundary processing
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 5; j++) {
                output[i][j] = output[i][5];
                output[i][255 - j] = output[i][250];
                output[j][i] = output[5][i];
                output[255 - j][i] = output[250][i];
            }
        }
    }

    /**
     * @brief Execute GEMM inference and DRC iterative correction.
     *
     * @details
     * Complete inference computation workflow:
     * 1. GEMM core inference (P_Matrix × W_predict)
     * 2. Compute fixed target T_fixed
     * 3. 16 rounds of DRC iterative correction
     * 4. Synchronize state and update core vulnerability
     *
     * @note This is the most computationally intensive part, using GELU activation and momentum correction
     *
     * GEMM (General Matrix Multiply) performs core neural computations.
     * DRC (Dynamic Recalibration Correction) iteratively refines results for accuracy.
     *
     * The function performs:
     * - Matrix multiplication with learned weights
     * - Knowledge context integration from KFE
     * - GELU activation for non-linear transformation
     * - Iterative refinement with attention modulation
     * - Historical momentum for stable learning
     */
    __device__ void executeGEMMAndDRC() {
        addPositionalEncoding();
        // === Step 1: GEMM core inference ===
        double P_Next[256][256];
        double temp_product[256][256];

        double P_Original[256][256];
        memcpy(&P_Original, &P_Matrix, sizeof(P_Matrix));

        double W_backup[256][256];
        if (training) {
            memcpy(W_backup, W_predict, sizeof(W_predict));
            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    if (curand_uniform(&rand_state) < 0.05) {
                        W_predict[i][j] = 0.0;
                    }
                }
            }
        }

        // P_Matrix × W_predict
        matmul_double(&P_Matrix[0][0], &W_predict[0][0], &temp_product[0][0]);

        // Add KFE context and apply GELU activation
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                double x = temp_product[i][j] + M_KFE[i][j];
                // GELU activation
                P_Next[i][j] = 0.5 * x * (1.0 + tanh(0.797885 * (x + 0.044715 * x * x * x)));
            }
        }

        // === Step 2: Compute fixed target T_fixed ===
        double T_fixed[256][256];
        double alpha = 0.7;

        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                T_fixed[i][j] = alpha * __half2float(PS_aggregate[i][j]) +
                                (1.0 - alpha) * P_Next[i][j];
            }
        }

        // === Step 3: 16 rounds of DRC iterative correction ===
        double P_current[256][256];
        memcpy(P_current, P_Next, sizeof(P_current));

        double epsilon = 1e-4;
        double eta_base = 0.1;
        double lambda = 0.9;

        double prev_diff_norm = 0.0;

        for (int iter = 0; iter < 16; iter++) {
            double P_new[256][256];

            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    // 1. 基础修正项
                    double V_corr = (T_fixed[i][j] - P_current[i][j]) * eta_base;

                    // 2. 局部注意力调制
                    double local_feature = 0.0;
                    for (int p = 0; p < 4; p++) {
                        if (!port_in[p].empty()) {
                            NeuronInput temp = port_in[p].front();
                            local_feature += temp.array[i][j];
                        }
                    }
                    local_feature /= 4.0;

                    double attn_weight = 1.0 / (1.0 + exp(-(local_feature * P_current[i][j])));
                    double M_attn = attn_weight * V_corr;

                    // 3. 历史动量项
                    double V_hist = 0.0;
                    if (iter > 0) {
                        for (int h = 1; h <= min(iter, 3); h++) {
                            int hist_idx = (history_index - h + 5) % 5;
                            int prev_idx = (hist_idx - 1 + 5) % 5;
                            double delta = P_history[hist_idx][i][j] -
                                           P_history[prev_idx][i][j];
                            V_hist += pow(lambda, h) * delta;
                        }
                    }

                    // 组合修正
                    P_new[i][j] = P_current[i][j] + V_corr + M_attn + V_hist;
                }
            }

            // 检查收敛
            double diff_norm = 0.0;
            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    double diff = P_new[i][j] - P_current[i][j];
                    diff_norm += diff * diff;
                }
            }
            diff_norm = sqrt(diff_norm);

            // 更新历史
            history_index = (history_index + 1) % 5;
            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    P_history[history_index][i][j] = __float2half(P_current[i][j]);
                }
            }
            memcpy(P_current, P_new, sizeof(P_current));

            // 早停
            if (diff_norm < epsilon) {
                break;
            }
            if (iter > 8 && diff_norm > prev_diff_norm) {
                // 开始震荡,停止
                break;
            }
            prev_diff_norm = diff_norm;
        }

        double beta_schedule[16]; // 噪声调度

        // 余弦调度 (类似Improved DDPM)
        for (int t = 0; t < 16; t++) {
            double alpha_t = cos(PI * t / 32.0);
            beta_schedule[t] = 1.0 - alpha_t * alpha_t;
        }

        double P_Nsc[256][256];
        memcpy(&P_Nsc, &P_Original, sizeof(P_Nsc));

        // 迭代去噪
        for (int t = 15; t >= 0; t--) {
            // 反向扩散
            double beta = beta_schedule[t];

            // 预测噪声
            double noise_pred[256][256];
            predictNoise(P_Nsc, noise_pred);

            // 去噪一步
            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    P_Nsc[i][j] -= sqrt(beta) * noise_pred[i][j];
                    P_Nsc[i][j] = (P_Nsc[i][j] -
                                   sqrt(beta) * noise_pred[i][j]) /
                                  sqrt(1.0 - beta);
                }
            }
        }

        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                constexpr double alpha_c = 0.7;
                P_current[i][j] += alpha_c * P_current[i][j] +
                        (1 - alpha_c) * P_Nsc[i][j];
                P_current[i][j] /= 2;
            }
        }

        // === 步骤4: 同步状态 ===
        memcpy(P_Matrix, P_current, sizeof(P_Matrix));
        memcpy(P_stable, P_current, sizeof(P_stable));

        updateCoreVulnerability();

        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                if (training && W_predict[i][j] < 0.01) {
                    W_predict[i][j] = W_backup[i][j];
                }
            }
        }

        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                P_Matrix[i][j] = 0.9 * P_Matrix[i][j] + 0.1 * P_Original[i][j];
            }
        }

        layerNorm(&P_Matrix[0][0], 256 * 256);
    }

    /**
     * @brief Execute low-cost micro-correction computation
     *
     * @details
     * Lightweight inference correction:
     * - Linear interpolation based on current state and neighbor consensus
     * - Lower learning rate (0.05)
     * - Suitable for minor adjustments in stable states
     *
     * @note Micro-correction has less computational load than full GEMM, suitable for frequent execution
     */
    __device__ void executeMicroCorrection() {
        double alpha = 0.3;
        double eta_micro = 0.05 + max(min(getLearningRate(3), 1.0), 0.0) * 0.0001;

        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                double T_micro = alpha * __half2float(PS_aggregate[i][j]) +
                                 (1.0 - alpha) * P_Matrix[i][j];
                P_Matrix[i][j] += eta_micro * (T_micro - P_Matrix[i][j]);
                P_Matrix[i][j] += max(min(noise, 1.0), 0.0) * 0.0001 * (randomInRange(0, 1) - 0.5);
            }
        }
    }

    // 在Neuron类中
    __device__ double computeImportance() {
        double importance = 0.0;

        // 因素1: 核心脆弱性(40%权重)
        // 系统不稳定时,消息更重要
        importance += core_vulnerability * 0.4;

        // 因素2: 活跃度(30%权重)
        // 高活跃神经元的输出更重要
        importance += activity * 0.3;

        // 因素3: 预测误差(20%权重)
        // 误差大说明有重要信息
        double deviation_norm = 0.0;
        for (int i = 0; i < 256; i += 16) {
            for (int j = 0; j < 256; j += 16) {
                deviation_norm += Deviation[i][j] * Deviation[i][j];
            }
        }
        deviation_norm = sqrt(deviation_norm / 256.0);
        importance += fmin(deviation_norm, 1.0) * 0.2;

        // 因素4: 连接数(10%权重)
        // 连接多的神经元是枢纽,消息重要
        double conn_ratio = (double) output_conn_count / 2048.0;
        importance += conn_ratio * 0.1;

        return fmin(importance, 1.0);
    }

    // ===== 广播输出 =====
    __device__ void broadcastOutput() {
        for (int out_idx = 0; out_idx < output_conn_count; out_idx++) {
            Message out_msg;
            memcpy(out_msg.from_coord, local_coord, sizeof(local_coord));
            memcpy(out_msg.to_coord, output_conns[out_idx].coord, sizeof(ll) * 3);
            memcpy(out_msg.last_proxy_coord, local_coord, sizeof(local_coord));

            int port = output_conns[out_idx].port;

            // 通过output_multiplex_array变换输出
            double output_temp[256][256];
            matmul_double(&P_Matrix[0][0], &output_multiplex_array[0][0][port],
                          &output_temp[0][0]);

            if (computeImportance() > 0.7 && activity > 0.3 && core_vulnerability > 0.3) {
                out_msg.compression_mode = MODE_FULL;
                out_msg.adaptive_msg.full_msg = FullMessage{};
                encoder.encodeFull(output_temp, (out_msg.adaptive_msg.full_msg));
            } else if (computeImportance() > 0.4 && activity > 0.2) {
                out_msg.compression_mode = MODE_RESIDUAL;
                out_msg.adaptive_msg.res_msg = ResidualMessage{};
                encoder.encodeResidual(output_temp, (out_msg.adaptive_msg.res_msg));
            } else {
                out_msg.compression_mode = MODE_CONV_ONLY;
                out_msg.adaptive_msg.full_msg = FullMessage{};
                encoder.encodeConv(output_temp, (out_msg.adaptive_msg.conv_msg));
            }
            out_msg.activity = activity;
            out_msg.weight = 1.0;
            out_msg.type = NEURON_DATA;

            route(out_msg);
        }
    }

    // ===== 更新核心脆弱性 =====
    __device__ void updateCoreVulnerability() {
        double deviation_sum = 0.0;
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                double diff = P_Matrix[i][j] - P_stable[i][j];
                deviation_sum += diff * diff;
            }
        }

        core_vulnerability = sqrt(deviation_sum / (256.0 * 256.0));
        core_vulnerability = tanh(core_vulnerability); // 归一化到[0,1]
    }

    // ===== 更新活跃度 =====
    __device__ void updateActivity() {
        // 基于最近输入的平均活跃度
        double total_activity = 0.0;
        int count = 0;

        for (int p = 0; p < 4; p++) {
            if (!port_in[p].empty()) {
                NeuronInput temp = port_in[p].front();
                total_activity += temp.activity;
                count++;
            }
        }

        if (count > 0) {
            double new_activity = total_activity / count;
            activity = activity * 0.9 + new_activity * 0.1; // 指数移动平均
        } else {
            activity *= 0.95; // 无输入时衰减
        }
    }

    // ===== KFE衰减 =====
    __device__ void kfeDecay() {
        GPUMutexGuard lock(&kfe_mutex);
        for (int k = 0; k < 16; k++) {
            // 效用衰减
            kfe_local[k].Ulocal *= 0.95;

            // 周期老化
            int age = cycle_counter - kfe_local[k].Rcycles;
            if (age > 100) {
                kfe_local[k].Icore *= 0.9;
            }

            // 清除无效槽位
            if (kfe_local[k].Ulocal < 0.01 && kfe_local[k].Icore < 0.01) {
                kfe_local[k] = {};
            }

            //Save important Slot to persistence slot and clear local slot
            if (kfe_local[k].V == 0.0) {
                if (kfe_storage_queue) {
                    kfe_storage_queue->push(kfe_local[k]);
                    GPUMutexGuard lock_ext(&ext_kfe_mutex);
                    ExtKFE_Slot cache_ext;
                    memcpy(cache_ext.conv16, kfe_local[k].conv16, sizeof(kfe_local[k].conv16));
                    cache_ext.hash = kfe_local[k].hash();
                    cache_ext.importance = computeImportance();
                    cache_ext.last_access_time = 0.0;
                    ext_kfe_slots.push_back(cache_ext);
                    kfe_local[k] = {0.0, 0, 0.0, 0.0, {}};
                }
            }
        }
    }

    // ===== KFE更新(当有重要事件时调用) =====
    __device__ void kfeUpdate(double importance) {
        GPUMutexGuard lock(&kfe_mutex);
        // 如果重要性不够,不记录
        if (importance < 0.3) return;

        // 找到最不重要的槽位
        int target_slot = findLeastUsefulSlot();

        // 如果当前槽位已经很重要,可能不替换
        double current_value = kfe_local[target_slot].Ulocal *
                               kfe_local[target_slot].Icore *
                               (100.0 - (cycle_counter - kfe_local[target_slot].Rcycles));

        if (current_value > importance * 50.0) {
            return; // 不替换
        }

        // 记录新的知识片段
        kfe_local[target_slot].Ulocal = importance;
        kfe_local[target_slot].Rcycles = cycle_counter;
        kfe_local[target_slot].Icore = importance;
        kfe_local[target_slot].V = 1.0;
        kfe_local[target_slot].conv();

        // 保存当前的Deviation作为知识片段
        memcpy(kfe_local[target_slot].Vmem, Deviation, sizeof(Deviation));
    }

    __device__ int findLeastUsefulSlot() {
        int min_index = 0;
        double min_value = 999999999.0;

        for (int i = 0; i < 16; i++) {
            double value = kfe_local[i].Ulocal * kfe_local[i].Icore *
                           (100.0 - (cycle_counter - kfe_local[i].Rcycles)) *
                           kfe_local[i].V;

            if (value < min_value) {
                min_value = value;
                min_index = i;
            }
            if (kfe_local[i].Ulocal < 0.01 && kfe_local[i].Icore < 0.01) {
                // 空槽位优先使用
                return i;
            }
        }
        if (kfe_local[min_index].Ulocal > 0.5 && kfe_local[min_index].Icore > 0.5) {
            // 如果最小槽位还很重要,则通过队列存储到外部
            GPUMutexGuard lock(&ext_kfe_mutex);
            if (kfe_storage_queue && kfe_local[min_index].Ulocal > 0.5 && kfe_local[min_index].Icore > 0.5) {
                kfe_storage_queue->push(kfe_local[min_index]);
                // 存储成功后清空本地槽位
                kfe_local[min_index] = {0.0, 0, 0.0, 0.0, {}};
            }
        }
        return min_index;
    }

    // ===== 发起神经元发现 =====
    __device__ void initiateFindNeuron() {
        // 发送探索消息到随机方向
        int num_explore = 5; // 发送5个探索消息

        for (int i = 0; i < num_explore; i++) {
            Message explore_msg{};
            memcpy(explore_msg.from_coord, local_coord, sizeof(local_coord));
            memcpy(explore_msg.last_proxy_coord, local_coord, sizeof(local_coord));

            // 随机目标坐标(在附近区域)
            explore_msg.to_coord[0] = local_coord[0] + randomInRange(-10, 10);
            explore_msg.to_coord[1] = local_coord[1] + randomInRange(-10, 10);
            explore_msg.to_coord[2] = local_coord[2] + randomInRange(-10, 10);

            explore_msg.type = FIND_NEURON;
            explore_msg.remains = 5; // 最多转发5跳
            explore_msg.activity = activity;

            // 发送到随机方向
            int direction = randomULLInRange(0, 6);
            sendMessage(explore_msg, direction);
        }
    }

    // ===== 端口变换矩阵更新(Hebbian学习) =====
    __device__ void updateMultiplexMatrices() {
        double learning_rate = getLearningRate(2);

        // 对每个端口进行Hebbian更新
        for (int p = 0; p < 4; p++) {
            if (port_in[p].empty()) continue;

            NeuronInput inp = port_in[p].front();

            // Hebbian规则: ΔW = η * input * output
            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    // 输入变换矩阵更新
                    double delta_in = learning_rate * inp.array[i][j] * P_Matrix[i][j];
                    input_multiplex_array[i][j][p] += delta_in * (1.0 + max(min(learning_rate, 1.0), 0.0) * 0.001) +
                            noise *
                            0.0001 * (randomInRange(0, 1) - 0.5);

                    // 输出变换矩阵更新(对称)
                    double delta_out = learning_rate * P_Matrix[i][j] * inp.array[i][j];
                    output_multiplex_array[i][j][p] += delta_out * (1.0 + max(min(learning_rate, 1.0), 0.0) * 0.001) +
                            noise *
                            0.0001 * (randomInRange(0, 1) - 0.5);

                    // 防止权重爆炸
                    input_multiplex_array[i][j][p] = fmax(-2.0, fmin(2.0, input_multiplex_array[i][j][p]));
                    output_multiplex_array[i][j][p] = fmax(-2.0, fmin(2.0, output_multiplex_array[i][j][p]));
                }
            }
        }
    }

    __device__ void layerNorm(double *data, int size) {
        double mean = 0.0, var = 0.0;
        for (int i = 0; i < size; i++) mean += data[i];
        mean /= size;

        for (int i = 0; i < size; i++) var += (data[i] - mean) * (data[i] - mean);
        var /= size;

        double std = sqrt(var + 1e-6);
        for (int i = 0; i < size; i++) {
            data[i] = (data[i] - mean) / std;
        }
    }

    // ===== 单步推理函数 =====
    // 输入: Message (to_coord必须等于local_coord)
    // 输出: Message (包含推理结果)
    // 不处理路由，只做推理计算
    __device__ Message stepInference(Message input_msg) {
        // 1. 处理输入消息
        processMessage(input_msg);

        // 2. 处理端口输入并执行推理
        bool processed_input = false;
        for (int p = 0; p < 4; p++) {
            if (!port_in[p].empty()) {
                processUpdate(p);
                processed_input = true;
            }
        }

        // 3. 如果没有输入，也执行一次推理
        if (!processed_input) {
            if (shouldTriggerGEMM()) {
                executeGEMMAndDRC();
            } else {
                executeMicroCorrection();
            }
        }

        // 4. 更新活跃度
        updateActivity();

        // 5. 准备输出消息
        Message output_msg{};
        memcpy(output_msg.from_coord, local_coord, sizeof(local_coord));
        memcpy(output_msg.to_coord, input_msg.from_coord, sizeof(ll) * 3);
        memcpy(output_msg.last_proxy_coord, local_coord, sizeof(local_coord));

        output_msg.activity = activity;
        output_msg.weight = computeImportance();
        output_msg.type = NEURON_DATA;

        // 选择压缩模式
        if (computeImportance() > 0.7 && activity > 0.3 && core_vulnerability > 0.3) {
            output_msg.compression_mode = MODE_FULL;
            output_msg.adaptive_msg.full_msg = FullMessage{};
            encoder.encodeFull(P_Matrix, output_msg.adaptive_msg.full_msg);
        } else if (computeImportance() > 0.4 && activity > 0.2) {
            output_msg.compression_mode = MODE_RESIDUAL;
            output_msg.adaptive_msg.res_msg = ResidualMessage{};
            encoder.encodeResidual(P_Matrix, output_msg.adaptive_msg.res_msg);
        } else {
            output_msg.compression_mode = MODE_CONV_ONLY;
            output_msg.adaptive_msg.conv_msg = ConvMessage{};
            encoder.encodeConv(P_Matrix, output_msg.adaptive_msg.conv_msg);
        }

        // 6. 维护任务
        cycle_counter++;
        if (cycle_counter % 10 == 0) kfeDecay();
        if (cycle_counter % 50 == 0) updateMultiplexMatrices();

        return output_msg;
    }

    // ===== 简化版本：直接处理输入数据 =====
    // 输入: 256x256数据矩阵
    // 输出: Message (包含推理结果)
    __device__ Message stepInferenceWithData(const double input_data[256][256], ll from_coord[3]) {
        // 1. 创建输入消息
        Message input_msg{};
        memcpy(input_msg.from_coord, from_coord, sizeof(ll) * 3);
        memcpy(input_msg.to_coord, local_coord, sizeof(ll) * 3);
        memcpy(input_msg.last_proxy_coord, from_coord, sizeof(ll) * 3);
        input_msg.activity = 1.0;
        input_msg.weight = 1.0;
        input_msg.type = NEURON_DATA;
        input_msg.compression_mode = MODE_FULL;

        // 2. 解码输入数据
        NeuronInput temp_input{};
        memcpy(temp_input.array, input_data, sizeof(double) * 256 * 256);
        temp_input.activity = 1.0;
        temp_input.weight = 1.0;
        memcpy(temp_input.from_coord, from_coord, sizeof(ll) * 3);

        // 3. 注入到端口0
        port_in[0].push(temp_input);

        // 4. 执行推理
        return stepInference(input_msg);
    }

    // ===== 最简版本：只执行一次推理 =====
    // 输出: 当前状态矩阵 (256x256)
    __device__ void stepInferenceOnly(double output_data[256][256]) {
        // 执行推理计算
        if (shouldTriggerGEMM()) {
            executeGEMMAndDRC();
        } else {
            executeMicroCorrection();
        }

        // 更新活跃度
        updateActivity();

        // 复制输出
        memcpy(output_data, P_Matrix, sizeof(double) * 256 * 256);

        // 维护任务
        cycle_counter++;
        if (cycle_counter % 10 == 0) kfeDecay();
        if (cycle_counter % 50 == 0) updateMultiplexMatrices();
    }
};


__global__ void all_neurons_kernel(Neuron *neurons, bool *active_flags, ull count) {
    ull tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count && active_flags[tid]) {
        try {
            neurons[tid].step();
        } catch (...) {
            // 捕获所有异常，防止CUDA崩溃
        }
    }
}

__global__ void update_activity(Neuron *neurons, bool *active_flags, double *trace, double score, bool training) {
    ull tid = blockIdx.x * blockDim.x + threadIdx.x;
    active_flags[tid] = neurons[tid].is_active();
    if (active_flags[tid] && training) {
        trace[tid] += neurons[tid].get_activity() * 0.1 * max(min(score, 1.0), 0.0);
    } else {
        trace[tid] -= 0.01;
    }
}

__global__ void reset_trace(double *trace) {
    ull tid = blockIdx.x * blockDim.x + threadIdx.x;
    trace[tid] = 0;
}

__global__ void apply_trace_to_neurons(
    Neuron *neurons,
    double *trace,
    double global_score,
    ull count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    if (trace[tid] > 0.01) {
        double activity = neurons[tid].get_activity();

        // Policy Gradient: ∇J = trace × (R - baseline)
        double baseline = 0.5; // 可以用移动平均
        double advantage = global_score - baseline;

        // 最终梯度
        double gradient = trace[tid] * advantage * activity;

        // 自适应学习率
        double learning_rate = 0.001 / (1.0 + neurons[tid].getcs() * 0.0001);

        // 更新权重
        neurons[tid].adjust_weights_rl(gradient * learning_rate);

        neurons[tid].set_noise(1 - trace[tid]);

        // Trace衰减
        trace[tid] *= 0.95;
    }
}
#endif //SRC_NEURON_H
