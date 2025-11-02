#pragma once
/**
 * @file Neuron.cu
 * @brief Implements a single neuron in a 3D neural network.
 *
 * This file contains the core logic of the neuron, including:
 * - 3D spatial positioning and neighbor connections.
 * - Adaptive message compression and routing.
 * - Short-term memory (KFE) system.
 * - Hybrid convolution and GEMM inference.
 * - Multi-port input-output system.
 */

#ifndef SRC_NEURON_H
#define SRC_NEURON_H

#include <iostream>
#include "deviceQueue.cpp"
#include "matrixMultiplex.cu"
#include <curand_kernel.h>
#include "cern.cuh"
#include "isw.hpp"
#include "conv16_res_msg.cuh"
#include <cmath>
#include <sm_20_intrinsics.h>
#include <vector>
#include "hasher.h"
#include "structs.h"
#include "sim.cu"
#include "GPUMutex.cu"
#include <cuda_fp16.h>
#include "gpu_containers.cuh"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "nlohmann/json.hpp"

#define ll long long
#define ull unsigned ll
#define retpc reinterpret_cast
#define stpc static_cast
#undef atomicAdd

__device__ double d_ema_baseline = 0.0;
__constant__ double ema_beta = 0.9;


__global__ void aggregateNeuronInputsShared(
    const half *inputs, // [num_inputs, 256]
    const half *weights, // [num_inputs]
    half *output, // [256]
    int num_inputs
) {
    __shared__ half shared_partial[32][256];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Phase 1: Warp-level reduction (like per-head reduction in Attention)
    half local_sum = 0;
    for (int i = warp_id; i < num_inputs; i += blockDim.x / 32) {
        int idx = i * 256 + lane_id;
        if (lane_id < 256) {
            local_sum = __hfma(inputs[idx], weights[i], local_sum);
        }
    }

    // Store to shared memory
    if (lane_id < 256) {
        shared_partial[warp_id][lane_id] = local_sum;
    }
    __syncthreads();

    // Phase 2: Final reduction across warps
    if (warp_id == 0 && lane_id < 256) {
        half sum = 0;
#pragma unroll
        for (int w = 0; w < 32; ++w) {
            sum = __hadd(sum, shared_partial[w][lane_id]);
        }
        output[lane_id] = sum;
    }
}

static __global__ void deconv2d_8x8_kernel(const double input[32][32],
                                           const ConvKernel &kernel,
                                           double output[256][256]) {
    int base_i = blockIdx.y * 8;
    int base_j = blockIdx.x * 8;
    int local_i = threadIdx.y; // 0-7
    int local_j = threadIdx.x; // 0-7

    __shared__ double s_kernel[8][8];
    __shared__ double s_input[32][32];

    if (local_i < 8 && local_j < 8) {
        s_kernel[local_i][local_j] = kernel.kernel[local_i][local_j];
    }

    if (local_i < 32 && local_j < 32) {
        s_input[local_i][local_j] = input[local_i][local_j];
    }

    __syncthreads();

    int output_i = base_i + local_i;
    int output_j = base_j + local_j;

    if (output_i < 256 && output_j < 256) {
        int input_i = output_i / 8;
        int ki = output_i % 8;
        int input_j = output_j / 8;
        int kj = output_j % 8;
        if (input_i < 32 && input_j < 32 && ki < 8 && kj < 8) {
            output[output_i][output_j] = s_input[input_i][input_j] * s_kernel[ki][kj];
        }
    }
}

static __global__ void conv2d_8x8_kernel(const double input[256][256],
                                         const ConvKernel &kernel,
                                         double output[32][32]) {
    int output_i = blockIdx.y;
    int output_j = blockIdx.x;
    if (output_i >= 32 || output_j >= 32) return;
    int ki = threadIdx.y;
    int kj = threadIdx.x;

    __shared__ double s_kernel[8][8];
    __shared__ double s_bias;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_bias = kernel.bias;
    }
    if (ki < 8 && kj < 8) {
        s_kernel[ki][kj] = kernel.kernel[ki][kj];
    }
    __syncthreads();
    if (ki < 8 && kj < 8) {
        int input_i = output_i * 8 + ki;
        int input_j = output_j * 8 + kj;
        if (input_i < 256 && input_j < 256) {
            double contribution = input[input_i][input_j] * s_kernel[ki][kj];
            atomicAdd(&output[output_i][output_j], contribution);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int oi = 0; oi < 32; oi++) {
            for (int oj = 0; oj < 32; oj++) {
                double val = output[oi][oj] + s_bias;
                output[oi][oj] = fmax(0.0, val);
            }
        }
    }
}

__global__ void onlineSoftmaxShared(
    half *attention_scores, // [seq_len, seq_len]
    const int seq_len
) {
    __shared__ float shared_max[32];
    __shared__ float shared_sum[32];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Phase 1: Find max (numerically stable)
    float local_max = -INFINITY;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float val = __half2float(attention_scores[row * seq_len + i]);
        local_max = fmaxf(local_max, val);
    }

    // Warp reduce max
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    if (tid % 32 == 0) {
        shared_max[tid / 32] = local_max;
    }
    __syncthreads();

    // Final max reduction
    if (tid < 32) {
        local_max = shared_max[tid];
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        }
        if (tid == 0) shared_max[0] = local_max;
    }
    __syncthreads();

    float max_val = shared_max[0];

    // Phase 2: Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float val = __half2float(attention_scores[row * seq_len + i]);
        float exp_val = expf(val - max_val);
        attention_scores[row * seq_len + i] = __float2half(exp_val);
        local_sum += exp_val;
    }

    // Warp reduce sum
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (tid % 32 == 0) {
        shared_sum[tid / 32] = local_sum;
    }
    __syncthreads();

    // Final sum
    if (tid < 32) {
        local_sum = shared_sum[tid];
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
        if (tid == 0) shared_sum[0] = local_sum;
    }
    __syncthreads();

    // Phase 3: Normalize
    float sum_inv = 1.0f / shared_sum[0];
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float val = __half2float(attention_scores[row * seq_len + i]);
        attention_scores[row * seq_len + i] = __float2half(val * sum_inv);
    }
}

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
            {"port_counts", port_counts},
            {"core_vul", core_vul},
            {"importance", importance}
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
    Neuron() = default;

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
    __host__ __device__ Neuron(DeviceQueue<Message, 32> *queues[6], ll coord[3], ull seed,
                               DeviceQueue<Message, 32> *queue_ptr,
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

#ifdef __CUDA_ARCH__
        // Device-specific initialization
        // Initialize random number generator
        curand_init(seed, 0, 0, &rand_state);
#else
        // Host-specific initialization
        // Initialize random number generator for host (using std)
        srand(seed);
#endif

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
        // Note: This needs to be modified because port_in and port_out are of type DeviceQueue.
        // We need to iterate through all elements in the queue.
        // Simplified here; proper traversal of DeviceQueue should be implemented.
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

    // 主机端版本的save方法，用于主机/设备数据传输
    NeuronData host_save() {
        NeuronData data{};
        // 拷贝端口计数
        memcpy(data.port_counts, port_counts, sizeof(port_counts));

        // 拷贝连接信息
        memcpy(data.input_conns, input_conns, sizeof(input_conns));
        memcpy(data.output_conns, output_conns, sizeof(output_conns));
        data.input_conn_count = input_conn_count;
        data.output_conn_count = output_conn_count;

        // 拷贝变换矩阵
        memcpy(data.input_multiplex_array, input_multiplex_array, sizeof(input_multiplex_array));
        memcpy(data.output_multiplex_array, output_multiplex_array, sizeof(output_multiplex_array));

        // 拷贝推理状态矩阵
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
            // 注意：port_in 和 port_out 是设备队列，不能直接在主机端操作
            // 我们只需要拷贝数据字段，队列操作应该在设备端进行

            // 拷贝端口计数
            memcpy(port_counts, data.port_counts, sizeof(port_counts));

            // 拷贝连接信息
            memcpy(input_conns, data.input_conns, sizeof(input_conns));
            memcpy(output_conns, data.output_conns, sizeof(output_conns));
            input_conn_count = data.input_conn_count;
            output_conn_count = data.output_conn_count;

            // 拷贝其他数据
            activity = data.activity;
            training = data.training;
            learn = data.learn;
            noise = data.noise;
            core_vulnerability = data.core_vulnerability;
            importance = data.importance;
            training_interval = data.training_interval;
            training_count = data.training_count;
            last_training_time = data.last_training_time;

            return true;
        } catch (...) {
            return false;
        }
    }

    __device__ bool load_device(NeuronData data) {
        // 设备版本的load函数，不使用异常处理
        // 注意：port_in 和 port_out 是设备队列，不能直接在主机端操作
        // 我们只需要拷贝数据字段，队列操作应该在设备端进行

        // 拷贝端口计数
        memcpy(port_counts, data.port_counts, sizeof(port_counts));

        // 拷贝连接信息
        memcpy(input_conns, data.input_conns, sizeof(input_conns));
        memcpy(output_conns, data.output_conns, sizeof(output_conns));
        input_conn_count = data.input_conn_count;
        output_conn_count = data.output_conn_count;

        // 拷贝其他数据
        activity = data.activity;
        training = data.training;
        learn = data.learn;
        noise = data.noise;
        core_vulnerability = data.core_vulnerability;
        importance = data.importance;
        training_interval = data.training_interval;
        training_count = data.training_count;
        last_training_time = data.last_training_time;

        // 拷贝变换矩阵
        memcpy(input_multiplex_array, data.input_multiplex_array, sizeof(input_multiplex_array));
        memcpy(output_multiplex_array, data.output_multiplex_array, sizeof(output_multiplex_array));

        // 拷贝推理状态矩阵
        memcpy(P_Matrix, data.P_Matrix, sizeof(P_Matrix));
        memcpy(P_stable, data.P_stable, sizeof(P_stable));
        memcpy(W_predict, data.W_predict, sizeof(W_predict));
        memcpy(M_KFE, data.M_KFE, sizeof(M_KFE));
        memcpy(Deviation, data.Deviation, sizeof(Deviation));
        memcpy(PS_aggregate, data.PS_aggregate, sizeof(PS_aggregate));

        return true;
    }

    [[nodiscard]] double get_noise() const { return noise; }
    [[nodiscard]] double get_learn_rt() const { return learn; }

    __host__ __device__ void set_noise(double new_ns) {
        noise = new_ns;
    }

    __host__ __device__ void set_learn_rt(double new_rt) {
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

    static __host__ __device__ double randomInRange(double min, double max) {
#ifdef __CUDA_ARCH__
        return curand_uniform_double(&rand_state) * (max - min) + min;
#else
        // Host implementation using standard random
        // 简单的线性同余生成器实现
        static unsigned int seed = 1;
        seed = seed * 1103515245 + 12345;
        double normalized = stpc<double>(seed % 1000000) / 1000000.0;
        return min + normalized * (max - min);
#endif
    }

    __device__ ull randomULLInRange(ull min, ull max) {
        return curand(&rand_state) % (max - min) + min;
    }

    [[nodiscard]] __host__ __device__ double get_activity() const {
        return activity;
    }

    __host__ __device__ NeuronStats get_stats() {
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

    __device__ bool inject(NeuronInput inp, int port) {
        // 使用设备端push函数
        return port_in[port].push(inp);
    }

    NeuronInput detach(int port) {
        NeuronInput ni_cache{};
        port_out[port].h_pop(ni_cache);
        return ni_cache;
    }

    __host__ __device__ bool is_active() {
        for (auto &port: port_in) {
            if (!port.h_empty()) {
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

            // Update convolution kernels
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

    void setQueuePointer(DeviceQueue<Message, 32> *queue_ptr) {
        queue = queue_ptr;
    }

    void setNeighbourQueuePointers(DeviceQueue<Message, 32> *queues[6]) {
        for (int i = 0; i < 6; i++) {
            neighbour_queues[i] = queues[i];
        }
    }

    void resetPointersForSerialization() {
        queue = nullptr;
        for (int i = 0; i < 6; i++) {
            neighbour_queues[i] = nullptr;
        }
        kfe_storage_queue = nullptr;
        kfe_query_queue = nullptr;
        kfe_result_queue = nullptr;
    }

    DeviceQueue<Message, 32> *getNeighbourQueue(int index) const {
        if (index >= 0 && index < 6) {
            return neighbour_queues[index];
        }
        return nullptr;
    }

    DeviceQueue<Message, 32> *getQueue() const {
        return queue;
    }

    __device__ void adjust_weights_rl(double delta) {
        update_count++;
        const ull t = update_count;
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                const double core = P_Matrix[i][j] * Deviation[i][j];
                const double aux = 0.3 * stpc<double>(__half2float(PS_aggregate[i][j])) + 0.3 * M_KFE[i][j] + 0.4
                                   * STM_aggregate_utility;
                const double product = delta * core * aux;
                const double scaled = pow(product, 1.0 / 3);
                const double g = max(min(delta * scaled, 1.0), 0.0);

                m[i][j] = beta1 * m[i][j] + (1 - beta1) * g;
                v[i][j] = beta2 * v[i][j] + (1 - beta2) * g * g;

                const double m_hat = m[i][j] / (1.0 - pow(beta1, t));
                const double v_hat = v[i][j] / (1.0 - pow(beta2, t));

                const double update = m_hat / (sqrt(v_hat) + eps);

                W_predict[i][j] += update;
                W_predict[i][j] = fmax(-2.0, fmin(2.0, W_predict[i][j]));
            }
        }
    }

    __device__ void apply_trace_update(double global_score, double learning_rate, double trace) {
        double local_gradient = trace * global_score * activity;
        local_gradient = fmax(-1.0, fmin(1.0, local_gradient));
        double adaptive_lr = learning_rate / (1.0 + cycle_counter * 0.0001) * getLearningRate(3);
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                W_predict[i][j] += adaptive_lr * local_gradient * P_Matrix[i][j];
                W_predict[i][j] = fmax(-2.0, fmin(2.0, W_predict[i][j]));
            }
        }
        trace *= 0.95;
    }

    __host__ __device__ ull getcs() { return cycle_counter; }

    void enable_training() { training = true; }
    void disable_training() { training = false; }

    void set_size(ull size) { GRID_SIZE = size; }

    // ===== Adam Optimizer State =====
    double m[256][256]{};
    double v[256][256]{};
    ull update_count;
    size_t weight_dim;
    const double beta1 = 0.9;
    const double beta2 = 0.999;
    const double eps = 1e-8;

public:
    // ===== Random number&basic states =====
    curandStatePhilox4_32_10_t rand_state{};
    double activity;
    ll local_coord[3]{0, 0, 0};
    bool training;
    ull GRID_SIZE;
    double global_lr = 0.01;
    double lr_schedule[4] = {1.0, 0.5, 0.3, 0.1};
    // KFE External Storage queue pointers;Communication between device&host
    DeviceQueue<KFE_STM_Slot, 32> *kfe_storage_queue; // storage requests queue
    DeviceQueue<GPUString, 32> *kfe_query_queue; // storage query queue
    DeviceQueue<KFE_STM_Slot, 32> *kfe_result_queue; // query result queue
    GPUVector<ExtKFE_Slot> ext_kfe_slots;
    GPUMutex ext_kfe_mutex, kfe_mutex;

    // ===== KFE Short-Term Memory =====
    // Local KFE (Knowledge Feature Encoding) short-term memory slots
    // Each neuron maintains 16 local KFE slots for contextual knowledge
    // These slots store compressed knowledge fragments for rapid access
    KFE_STM_Slot kfe_local[16]{};

    // ===== Message queue system =====
    DeviceQueue<Message, 32> *queue{};
    DeviceQueue<Message, 32> *neighbour_queues[6]{};

    // ===== Port System (4 Logical Ports) =====
    // Each neuron has 4 logical ports for input/output operations
    // This allows for multichannel communication between neurons
    DeviceQueue<NeuronInput, 1024> port_in[4]{};
    DeviceQueue<NeuronInput, 1024> port_out[4]{};
    ll port_counts[4]{};

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
    double input_multiplex_array[4][256][256]{};
    double output_multiplex_array[4][256][256]{};

    // ===== GEMM/DRC Inference State =====
    // Core matrices for GEMM (General Matrix Multiply) and DRC (Dynamic Recalibration Correction) inference
    // P_Matrix: Current state matrix
    // P_stable: Stable prediction baseline
    // W_predict: Autoregressive weights
    // M_KFE: KFE knowledge context
    // Deviation: Prediction error
    // PS_aggregate: Neighbor consensus
    double P_Matrix[256][256]{};
    double P_stable[256][256]{};
    double W_predict[256][256]{};
    double Deviation[256][256]{};
    half PS_aggregate[256][256]{};
    double M_KFE[256][256]{};
    double h_state[256];
    half time_decay[256];
    half time_first[256];

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
    half P_history[5][256][256]{};
    int history_index;
    MessageEncoder encoder{};
    MessageDecoder decoder{};
    double importance;

    // ===== XOR array(deprecated) =====
    /*  __DEPRECATED__
    bool core_xor_array[2048][2048]{};
    double cor_xor_clip_array[2048][2048]{};
    */

    double noise;
    double learn;

    // ===== 训练相关变量 =====
    int training_interval;
    int training_count;
    int last_training_time;

    __device__ double getLearningRate(int update_type) {
        double decay = 1.0 / (1.0 + cycle_counter * 0.0001);
        return global_lr * lr_schedule[update_type] * decay;
    }

    // ===== Conv states =====
    ConvKernel input_conv_kernels[4][8]{};
    ConvKernel output_conv_kernels[4][8]{};
    double conv_feature_maps[4][8][32][32]{};

    __device__ void sendAdaptiveMessage(const double data[256][256],
                                        ll to_coord[3]) {
        CompressionMode mode = CompressionDecider::decide(
            activity,
            core_vulnerability,
            importance,
            global_memory_pool.getFreeCount(),
            encoder.getAvgError()
        );

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
            const auto msg = stpc<FullMessage *>(msg_ptr);
            decoder.decodeFull(*msg, output);
            global_memory_pool.release(msg->pool_block_id);
        } else if (mode == MODE_RESIDUAL) {
            const auto *msg = stpc<ResidualMessage *>(msg_ptr);
            decoder.decodeResidual(*msg, output);
        } else if (mode == MODE_CONV_ONLY) {
            const auto *msg = stpc<ConvMessage *>(msg_ptr);
            decoder.decodeConv(*msg, output);
        }
    }

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
    __host__ __device__ void initializeMatrices() {
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
        for (int h = 0; h < 5; h++) {
            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    P_history[h][i][j] = 0;
                }
            }
        }
    }

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
    static __device__ void conv2d_8x8(const double input[256][256],
                                      const ConvKernel &kernel,
                                      double output[32][32]) {
        /*
        // 256×256 → 32×32 (stride=8, no padding)
        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 32; j++) {
                double sum = 0.0;

                // Convolution calculation
                for (int ki = 0; ki < 8; ki++) {
                    for (int kj = 0; kj < 8; kj++) {
                        int input_i = i * 8 + ki;
                        int input_j = j * 8 + kj;
                        sum += input[input_i][input_j] * kernel.kernel[ki][kj];
                    }
                }

                // Add bias and ReLU activation
                output[i][j] = fmax(0.0, sum + kernel.bias);
            }
        }
        */
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        dim3 grid(32, 32);
        dim3 block(8, 8);
        conv2d_8x8_kernel<<<grid, block, 0, stream>>>(input, kernel, output);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
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
    static __device__ __host__ void deconv2d_8x8(const double input[32][32],
                                                 const ConvKernel &kernel,
                                                 double output[256][256]) {
        /*
        // 32×32 → 256×256
        memset(output, 0, sizeof(double) * 256 * 256);

        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 32; j++) {
                double value = input[i][j];
                for (int ki = 0; ki < 8; ki++) {
                    for (int kj = 0; kj < 8; kj++) {
                        int output_i = i * 8 + ki;
                        int output_j = j * 8 + kj;
                        output[output_i][output_j] += value * kernel.kernel[ki][kj];
                    }
                }
            }
        }
        */
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        dim3 block(8, 8);
        dim3 grid(32, 32);
        deconv2d_8x8_kernel<<<grid, block, 0, stream>>>(
            input,
            kernel,
            output);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
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
        for (int k = 0; k < 8; k++) {
            conv2d_8x8(input, input_conv_kernels[port][k],
                       conv_feature_maps[port][k]);
        }
    }

    // ===== 特征聚合(替代简单的matmul) =====
    __device__ void aggregateFeatures(int port, double output[256][256]) const {
        // 将8个特征图反卷积并加权融合
        double temp_outputs[8][256][256];

        for (int k = 0; k < 8; k++) {
            deconv2d_8x8(conv_feature_maps[port][k],
                         input_conv_kernels[port][k],
                         temp_outputs[k]);
        }
        memset(output, 0, sizeof(double) * 256 * 256);
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                for (int k = 0; k < 8; k++) {
                    output[i][j] += temp_outputs[k][i][j] / 8.0;
                }
            }
        }
    }

    __device__ void updateConvKernels(int port) {
        double learning_rate = getLearningRate(1);

        for (int k = 0; k < 8; k++) {
            for (int ki = 0; ki < 8; ki++) {
                for (int kj = 0; kj < 8; kj++) {
                    double grad = 0.0;
                    for (int i = 0; i < 32; i++) {
                        for (int j = 0; j < 32; j++) {
                            grad += conv_feature_maps[port][k][i][j] *
                                    Deviation[i * 8 + ki][j * 8 + kj];
                        }
                    }
                    input_conv_kernels[port][k].kernel[ki][kj] -=
                            learning_rate * grad / (32.0 * 32.0);
                }
            }
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

public:
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
            if (msg.to_coord[0] == local_coord[0] &&
                msg.to_coord[1] == local_coord[1] &&
                msg.to_coord[2] == local_coord[2]) {
                receiveMessages(msg);
            } else {
                route(msg);
            }
        } else if (msg.type == FIND_NEURON) {
            if (msg.remains > 1) {
                Message msg_forward = msg;
                msg_forward.remains--;
                sendMessage(msg_forward, randomULLInRange(0, 6));
            }

            Message msg_reply{};
            memcpy(msg_reply.last_proxy_coord, local_coord, 3 * sizeof(ll));
            memcpy(msg_reply.from_coord, local_coord, 3 * sizeof(ll));
            memcpy(msg_reply.to_coord, msg.from_coord, 3 * sizeof(ll));
            msg_reply.activity = activity;
            msg_reply.type = REPLY_NEURON_FIND;
            route(msg_reply);

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
            if (msg.to_coord[0] == local_coord[0] &&
                msg.to_coord[1] == local_coord[1] &&
                msg.to_coord[2] == local_coord[2]) {
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

public:
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
            sendMessage(msg, 0); // +X
        } else if (msg.to_coord[0] < local_coord[0]) {
            sendMessage(msg, 1); // -X
        } else if (msg.to_coord[1] > local_coord[1]) {
            sendMessage(msg, 2); // +Y
        } else if (msg.to_coord[1] < local_coord[1]) {
            sendMessage(msg, 3); // -Y
        } else if (msg.to_coord[2] > local_coord[2]) {
            sendMessage(msg, 4); // +Z
        } else if (msg.to_coord[2] < local_coord[2]) {
            sendMessage(msg, 5); // -Z
        }
    }

    __device__ void sendMessage(const Message &msg, const int direction) const {
        if (direction >= 0 && direction < 6 && neighbour_queues[direction]) {
            neighbour_queues[direction]->push(msg);
        }
    }

    // ===== Receive messages&send to port =====
    __device__ void receiveMessages(Message msg) {
        for (int i = 0; i < input_conn_count; i++) {
            if (input_conns[i].coord[0] == msg.from_coord[0] &&
                input_conns[i].coord[1] == msg.from_coord[1] &&
                input_conns[i].coord[2] == msg.from_coord[2]) {
                NeuronInput cache_inp{};
                cache_inp.activity = msg.activity;
                cache_inp.weight = msg.weight;
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
        ll pos = local_coord[0] * GRID_SIZE * GRID_SIZE +
                 local_coord[1] * GRID_SIZE +
                 local_coord[2];

        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                int d = i * 256 + j;

                double freq = 1.0 / pow(10000.0, 2.0 * d / 65536.0);

                if (d % 2 == 0) {
                    P_Matrix[i][j] += 0.1 * sin(pos * freq);
                } else {
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
        double weight_sum = 0.0;
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                PS_aggregate[i][j] = 0.0;
            }
        }
        selectiveSSM();
        for (int p = 0; p < 4; p++) {
            if (!port_in[p].empty()) {
                NeuronInput temp_inp = port_in[p].front();

                double transformed_input[256][256];

                matmul_double(&temp_inp.array[0][0], &input_multiplex_array[0][0][p],
                              &transformed_input[0][0]);
                extractConvFeatures(p, transformed_input);

                double score = 0.0;
                for (int i = 0; i < 256; i++) {
                    for (int j = 0; j < 256; j++) {
                        // Q = P_Matrix[i][j]
                        // K = all_inputs[k]
                        score += P_Matrix[i][j] * transformed_input[i][j];
                    }
                }
                score /= sqrt(256.0 * 256.0);

                double aggregated[256][256];
                aggregateFeatures(p, aggregated);

                double w = temp_inp.weight * temp_inp.activity;
                weight_sum += w;

                for (int i = 0; i < 256; i++) {
                    double wkv = 0.0;
                    double state = h_state[i];
                    for (int j = 0; j < 256; j++) {
                        double k = PS_aggregate[i][j]; // key
                        double v = PS_aggregate[i][j]; // value
                        double w = -exp(time_decay[i]);
                        //wkv compute
                        wkv += exp(__half2float(time_first[i]) + k) * v;
                        state = state * exp(w) + exp(k) * v;
                        PS_aggregate[i][j] += transformed_input[i][j] * w * aggregated[i][j] * score + wkv / (
                            wkv + state);
                    }
                }
            }
        }

        // normalize
        if (weight_sum > 1e-6) {
            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    PS_aggregate[i][j] /= weight_sum;
                }
            }
        }

        // deviation
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                Deviation[i][j] = __half2float(PS_aggregate[i][j]) - P_stable[i][j];
            }
        }
        STM_aggregate_utility = computeKFEAttention();

        bool trigger_gemm = false;
        if (cycle_counter % 16 == 0) {
            trigger_gemm = true;
        }
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
        if (core_vulnerability > 0.7) {
            trigger_gemm = true;
        }
        if (STM_aggregate_utility > 0.6) {
            trigger_gemm = true;
        }

        if (trigger_gemm) {
            executeGEMMAndDRC();
        } else {
            executeMicroCorrection();
        }


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

            double dot_product = 0.0;
            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    dot_product += kfe_local[k].Vmem[i][j] * Deviation[i][j];
                }
            }

            usum += kfe_local[k].Ulocal;
            isum += kfe_local[k].Icore;
            vsum += kfe_local[k].V;

            double attention_weight = 1.0 / (1.0 + exp(-dot_product));
            double weighted_attention = attention_weight * kfe_local[k].Icore;

            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    M_KFE[i][j] += weighted_attention * kfe_local[k].Vmem[i][j];
                }
            }

            utility += weighted_attention;

            kfe_mutex.lock();
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
            // query external KFE through queue
            if (kfe_query_queue && kfe_result_queue) {
                kfe_query_queue->push(ext_kfe_slots[max_index].hash.data());
                for (int wait_cycle = 0; wait_cycle < 100; wait_cycle++) {
                    if (kfe_result_queue->size() > 0) {
                        break;
                    }
                    __nanosleep(100);
                }
                if (!kfe_result_queue->pop(ext_kfe_pulled)) {
                    ext_kfe_pulled = {};
                }
            }
            double dot_product = 0.0;
            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    dot_product += ext_kfe_pulled.Vmem[i][j] * Deviation[i][j];
                }
            }
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

    static __host__ __device__ double gelu(double x) {
        return 0.5 * x * (1.0 + tanh(0.797885 * (x + 0.044715 * x * x * x)));
    }

    __device__ void selectiveSSM() {
        double B[256], C[256];

        for (int i = 0; i < 256; i++) {
            double input_i = 0.0;
            for (int j = 0; j < 256; j++) {
                input_i += __half2float(PS_aggregate[i][j]);
            }
            input_i /= 256.0;

            B[i] = gelu(input_i);
            C[i] = gelu(-input_i);
        }

        for (int i = 0; i < 256; i++) {
            // Δ = B × input + decay × h_old
            double delta = B[i] * __half2float(PS_aggregate[i][0]);
            h_state[i] = 0.9 * h_state[i] + delta;
        }

        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                P_Matrix[i][j] += C[i] * h_state[i];
            }
        }
    }

    __device__ bool shouldUseFullAttention() {
        return (core_vulnerability > 0.7) ||
               (STM_aggregate_utility > 0.6);
    }

    __host__ __device__ void predictNoise(double input[256][256], double output[256][256]) {
        double scale1[256][256]; // 细节特征
        double scale2[256][256]; // 中等尺度特征
        double scale3[256][256]; // 粗糙特征

        for (int i = 1; i < 255; i++) {
            for (int j = 1; j < 255; j++) {
                double gx = -input[i - 1][j - 1] - 2 * input[i][j - 1] - input[i + 1][j - 1] +
                            input[i - 1][j + 1] + 2 * input[i][j + 1] + input[i + 1][j + 1];
                double gy = -input[i - 1][j - 1] - 2 * input[i - 1][j] - input[i - 1][j + 1] +
                            input[i + 1][j - 1] + 2 * input[i + 1][j] + input[i + 1][j + 1];
                scale1[i][j] = sqrt(gx * gx + gy * gy) * 0.1; // 缩放因子
            }
        }

        for (int i = 2; i < 254; i++) {
            for (int j = 2; j < 254; j++) {
                double sum = 0.0;
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
                scale2[i][j] = sum / 256.0;
            }
        }

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

        for (int i = 4; i < 252; i++) {
            for (int j = 4; j < 252; j++) {
                double fused = 0.4 * scale1[i][j] + 0.3 * scale2[i][j] + 0.3 * scale3[i][j];

                // swish:x * sigmoid(x)
                double sigmoid_val = 1.0 / (1.0 + exp(-fused));
                output[i][j] = fused * sigmoid_val;
            }
        }

        double attention_map[256][256];
        double mean_val = 0.0;

        for (int i = 4; i < 252; i++) {
            for (int j = 4; j < 252; j++) {
                mean_val += fabs(output[i][j]);
            }
        }
        mean_val /= (248 * 248);

        for (int i = 4; i < 252; i++) {
            for (int j = 4; j < 252; j++) {
                double diff = fabs(output[i][j]) - mean_val;
                attention_map[i][j] = 1.0 + tanh(diff * 2.0);
            }
        }

        for (int i = 4; i < 252; i++) {
            for (int j = 4; j < 252; j++) {
                output[i][j] *= attention_map[i][j];
            }
        }
        for (int i = 4; i < 252; i++) {
            for (int j = 4; j < 252; j++) {
                output[i][j] += 0.1 * input[i][j]; // soft connect
            }
        }

        // Final smoothing processing
        double temp[256][256];
        for (int i = 5; i < 251; i++) {
            for (int j = 5; j < 251; j++) {
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
        /*
        dim3 block(16, 16);
        dim3 grid((256 + 15) / 16, (256 + 15) / 16);

        tiledMatMulShared<16><<<grid, block>>>(
            &P_Matrix[0][0], &W_predict[0][0], &temp_product[0][0],
            256, 256, 256
        );
        */

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
                {
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

    __device__ double computeImportance() const {
        double importance = 0.0;
        importance += core_vulnerability * 0.4;
        importance += activity * 0.3;
        double deviation_norm = 0.0;
        for (int i = 0; i < 256; i += 16) {
            for (int j = 0; j < 256; j += 16) {
                deviation_norm += Deviation[i][j] * Deviation[i][j];
            }
        }
        deviation_norm = sqrt(deviation_norm / 256.0);
        importance += fmin(deviation_norm, 1.0) * 0.2;
        const double conn_ratio = stpc<double>(output_conn_count) / 2048.0;
        importance += conn_ratio * 0.1;

        return fmin(importance, 1.0);
    }

    __device__ void broadcastOutput() {
        for (int out_idx = 0; out_idx < output_conn_count; out_idx++) {
            Message out_msg;
            memcpy(out_msg.from_coord, local_coord, sizeof(local_coord));
            memcpy(out_msg.to_coord, output_conns[out_idx].coord, sizeof(ll) * 3);
            memcpy(out_msg.last_proxy_coord, local_coord, sizeof(local_coord));

            int port = output_conns[out_idx].port;

            double output_temp[256][256];
            matmul_double(&P_Matrix[0][0], &output_multiplex_array[0][0][port],
                          &output_temp[0][0]);
            /*
            dim3 block(16, 16);
            dim3 grid((256 + 15) / 16, (256 + 15) / 16);
            tiledMatMulShared<16><<<grid, block>>>(
                &P_Matrix[0][0], &W_predict[0][0], &output_temp[0][0],
                256, 256, 256
            );
            */

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

__global__ void kfeAttentionShared(
    Neuron *neurons,
    int neuron_count
) {
    int neuron_id = blockIdx.x;
    if (neuron_id >= neuron_count) return;

    Neuron &n = neurons[neuron_id];

    // Shared memory
    __shared__ half s_deviation[256][16]; // 8KB - 缓存部分 Deviation
    __shared__ float s_attention[16]; // 64B - attention scores
    __shared__ float s_utility[16]; // 64B - utility values

    int tid = threadIdx.x; // 0-255
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Phase 1: 加载 Deviation 的前 16 列到 shared memory
    for (int col = 0; col < 16; col++) {
        s_deviation[tid][col] = __float2half(n.Deviation[tid][col]);
    }
    __syncthreads();

    // Phase 2: 每 2 个 warp 处理一个 KFE slot
    int kfe_idx = warp_id * 2 + (lane_id >= 16 ? 1 : 0);
    int local_lane = lane_id % 16;

    if (kfe_idx < 16) {
        if (n.kfe_local[kfe_idx].Icore < 0.01) {
            // 空 slot
            if (local_lane == 0) {
                s_attention[kfe_idx] = 0.0f;
                s_utility[kfe_idx] = 0.0f;
            }
        } else {
            // 计算 dot product (简化版,只用前16列)
            float dot = 0.0f;

            for (int i = local_lane; i < 256; i += 16) {
                for (int j = 0; j < 16; j++) {
                    float kfe_val = n.kfe_local[kfe_idx].Vmem[i][j];
                    float dev_val = __half2float(s_deviation[i][j]);
                    dot += kfe_val * dev_val;
                }
            }

            // Warp reduction
#pragma unroll
            for (int offset = 8; offset > 0; offset /= 2) {
                dot += __shfl_down_sync(0xffff, dot, offset);
            }

            if (local_lane == 0) {
                // Sigmoid
                float attention = 1.0f / (1.0f + expf(-dot));
                float weighted = attention * n.kfe_local[kfe_idx].Icore;

                s_attention[kfe_idx] = weighted;
                s_utility[kfe_idx] = attention;
            }
        }
    }
    __syncthreads();

    // Phase 3: 聚合到 M_KFE (所有线程)
    for (int j = 0; j < 256; j++) {
        float sum = 0.0f;

#pragma unroll 4
        for (int k = 0; k < 16; k++) {
            if (s_attention[k] > 0.001f) {
                sum += s_attention[k] * n.kfe_local[k].Vmem[tid][j];
            }
        }

        n.M_KFE[tid][j] = sum;
    }

    // Phase 4: 计算总 utility (单 warp)
    if (tid < 32) {
        float total_utility = 0.0f;
        for (int i = tid; i < 16; i += 32) {
            total_utility += s_utility[i];
        }

#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            total_utility += __shfl_down_sync(0xffffffff, total_utility, offset);
        }

        if (tid == 0) {
            n.STM_aggregate_utility = total_utility / 16.0;
        }
    }
}

__global__ void step_inf(
    Neuron *neurons,
    int neuron_count,
    double score
) {
    ull neuron_id = blockIdx.x;
    int port = (threadIdx.x % 2 + 1) * (threadIdx.y % 2 + 1) - 1; // 0-3
    ull obj_x = threadIdx.x / 2;
    ull obj_y = threadIdx.y / 2;
    if (neuron_id >= neuron_count) return;

    Neuron &n = neurons[neuron_id];

    // Shared memory
    __shared__ half s_port_data[4][256 * 256]; // 4 个端口的部分数据
    __shared__ half s_temp_data[4][256 * 256]; // 临时数据
    __shared__ float s_port_weights[4][256 * 256];
    __shared__ float s_ptw[4];
    __shared__ bool s_port_valid[4]; // 4 个端口是否有效
    __shared__ float sum;

    if (n.port_in[port].empty()) {
        s_port_valid[port] = false;
    } else {
        s_port_valid[port] = true;
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        sum = 0.0;
    }
    __syncthreads();
    __shared__ double As[16][16];
    __shared__ double Bs[16][16];
    if (s_port_valid[port]) {
        NeuronInput inp = n.port_in[port].front();
        s_port_data[port][obj_x * obj_y] = __float2half(inp.array[obj_x][obj_y]);
        s_port_weights[port][obj_x * obj_y] = __float2half(n.input_multiplex_array[port][obj_x][obj_y]);
        if (obj_x == 0 && obj_y == 0) {
            s_ptw[port] = inp.weight * inp.activity;
        }
        if (obj_x < 16 && obj_y < 16) {
            int tx = threadIdx.x % 16;
            int ty = threadIdx.y % 16;
            int row = obj_x * 16 + ty;
            int col = obj_y * 16 + tx;
            int t = sqrt(obj_x * obj_y);
            if (row < 256 && t * 16 + tx < 256)
                As[ty][tx] = s_port_data[port][row * 256 + t * 16 + tx];
            else
                As[ty][tx] = 0;
            if (col < 256 && t * 256 + ty < 256)
                Bs[ty][tx] = s_port_weights[port][(t * 16 + ty) * 256 + col];
            else
                Bs[ty][tx] = 0;
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                sum += As[ty][k] * Bs[k][tx]; // FMA
            }
            if (obj_x == 0 && obj_y == 0) {
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    atomicAdd(&sum, As[ty][k] * Bs[k][tx]);
                }
            }
            if (row < 256 && col < 256) {
                s_temp_data[port][row * 256 + col] = sum;
            }
        }
    } else {
        s_port_data[port][obj_x * obj_y] = __float2half(0.0f);
    }
    __syncthreads();
    __shared__ double temp_dta[4][256][256];
    temp_dta[port][obj_x][obj_y] = 0;
    __syncthreads();
    atomicAdd(&temp_dta[port][obj_x][obj_y], __half2float(s_port_data[port][obj_x * obj_y]) / 8.0);
    if (obj_x == 0 && obj_y == 0) {
        sum = 0;
    }
    __syncthreads();
    if (obj_x < 4 && obj_y == 0) {
        double w = s_ptw[obj_x];
        sum += w;
    }
    __syncthreads();
    if (obj_x < 256 && obj_y < 256 && port == 0) {
        double wkv = 0.0;
        double state = n.h_state[obj_x];
        double k = n.PS_aggregate[obj_x][obj_y]; // key
        double v = n.PS_aggregate[obj_x][obj_y]; // value
        double w = -exp(n.time_decay[obj_x]);
        //wkv compute
        wkv += exp(__half2float(n.time_first[obj_x]) + k) * v;
        state = state * exp(w) + exp(k) * v;
        atomicAdd(&sum, 0);
        n.PS_aggregate[obj_x][obj_y] += __float2half(
            __half2float(s_temp_data[port][obj_x * obj_y]) * w * temp_dta[port][obj_x][
                obj_y] * score + wkv / (
                wkv + state));
    }
    if (sum > 1e-6 && port == 0 && obj_x == 0 && obj_y == 0) {
        n.PS_aggregate[obj_x][obj_y] /= sum;
    }
    __syncthreads();
    if (port == 0) {
        n.Deviation[obj_x][obj_y] = __half2float(n.PS_aggregate[obj_x][obj_y]) - n.P_stable[obj_x][obj_y];
        memset(n.M_KFE, 0, sizeof(n.M_KFE));
    }
    __syncthreads();
    double dot_product = 0.0;
    __shared__ double *utt_overflow;
    __shared__ double usum;
    __shared__ double isum;
    __shared__ double vsum;
    if (port == 0 && obj_x == 0 && obj_y == 0) {
        usum = 0.0;
        isum = 0.0;
        vsum = 0.0;
        memset(&temp_dta, 0, sizeof(temp_dta));
        memset(&s_temp_data, 0, sizeof(s_temp_data));
        if (neuron_count > 327420) {
            utt_overflow = new double[neuron_count - 327420];
        } else utt_overflow = nullptr;
    }
    __syncthreads();
    for (int i = 1; i <= 4; i++) {
        ull k = port * i - 1;
        dot_product = n.kfe_local[k].Vmem[obj_x][obj_y] * n.Deviation[obj_x][obj_y];
        // Warp reduction
#pragma unroll
        for (int offset = 8; offset > 0; offset /= 2) {
            dot_product += __shfl_down_sync(0xffff, dot_product, offset);
        }
        atomicAdd(&usum, n.kfe_local[k].Ulocal);
        atomicAdd(&isum, n.kfe_local[k].Icore);
        atomicAdd(&vsum, n.kfe_local[k].V);
        __syncthreads();
        double attention_weight = 1.0 / (1.0 + exp(-dot_product));
        double weighted_attention = attention_weight * n.kfe_local[k].Icore;
        atomicAdd(&n.M_KFE[obj_x][obj_y], weighted_attention * n.kfe_local[k].Vmem[obj_x][obj_y]);
        if (obj_x == 0 && obj_y == 0) {
            if (neuron_id / 4 / 256 < 256) {
                atomicAdd(&temp_dta[neuron_id % 4][min(neuron_id / 4 / 256, 255ULL)][neuron_id % 256],
                          weighted_attention);
            } else if (neuron_id <= 327420) {
                atomicAdd(retpc<ull *>(&s_temp_data[neuron_id % 4][neuron_id]), 0);
                s_temp_data[neuron_id % 4][neuron_id / 4 - 65280] += weighted_attention;
            } else if (utt_overflow != nullptr) {
                atomicAdd(&utt_overflow[neuron_id - 327460], weighted_attention);
            }
            n.kfe_mutex.lock();
            n.kfe_local[k].Ulocal += 0.01 * (attention_weight - n.kfe_local[k].Ulocal);
            n.kfe_local[k].Icore += 0.01 * (0.5 - n.kfe_local[k].Icore);
            n.kfe_local[k].V -= 0.01 * (1.0 - n.kfe_local[k].V);
            n.kfe_local[k].Rcycles = n.cycle_counter;
            n.kfe_mutex.unlock();
        }
        __syncthreads();
    }
    __shared__ bool trigger_gemm[neuron_count];
    __shared__ double deviation_norm[neuron_count];
    if (port == 0 && obj_x == 0 && obj_y == 0) {
        memset(trigger_gemm, 0, sizeof(trigger_gemm));
    }
    __syncthreads();
    if (n.cycle_counter % 16 == 0) {
        trigger_gemm[neuron_id] = true;
    }
    atomicAdd(&deviation_norm[neuron_id], n.Deviation[obj_x][obj_y] * n.Deviation[obj_x][obj_y]);
    __syncthreads();
    deviation_norm[neuron_id] = sqrt(deviation_norm[neuron_id] / (256.0 * 256.0));
    __syncthreads();
    if (port == 0 && obj_x == 0 && obj_y == 0) {
        if (deviation_norm[neuron_id] > 0.5) {
            trigger_gemm[neuron_id] = true;
        }
        if (n.core_vulnerability > 0.7) {
            trigger_gemm[neuron_id] = true;
        }
        if (neuron_id / 4 / 256 < 256) {
            if (temp_dta[neuron_id % 4][min(neuron_id / 4 / 256, 255ULL)][neuron_id % 256] > 0.6) {
                trigger_gemm[neuron_id] = true;
            }
        } else if (neuron_id <= 327420) {
            if (s_temp_data[neuron_id % 4][neuron_id / 4 - 65280] > __float2half(0.6)) {
                trigger_gemm[neuron_id] = true;
            }
        } else if (utt_overflow != nullptr) {
            if (utt_overflow[neuron_id - 327460] > 0.6) {
                trigger_gemm[neuron_id] = true;
            }
        }
    }
    __syncthreads();
    if (port == 0 && neuron_id == 0 && obj_x == 0 && obj_y == 0 && utt_overflow != nullptr) {
        delete[] utt_overflow;
    }
    __syncthreads();
    if (port != 0)
        return;
    if (!trigger_gemm)
        goto not_trg;
trg:
    double (*W_backup)[256];
    double P_Original = n.P_Matrix[obj_x][obj_y];
    if (n.training && port == 0 && obj_x == 0 && obj_y == 0) {
        W_backup = new double[256][256];
        memcpy(W_backup, n.W_predict, sizeof(n.W_predict));
    }
    if (port == 0) {
        ll pos = n.local_coord[0] * n.GRID_SIZE * n.GRID_SIZE +
                 n.local_coord[1] * n.GRID_SIZE +
                 n.local_coord[2];
        int d = obj_x * 256 + obj_y;

        double freq = 1.0 / pow(10000.0, 2.0 * d / 65536.0);

        if (d % 2 == 0) {
            n.P_Matrix[obj_x][obj_y] += 0.1 * sin(pos * freq);
        } else {
            n.P_Matrix[obj_x][obj_y] += 0.1 * cos(pos * freq);
        }
        if (curand_uniform(&n.rand_state) < 0.05 && n.training) {
            n.W_predict[obj_x][obj_y] = 0.0;
        }
    }
    __syncthreads();
    __shared__ double *P_Next;
    // matmul_double(&P_Matrix[0][0], &W_predict[0][0], &temp_product[0][0]);
    if (port == 0 && obj_x == 0 && obj_y == 0) {
        memset(&As, 0, sizeof(As));
        memset(&Bs, 0, sizeof(Bs));
        memset(&temp_dta, 0, sizeof(temp_dta));
        memset(&s_temp_data, 0, sizeof(s_temp_data));
        if (neuron_count > 327420) {
            utt_overflow = new double[neuron_count - 327420];
        } else utt_overflow = nullptr;
        P_Next = new double[256 * 256];;
        sum = 0;
    }
    __syncthreads();
    int tx = obj_x % 16;
    int ty = obj_y % 16;
    int row = obj_x + ty;
    int col = obj_y + tx;

    // Tile iteration
#pragma unroll
    for (int t = 0; t < 16; ++t) {
        // coalesced access
        if (row < 256 && t * 256 + tx < 256)
            As[ty][tx] = (&n.P_Matrix[0][0])[row * 256 + t * 16 + tx];
        else
            As[ty][tx] = 0;

        if (col < 256 && t * 16 + ty < 256)
            Bs[ty][tx] = (&n.W_predict[0][0])[(t * 16 + ty) * 256 + col];
        else
            Bs[ty][tx] = 0;

        __syncthreads();

        // Compute on shared memory (fast!)
#pragma unroll
        for (int k = 0; k < 15; ++k) {
            sum = As[ty][k] * Bs[k][tx]; // FMA
        }
#pragma unroll
        for (int offset = 8; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffff, sum, offset);
        }
        __syncthreads();
    }

    if (row < 256 && col < 256) {
        if (neuron_id / 4 / 256 < 256) {
            temp_dta[neuron_id % 4][min(neuron_id / 4 / 256, 255ULL)][neuron_id % 256] = sum;
        } else if (neuron_id <= 327420) {
            s_temp_data[neuron_id % 4][neuron_id / 4 - 65280] = sum;
        } else if (utt_overflow != nullptr) {
            utt_overflow[neuron_id - 327460] = sum;
        }
    }
    __shared__ double P_Nsc[256][256];
    __syncthreads();
    double x = 0;
    if (neuron_id / 4 / 256 < 256) {
        x += temp_dta[neuron_id % 4][min(neuron_id / 4 / 256, 255ULL)][neuron_id % 256];
    } else if (neuron_id <= 327420) {
        x += __half2float(s_temp_data[neuron_id % 4][neuron_id / 4 - 65280]);
    } else if (utt_overflow != nullptr) {
        x += utt_overflow[neuron_id - 327460];
    }
    x += n.M_KFE[obj_x][obj_y];
    P_Nsc[obj_x][obj_y] = x;
    // GELU activation
    // P_Next[obj_x*obj_y] = 0.5 * x * (1.0 + tanh(0.797885 * (x + 0.044715 * x * x * x)));
    P_Next[obj_x * obj_y] = Neuron::gelu(x);
    __shared__ double T_fixed[256][256];
    double alpha = 0.7;
    if (port == 0) {
        T_fixed[obj_x][obj_y] = alpha * __half2float(n.PS_aggregate[obj_x][obj_y]) +
                                (1.0 - alpha) * P_Next[obj_x * obj_y];
    }
    double epsilon = 1e-4;
    double eta_base = 0.1;
    double lambda = 0.9;
    __shared__ ull history_index;
    __shared__ double diff_norm;

    __shared__ double prev_diff_norm;
    if (obj_x == 0 && obj_y == 0) {
        prev_diff_norm = 0;
        history_index = 0;
        diff_norm = 0;
    }
    __syncthreads();
    for (int ik = 1; ik <= 16; ik++) {
        double pcij = 0;
        if (neuron_id / 4 / 256 < 256) {
            pcij += temp_dta[neuron_id % 4][min(neuron_id / 4 / 256, 255ULL)][neuron_id % 256];
        } else if (neuron_id <= 327420) {
            pcij += __half2float(s_temp_data[neuron_id % 4][neuron_id / 4 - 65280]);
        } else if (utt_overflow != nullptr) {
            pcij += utt_overflow[neuron_id - 327460];
        }
        __syncthreads();
        int iter = (port + 1) * ik - 1;
        double V_corr = (T_fixed[obj_x][obj_y] - P_Next[obj_x * obj_y]) * eta_base;
        double local_feature = 0.0;
        for (int p = 0; p < 4; p++) {
            if (!n.port_in[p].empty()) {
                NeuronInput temp = n.port_in[p].front();
                local_feature += temp.array[obj_x][obj_y];
            }
        }
        local_feature /= 4.0;
        double attn_weight = 1.0 / (1.0 + exp(-(local_feature * P_Next[obj_x * obj_y])));
        double M_attn = attn_weight * V_corr;
        double V_hist = 0.0;
        if (iter > 0) {
            for (int h = 1; h <= min(iter, 3); h++) {
                ull hist_idx = (history_index - h + 5) % 5;
                ull prev_idx = (hist_idx - 1 + 5) % 5;
                double delta = n.P_history[hist_idx][obj_x][obj_y] -
                               n.P_history[prev_idx][obj_x][obj_y];
                V_hist += pow(lambda, h) * delta;
            }
        }
        // DESTRUCTIVE!
        n.P_Matrix[obj_x][obj_y] = P_Next[obj_x * obj_y] + V_corr + M_attn + V_hist;
        __syncthreads();
        double diff = n.P_Matrix[obj_x][obj_y] - pcij;
        diff_norm += diff * diff;
        __syncthreads();
        diff_norm = sqrt(diff_norm);
        history_index = (history_index + 1) % 5;
        n.P_history[history_index][obj_x][obj_y] = __float2half(pcij);
        double pcij1 = 0;
        if (neuron_id / 4 / 256 < 256) {
            pcij1 += temp_dta[neuron_id % 4][min(neuron_id / 4 / 256, 255ULL)][neuron_id % 256];
        } else if (neuron_id <= 327420) {
            pcij1 += __half2float(s_temp_data[neuron_id % 4][neuron_id / 4 - 65280]);
        } else if (utt_overflow != nullptr) {
            pcij1 += utt_overflow[neuron_id - 327460];
        }
        if (diff_norm < epsilon)
            break;
        if (iter > 8 && diff_norm > prev_diff_norm)
            break;
        prev_diff_norm = diff_norm;
    }
    __syncthreads();
    __shared__ double beta_schedule[16];
    __syncthreads();
    if (port == 0 && obj_x == 0 && obj_y == 0) {
        for (int t = 0; t < 16; t++) {
            double alpha_t = cos(PI * t / 32.0);
            beta_schedule[t] = 1.0 - alpha_t * alpha_t;
        }
        for (int t = 15; t >= 0; t--) {
            double beta = beta_schedule[t];
            double noise_pred[256][256];
            n.predictNoise(P_Nsc, noise_pred);
            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    P_Nsc[i][j] -= sqrt(beta) * noise_pred[i][j];
                    P_Nsc[i][j] = (P_Nsc[i][j] -
                                   sqrt(beta) * noise_pred[i][j]) /
                                  sqrt(1.0 - beta);
                }
            }
        }
    }
    __syncthreads();
    constexpr double alpha_c = 0.7;
    n.P_Matrix[obj_x][obj_y] += alpha_c * n.P_Matrix[obj_x][obj_y] +
            (1 - alpha_c) * P_Nsc[obj_x][obj_y];
    n.P_Matrix[obj_x][obj_y] /= 2;

    __syncthreads();
    if (port == 0 && obj_x == 0 && obj_y == 0)
        memcpy(n.P_stable, n.P_Matrix, sizeof(n.P_stable));
    __syncthreads();
    if (n.training && n.W_predict[obj_x][obj_y] < 0.01) {
        n.W_predict[obj_x][obj_y] = W_backup[obj_x][obj_y];
    }
    __syncthreads();
    __shared__ double deviat_num;
    if (port == 0 && obj_x == 0 && obj_y == 0)
        deviat_num = 0;
    __syncthreads();
    double diff = n.P_Matrix[obj_x][obj_y] - n.P_stable[obj_x][obj_y];
    atomicAdd(&deviat_num,diff * diff);
    __syncthreads();
    if (port == 0 && obj_x == 0 & obj_y == 0) {
        n.core_vulnerability = sqrt(deviat_num / (256.0 * 256.0));
        n.core_vulnerability = tanh(n.core_vulnerability);
    }
    __syncthreads();
    if (W_backup != nullptr && port == 0 && obj_x == 0 && obj_y == 0)
        delete[] W_backup;
    n.P_Matrix[obj_x][obj_y] = 0.9 * n.P_Matrix[obj_x][obj_y] + 0.1 * P_Original;
    __syncthreads();
    __shared__ double mean, var, std;
    if (port == 0 && obj_x == 0 && obj_y == 0) {
        mean = 0;
        var = 0;
    }
    __syncthreads();
    atomicAdd(&mean, n.P_Matrix[obj_x][obj_y]);
    __syncthreads();
    if (port == 0 && obj_x == 0 && obj_y == 0)
        mean /= 256*256;
    atomicAdd(&var, (n.P_Matrix[obj_x][obj_y] - mean) * (n.P_Matrix[obj_x][obj_y] - mean));
    __syncthreads();
    if (port == 0 && obj_x == 0 && obj_y == 0) {
        var /= 256*256;
        std = sqrt(var + 1e-6);
    }
    __syncthreads();
    n.P_Matrix[obj_x][obj_y] = (n.P_Matrix[obj_x][obj_y] - mean) / std;
    __syncthreads();
    goto last;
not_trg:
    ull i = obj_x;;
    ull j = obj_y;
    alpha = 0.3;
    double eta_micro = 0.05 + max(min(n.getLearningRate(3), 1.0), 0.0) * 0.0001;
    double T_micro = alpha * __half2float(n.PS_aggregate[i][j]) +
                                 (1.0 - alpha) * n.P_Matrix[i][j];
    n.P_Matrix[i][j] += eta_micro * (T_micro - n.P_Matrix[i][j]);
    n.P_Matrix[i][j] += max(min(n.noise, 1.0), 0.0) * 0.0001 * (Neuron::randomInRange(0, 1) - 0.5);
    __syncthreads();
last:
    n.broadcastOutput();
    if (port == 0 && obj_x < 4 && obj_y == 0) {
        n.updateConvKernels(obj_x);
    }
}

__device__ double warpReduceSum(double val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ ull warpReduceSum(ull val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<typename T>
__device__ T* allocate(T* pool, ull &count, ull pool_size, ull request_size) {
    ull idx = atomicAdd(&count, request_size);
    if (idx >= pool_size) {
        atomicAdd(&count, -request_size);
        return nullptr;
    }
    return retpc<T*>(&pool[idx]);
}

template<typename T>
__device__ void free(T* ptr, ull &count, ull size) {
    atomicAdd(&count, -size);
    for (ull i = 0; i < size; i++) {
        ptr[i].~T();
    }
    memset(ptr, 0, sizeof(T) * size);
    for (ull i = 0; i < size; i++) {
        ptr[i] = T();
    }
}

__global__ void step_optimized(
    Neuron* neurons,
    ull neuron_count,
    double score,
    double *double_memory_pool,
    ull *ull_memory_pool,
    half *half_memory_pool,
    ull memory_pool_size
    ) {
    ull tid = blockIdx.x * blockDim.x + threadIdx.x;
    ull lane = threadIdx.x % 32;
    ull warp_id = threadIdx.x / 32;
    if (blockIdx.x * blockDim.x >= neuron_count * 4) return;
    ull neuron_id = (blockIdx.x * blockDim.x) / 4;
    ull port = (blockIdx.x * blockDim.x) % 4;
    Neuron &n = neurons[neuron_id];
    __shared__ double As[16][16];
    __shared__ double Bs[16][16];
    __shared__ ull memory_pool_count_ull;
    __shared__ ull memory_pool_count_double;
    __shared__ ull memory_pool_count_half;
    if (threadIdx.x == 0) {
        memory_pool_count_ull = 0;
        memory_pool_count_double = 0;
        memory_pool_count_half = 0;
    }
    __syncthreads();
    double input_matrix[256][256];
    // Load input matrix from port
    if (!n.port_in[port].empty()) {
        NeuronInput inp = n.port_in[port].front();
        input_matrix[tid / 256][tid % 256] = inp.array[tid / 256][tid % 256];
    } else {
        input_matrix[tid / 256][tid % 256] = 0.0;
    }
    __syncthreads();
    __shared__ double *transformed_input;
    if (threadIdx.x == 0) {
        transformed_input = allocate<double>(double_memory_pool, memory_pool_count_double, memory_pool_size, 256 * 256);
        if (transformed_input == nullptr) {
            // Out of memory, skip processing
            return;
        }
    }
    double sum_glb = 0;
    // 1024 threads per block
    // 1024 / 32 = 32 warps
    // 16 * 16 = 256 per tile
    // 256 * 256 = 65536 elements
    // 65536 / 256 = 256 tiles
    // each warp process 8 tiles
    // each tile is 16x16
    // process 4 tiles per block/cycle
    // 256 / 4 = 64 cycles
    // 256 / 32 = 8 warps per tile
    // 32 / 16 = 2 rows per warp
    // each warp process 2 rows of 16 cols
    for (int i = 0 ; i < 64 ; i++) {
        // element position in tile
        ull x_in_tile = threadIdx.x % 16;
        ull y_in_tile = (threadIdx.x / 16) % 16;
        // global element position
        ull element_x = (i % 16) * 16 + x_in_tile;
        ull element_y = (i / 16) * 16 + y_in_tile;
        // load elements into shared memory
        double element_a = 0.0;
        double element_b = 0.0;
        if (element_x < 256 && element_y < 256) {
            element_a = input_matrix[element_x][element_y];
            element_b = n.P_Matrix[element_x][element_y];
        }
        As[y_in_tile][x_in_tile] = element_a;
        Bs[y_in_tile][x_in_tile] = element_b;
        // synchronize to make sure the matrices are loaded
        __syncthreads();
#pragma unroll
        for (int k = 0 ; k < 16 ; k++) {
            sum_glb += As[y_in_tile][k] * Bs[k][x_in_tile];  // 改为累加
        }
        __syncthreads();
        if (element_x < 256 && element_y < 256) {  // 移除lane条件
            transformed_input[element_x * 256 + element_y] = sum_glb;
        }
        __syncthreads();
        sum_glb = 0;
    }
    /*
__device__ void processUpdate(int port) {
        if (port_in[port].empty()) return;

        NeuronInput curr_inp;
        port_in[port].pop(curr_inp);
        double weight_sum = 0.0;
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                PS_aggregate[i][j] = 0.0;
            }
        }
        selectiveSSM();
        for (int p = 0; p < 4; p++) {
            if (!port_in[p].empty()) {
                NeuronInput temp_inp = port_in[p].front();

                double transformed_input[256][256];

                matmul_double(&temp_inp.array[0][0], &input_multiplex_array[0][0][p],
                              &transformed_input[0][0]);
                extractConvFeatures(p, transformed_input);

                double score = 0.0;
                for (int i = 0; i < 256; i++) {
                    for (int j = 0; j < 256; j++) {
                        // Q = P_Matrix[i][j]
                        // K = all_inputs[k]
                        score += P_Matrix[i][j] * transformed_input[i][j];
                    }
                }
                score /= sqrt(256.0 * 256.0);

                double aggregated[256][256];
                aggregateFeatures(p, aggregated);

                double w = temp_inp.weight * temp_inp.activity;
                weight_sum += w;

                for (int i = 0; i < 256; i++) {
                    double wkv = 0.0;
                    double state = h_state[i];
                    for (int j = 0; j < 256; j++) {
                        double k = PS_aggregate[i][j]; // key
                        double v = PS_aggregate[i][j]; // value
                        double w = -exp(time_decay[i]);
                        //wkv compute
                        wkv += exp(__half2float(time_first[i]) + k) * v;
                        state = state * exp(w) + exp(k) * v;
                        PS_aggregate[i][j] += transformed_input[i][j] * w * aggregated[i][j] * score + wkv / (
                            wkv + state);
                    }
                }
            }
        }

        // normalize
        if (weight_sum > 1e-6) {
            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    PS_aggregate[i][j] /= weight_sum;
                }
            }
        }

        // deviation
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                Deviation[i][j] = __half2float(PS_aggregate[i][j]) - P_stable[i][j];
            }
        }
        STM_aggregate_utility = computeKFEAttention();

        bool trigger_gemm = false;
        if (cycle_counter % 16 == 0) {
            trigger_gemm = true;
        }
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
        if (core_vulnerability > 0.7) {
            trigger_gemm = true;
        }
        if (STM_aggregate_utility > 0.6) {
            trigger_gemm = true;
        }

        if (trigger_gemm) {
            executeGEMMAndDRC();
        } else {
            executeMicroCorrection();
        }


        broadcastOutput();

        for (int p = 0; p < 4; p++) {
            updateConvKernels(p);
        }
    }
     */

}

// ===== 补充3: DRC 迭代优化 (双缓冲) =====

/**
 * @brief DRC 迭代优化 - 使用 ping-pong buffer
 * @note 在 shared memory 中进行迭代
 *
 * Launch: <<<neuron_count, 256, smem_size>>>
 * smem_size = 3 * 256 * 256 * sizeof(half)  // 384KB
 */
__global__ void drcIterationShared(
    Neuron *neurons,
    int neuron_count,
    int num_iterations = 16
) {
    extern __shared__ char smem[];

    int neuron_id = blockIdx.x;
    if (neuron_id >= neuron_count) return;

    Neuron &n = neurons[neuron_id];

    // 双缓冲
    half *P_buffer[2];
    P_buffer[0] = (half *) smem;
    P_buffer[1] = (half *) (smem + 256 * 256 * sizeof(half));
    half *T_fixed = (half *) (smem + 2 * 256 * 256 * sizeof(half));

    int tid = threadIdx.x;

    // Phase 1: 加载初始数据
    for (int idx = tid; idx < 256 * 256; idx += blockDim.x) {
        int i = idx / 256;
        int j = idx % 256;

        P_buffer[0][idx] = __float2half(n.P_Matrix[i][j]);

        // 计算 T_fixed (简化版)
        double alpha = 0.7;
        double t_val = alpha * __half2float(n.PS_aggregate[i][j]) +
                       (1.0 - alpha) * n.P_Matrix[i][j];
        T_fixed[idx] = __float2half(t_val);
    }
    __syncthreads();

    // Phase 2: 迭代优化 (ping-pong)
    int current = 0;
    float eta_base = 0.1f;

    for (int iter = 0; iter < num_iterations; iter++) {
        int next = 1 - current;

        for (int idx = tid; idx < 256 * 256; idx += blockDim.x) {
            float p_curr = __half2float(P_buffer[current][idx]);
            float t_fix = __half2float(T_fixed[idx]);

            // DRC 校正
            float v_corr = (t_fix - p_curr) * eta_base;

            // TODO: 添加其他校正项 (历史动量等)

            P_buffer[next][idx] = __float2half(p_curr + v_corr);
        }
        __syncthreads();

        current = next;
    }

    // Phase 3: 写回
    for (int idx = tid; idx < 256 * 256; idx += blockDim.x) {
        int i = idx / 256;
        int j = idx % 256;
        n.P_Matrix[i][j] = __half2float(P_buffer[current][idx]);
    }
}

__global__ void all_neurons_kernel(Neuron *neurons, bool *active_flags, ull count) {
    ull tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count && active_flags[tid]) {
        neurons[tid].step();
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
        double advantage = global_score - d_ema_baseline;
        double gradient = trace[tid] * advantage * activity;
        double learning_rate = 0.001 / (1.0 + neurons[tid].getcs() * 0.0001);
        neurons[tid].adjust_weights_rl(gradient * learning_rate);
        neurons[tid].set_noise(1 - trace[tid]);
        trace[tid] *= 0.95;
    }
}

__global__ void update_ema_baseline(double global_score) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_ema_baseline = ema_beta * d_ema_baseline + (1 - ema_beta) * global_score;
    }
}

static __global__ void injectNeuronKernel(Neuron *neurons, NeuronInput input, int neuron_index, int port) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        neurons[neuron_index].inject(input, port);
    }
}


static __global__ void saveNeuronKernel(Neuron *neurons, NeuronData *data, int neuron_index) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        *data = neurons[neuron_index].save();
    }
}


static __global__ void loadNeuronKernel(Neuron *neurons, NeuronData data, int neuron_index) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        neurons[neuron_index].load_device(data);
    }
}


static __global__ void updateNeuronKernel(Neuron *neurons, int neuron_index) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        neurons[neuron_index].step();
    }
}


static __global__ void processMessageKernel(Neuron *neurons, Message msg, int neuron_index) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        neurons[neuron_index].processMessage(msg);
    }
}


static __global__ void getNeuronStatsKernel(Neuron *neurons, NeuronStats *stats, int neuron_index) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        *stats = neurons[neuron_index].get_stats();
    }
}


static __global__ void setNeuronNoiseKernel(Neuron *neurons, double noise, int neuron_index) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        neurons[neuron_index].set_noise(noise);
    }
}


static __global__ void setNeuronLearnRateKernel(Neuron *neurons, double learn_rate, int neuron_index) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        neurons[neuron_index].set_learn_rt(learn_rate);
    }
}
#endif //SRC_NEURON_H
