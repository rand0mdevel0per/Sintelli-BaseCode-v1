// Neuron单步执行接口
// 为现有Neuron类添加单步执行功能

#ifndef SRC_NEURON_SINGLE_STEP_INTERFACE_H
#define SRC_NEURON_SINGLE_STEP_INTERFACE_H

#include "Neuron.cu"

enum NeuronStepMode {
    STEP_MESSAGE_PROCESSING,    // 消息处理阶段
    STEP_INPUT_PROCESSING,      // 输入处理阶段
    STEP_INFERENCE,             // 推理计算阶段
    STEP_OUTPUT_BROADCAST,      // 输出广播阶段
    STEP_MAINTENANCE            // 维护任务阶段
};

struct NeuronStepResult {
    bool success;               // 步骤是否成功执行
    NeuronStepMode next_mode;   // 建议的下一个步骤模式
    bool has_output;            // 是否有输出产生
    double activity_change;     // 活跃度变化
};

class NeuronWithSingleStep : public Neuron {
private:
    NeuronStepMode current_mode;
    int step_counter;
    bool has_pending_input;
    bool has_pending_messages;
    
public:
    NeuronWithSingleStep(DeviceQueue<Message, 32> *queues[6], 
                        ll coord[3], 
                        ull seed, 
                        DeviceQueue<Message, 32> *queue_ptr,
                        std::function<KFE_STM_Slot(std::string)>* find_kfe,
                        std::function<bool(KFE_STM_Slot)>* storage_kfe,
                        std::function<bool(KFE_STM_Slot)>* exp_kfe)
        : Neuron(queues, coord, seed, queue_ptr, find_kfe, storage_kfe, exp_kfe),
          current_mode(STEP_MESSAGE_PROCESSING),
          step_counter(0),
          has_pending_input(false),
          has_pending_messages(false) {
    }
    
    // 单步执行主函数
    __device__ NeuronStepResult step() {
        NeuronStepResult result{false, current_mode, false, 0.0};
        
        double prev_activity = activity;
        
        switch (current_mode) {
            case STEP_MESSAGE_PROCESSING:
                result = stepMessageProcessing();
                break;
            case STEP_INPUT_PROCESSING:
                result = stepInputProcessing();
                break;
            case STEP_INFERENCE:
                result = stepInference();
                break;
            case STEP_OUTPUT_BROADCAST:
                result = stepOutputBroadcast();
                break;
            case STEP_MAINTENANCE:
                result = stepMaintenance();
                break;
        }
        
        result.activity_change = activity - prev_activity;
        step_counter++;
        current_mode = result.next_mode;
        
        return result;
    }
    
    // 检查是否有待处理的任务
    __device__ bool hasPendingWork() const {
        return has_pending_messages || has_pending_input || 
               (queue && !queue->empty()) || 
               checkPortsForInput();
    }
    
    // 获取当前步骤模式
    __device__ NeuronStepMode getCurrentMode() const {
        return current_mode;
    }
    
    // 获取总步数
    __device__ int getStepCount() const {
        return step_counter;
    }
    
    // 重置单步执行状态
    __device__ void resetStepState() {
        current_mode = STEP_MESSAGE_PROCESSING;
        step_counter = 0;
        has_pending_input = false;
        has_pending_messages = false;
    }
    
private:
    // 消息处理步骤
    __device__ NeuronStepResult stepMessageProcessing() {
        NeuronStepResult result{false, STEP_MESSAGE_PROCESSING, false, 0.0};
        
        if (queue && !queue->empty()) {
            Message msg_cache{};
            if (queue->pop(msg_cache)) {
                processMessage(msg_cache);
                result.success = true;
                result.has_output = true;
            }
        }
        
        // 更新状态并决定下一步
        updatePendingStates();
        
        if (has_pending_input) {
            result.next_mode = STEP_INPUT_PROCESSING;
        } else if (shouldPerformInference()) {
            result.next_mode = STEP_INFERENCE;
        } else {
            result.next_mode = STEP_MAINTENANCE;
        }
        
        return result;
    }
    
    // 输入处理步骤
    __device__ NeuronStepResult stepInputProcessing() {
        NeuronStepResult result{false, STEP_INPUT_PROCESSING, false, 0.0};
        
        bool processed = false;
        for (int p = 0; p < 4; p++) {
            if (!port_in[p].empty()) {
                processUpdate(p);
                processed = true;
                result.success = true;
            }
        }
        
        // 更新活跃度
        updateActivity();
        
        // 决定下一步
        if (shouldPerformInference()) {
            result.next_mode = STEP_INFERENCE;
        } else if (has_pending_messages) {
            result.next_mode = STEP_MESSAGE_PROCESSING;
        } else {
            result.next_mode = STEP_MAINTENANCE;
        }
        
        return result;
    }
    
    // 推理计算步骤
    __device__ NeuronStepResult stepInference() {
        NeuronStepResult result{true, STEP_OUTPUT_BROADCAST, true, 0.0};
        
        // 执行推理计算
        if (shouldTriggerGEMM()) {
            executeGEMMAndDRC();
        } else {
            executeMicroCorrection();
        }
        
        return result;
    }
    
    // 输出广播步骤
    __device__ NeuronStepResult stepOutputBroadcast() {
        NeuronStepResult result{true, STEP_MAINTENANCE, true, 0.0};
        
        broadcastOutput();
        
        // 更新卷积核
        for (int p = 0; p < 4; p++) {
            updateConvKernels(p);
        }
        
        return result;
    }
    
    // 维护任务步骤
    __device__ NeuronStepResult stepMaintenance() {
        NeuronStepResult result{true, STEP_MESSAGE_PROCESSING, false, 0.0};
        
        cycle_counter++;
        
        // 周期性维护任务
        if (cycle_counter % 10 == 0) {
            kfeDecay();
        }
        
        // 神经元发现
        static int neuron_discover_countdown = 100;
        neuron_discover_countdown--;
        if (neuron_discover_countdown <= 0) {
            if (activity > 0.3 && output_conn_count < 1024) {
                initiateFindNeuron();
            }
            neuron_discover_countdown = 100;
        }
        
        // 端口变换矩阵更新
        if (cycle_counter % 50 == 0) {
            updateMultiplexMatrices();
        }
        
        // 更新状态
        updatePendingStates();
        
        if (has_pending_messages) {
            result.next_mode = STEP_MESSAGE_PROCESSING;
        } else if (has_pending_input) {
            result.next_mode = STEP_INPUT_PROCESSING;
        }
        
        return result;
    }
    
    // 检查是否需要执行推理
    __device__ bool shouldPerformInference() const {
        // 基于当前状态判断是否需要推理
        return activity > 0.1 || core_vulnerability > 0.2 || 
               STM_aggregate_utility > 0.3;
    }
    
    // 检查是否触发GEMM
    __device__ bool shouldTriggerGEMM() const {
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
    
    // 检查端口是否有输入
    __device__ bool checkPortsForInput() const {
        for (int p = 0; p < 4; p++) {
            if (!port_in[p].empty()) {
                return true;
            }
        }
        return false;
    }
    
    // 更新待处理状态
    __device__ void updatePendingStates() {
        has_pending_messages = (queue && !queue->empty());
        has_pending_input = checkPortsForInput();
    }
};

// 单步执行核函数
__global__ void neurons_step_kernel(NeuronWithSingleStep* neurons, int count, int steps_per_neuron) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        for (int i = 0; i < steps_per_neuron; i++) {
            neurons[tid].step();
        }
    }
}

// 状态监控核函数
__global__ void monitor_neurons_kernel(NeuronWithSingleStep* neurons, int count, NeuronState* states) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        states[tid].activity = neurons[tid].get_activity();
        states[tid].cycle_counter = neurons[tid].cycle_counter;
        states[tid].core_vulnerability = neurons[tid].core_vulnerability;
        states[tid].STM_aggregate_utility = neurons[tid].STM_aggregate_utility;
        states[tid].input_conn_count = neurons[tid].input_conn_count;
        states[tid].output_conn_count = neurons[tid].output_conn_count;
        states[tid].current_mode = neurons[tid].getCurrentMode();
        states[tid].step_count = neurons[tid].getStepCount();
    }
}

// 扩展的状态结构体
struct NeuronState {
    double activity;
    int cycle_counter;
    double core_vulnerability;
    double STM_aggregate_utility;
    int input_conn_count;
    int output_conn_count;
    NeuronStepMode current_mode;
    int step_count;
    
    NeuronState() : activity(0.0), cycle_counter(0), core_vulnerability(0.0), 
                   STM_aggregate_utility(0.0), input_conn_count(0), 
                   output_conn_count(0), current_mode(STEP_MESSAGE_PROCESSING),
                   step_count(0) {}
};

#endif // SRC_NEURON_SINGLE_STEP_INTERFACE_H