// Neuron单步执行版本
// 基于原始Neuron.cu重构的单步执行接口

#ifndef SRC_NEURON_SINGLE_STEP_H
#define SRC_NEURON_SINGLE_STEP_H

#include "Neuron.cu"

class NeuronSingleStep {
private:
    Neuron* neuron;
    bool initialized;
    int current_step;
    
public:
    NeuronSingleStep() : neuron(nullptr), initialized(false), current_step(0) {}
    
    // 初始化神经元
    __device__ bool initialize(DeviceQueue<Message, 32> *queues[6], 
                              ll coord[3], 
                              ull seed, 
                              DeviceQueue<Message, 32> *queue_ptr,
                              std::function<KFE_STM_Slot(std::string)>* find_kfe,
                              std::function<bool(KFE_STM_Slot)>* storage_kfe,
                              std::function<bool(KFE_STM_Slot)>* exp_kfe) {
        neuron = new Neuron(queues, coord, seed, queue_ptr, find_kfe, storage_kfe, exp_kfe);
        initialized = (neuron != nullptr);
        return initialized;
    }
    
    // 单步执行接口
    __device__ bool step() {
        if (!initialized || !neuron) return false;
        
        // 执行单步操作
        bool result = executeSingleStep();
        current_step++;
        return result;
    }
    
    // 获取当前状态
    __device__ NeuronState getState() const {
        if (!initialized || !neuron) return NeuronState{};
        
        NeuronState state;
        state.activity = neuron->get_activity();
        state.cycle_counter = neuron->cycle_counter;
        state.core_vulnerability = neuron->core_vulnerability;
        state.STM_aggregate_utility = neuron->STM_aggregate_utility;
        state.input_conn_count = neuron->input_conn_count;
        state.output_conn_count = neuron->output_conn_count;
        
        return state;
    }
    
    // 注入输入数据
    __device__ bool injectInput(NeuronInput inp, int port) {
        if (!initialized || !neuron) return false;
        return neuron->inject(inp, port);
    }
    
    // 获取输出数据
    __device__ NeuronInput getOutput(int port) {
        if (!initialized || !neuron) return NeuronInput{};
        return neuron->detach(port);
    }
    
    // 检查是否有输出
    __device__ bool hasOutput(int port) const {
        if (!initialized || !neuron) return false;
        return !neuron->port_out[port].empty();
    }
    
    // 获取当前步数
    __device__ int getCurrentStep() const {
        return current_step;
    }
    
    // 重置神经元状态
    __device__ void reset() {
        current_step = 0;
        // 注意：这里需要谨慎处理，避免重置关键状态
    }
    
private:
    // 单步执行核心逻辑
    __device__ bool executeSingleStep() {
        Message msg_cache{};
        
        // 1. 处理消息队列
        bool message_processed = processMessages();
        
        // 2. 处理输入端口数据
        bool input_processed = processInputs();
        
        // 3. 更新活跃度
        neuron->updateActivity();
        
        // 4. 周期性维护任务
        performMaintenance();
        
        return message_processed || input_processed;
    }
    
    // 处理消息队列
    __device__ bool processMessages() {
        if (!neuron->queue || neuron->queue->empty()) 
            return false;
            
        Message msg_cache{};
        if (neuron->queue->pop(msg_cache)) {
            neuron->processMessage(msg_cache);
            return true;
        }
        return false;
    }
    
    // 处理输入端口
    __device__ bool processInputs() {
        bool processed = false;
        
        for (int p = 0; p < 4; p++) {
            if (!neuron->port_in[p].empty()) {
                neuron->processUpdate(p);
                processed = true;
            }
        }
        
        return processed;
    }
    
    // 执行维护任务
    __device__ void performMaintenance() {
        neuron->cycle_counter++;
        
        // KFE衰减(每10步)
        if (neuron->cycle_counter % 10 == 0) {
            neuron->kfeDecay();
        }
        
        // 神经元发现(每100步)
        static int neuron_discover_countdown = 100;
        neuron_discover_countdown--;
        if (neuron_discover_countdown <= 0) {
            if (neuron->activity > 0.3 && neuron->output_conn_count < 1024) {
                neuron->initiateFindNeuron();
            }
            neuron_discover_countdown = 100;
        }
        
        // 端口变换矩阵更新(每50步)
        if (neuron->cycle_counter % 50 == 0) {
            neuron->updateMultiplexMatrices();
        }
    }
};

// 单步执行核函数
__global__ void neurons_single_step_kernel(NeuronSingleStep* neurons, int count, int max_steps) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        for (int step = 0; step < max_steps; step++) {
            neurons[tid].step();
        }
    }
}

// 状态结构体
struct NeuronState {
    double activity;
    int cycle_counter;
    double core_vulnerability;
    double STM_aggregate_utility;
    int input_conn_count;
    int output_conn_count;
    
    NeuronState() : activity(0.0), cycle_counter(0), core_vulnerability(0.0), 
                   STM_aggregate_utility(0.0), input_conn_count(0), output_conn_count(0) {}
};

#endif // SRC_NEURON_SINGLE_STEP_H