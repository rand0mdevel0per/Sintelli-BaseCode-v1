/**
 * @file KFEManager.h
 * @brief KFE存储管理器 - 处理神经元的外部KFE存储和查询
 * 
 * @details
 * 负责：
 * - 在主机端处理KFE存储请求
 * - 管理KFE查询队列
 * - 提供KFE持久化存储
 * - 异步处理设备端的KFE操作
 * 
 * @version 1.0
 * @date 2025-10-18
 */

#ifndef SRC_KFEMANAGER_H
#define SRC_KFEMANAGER_H

#include "structs.h"
#include "deviceQueue.cpp"
#include <unordered_map>
#include <thread>
#include <atomic>
#include <mutex>

/**
 * @class KFEManager
 * @brief KFE存储管理器类
 * 
 * @details
 * 管理神经元的知识特征编码存储，提供异步的存储和查询服务
 */
class KFEManager {
private:
    std::unordered_map<std::string, KFE_STM_Slot> kfe_storage_;  // KFE存储
    std::mutex storage_mutex_;                                   // 存储互斥锁
    std::atomic<bool> running_{false};                          // 运行状态
    std::thread worker_thread_;                                  // 工作线程
    
    // 设备端队列指针
    DeviceQueue<KFE_STM_Slot, 32>* storage_queue_;
    DeviceQueue<std::string, 32>* query_queue_;
    DeviceQueue<KFE_STM_Slot, 32>* result_queue_;

    /**
     * @brief 工作线程函数
     */
    void workerThread() {
        while (running_) {
            // 处理存储请求
            KFE_STM_Slot slot;
            if (storage_queue_ && storage_queue_->pop(slot)) {
                std::lock_guard<std::mutex> lock(storage_mutex_);
                kfe_storage_[slot.hash()] = slot;
            }
            
            // 处理查询请求
            std::string query_hash;
            if (query_queue_ && query_queue_->pop(query_hash)) {
                std::lock_guard<std::mutex> lock(storage_mutex_);
                auto it = kfe_storage_.find(query_hash);
                if (it != kfe_storage_.end() && result_queue_) {
                    result_queue_->push(it->second);
                }
            }
            
            // 短暂休眠避免忙等待
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

public:
    /**
     * @brief 构造函数
     * 
     * @param storage_queue KFE存储请求队列
     * @param query_queue KFE查询请求队列
     * @param result_queue KFE查询结果队列
     */
    KFEManager(DeviceQueue<KFE_STM_Slot, 32>* storage_queue,
               DeviceQueue<std::string, 32>* query_queue,
               DeviceQueue<KFE_STM_Slot, 32>* result_queue)
        : storage_queue_(storage_queue)
        , query_queue_(query_queue)
        , result_queue_(result_queue) {
        start();
    }
    
    /**
     * @brief 析构函数
     */
    ~KFEManager() {
        stop();
    }
    
    /**
     * @brief 启动KFE管理器
     */
    void start() {
        running_ = true;
        worker_thread_ = std::thread(&KFEManager::workerThread, this);
    }
    
    /**
     * @brief 停止KFE管理器
     */
    void stop() {
        running_ = false;
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }
    
    /**
     * @brief 手动存储KFE槽位
     * 
     * @param slot 要存储的KFE槽位
     */
    void storeKFE(const KFE_STM_Slot& slot) {
        std::lock_guard<std::mutex> lock(storage_mutex_);
        kfe_storage_[slot.hash()] = slot;
    }
    
    /**
     * @brief 手动查询KFE槽位
     * 
     * @param hash KFE哈希值
     * @return KFE_STM_Slot 找到的KFE槽位，如果不存在则返回空槽位
     */
    KFE_STM_Slot findKFE(const std::string& hash) {
        std::lock_guard<std::mutex> lock(storage_mutex_);
        auto it = kfe_storage_.find(hash);
        if (it != kfe_storage_.end()) {
            return it->second;
        }
        return KFE_STM_Slot{};
    }
    
    /**
     * @brief 获取存储的KFE数量
     * 
     * @return size_t KFE槽位数量
     */
    size_t getKFECount() const {
        std::lock_guard<std::mutex> lock(storage_mutex_);
        return kfe_storage_.size();
    }
    
    /**
     * @brief 清空KFE存储
     */
    void clear() {
        std::lock_guard<std::mutex> lock(storage_mutex_);
        kfe_storage_.clear();
    }
};

#endif // SRC_KFEMANAGER_H