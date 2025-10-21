#include "isw.hpp"
#include "structs.h"
#include <iostream>

int main() {
    // 测试ExternalStorage<KFE_STM_Slot>是否可以正常创建和使用
    ExternalStorage<KFE_STM_Slot> storage(1024, 100.0, 1.0);
    
    // 创建一个KFE_STM_Slot对象
    KFE_STM_Slot slot;
    slot.Ulocal = 1.0;
    slot.Rcycles = 10;
    slot.Icore = 0.5;
    
    // 初始化Vmem数组
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            slot.Vmem[i][j] = 0.0;
        }
    }
    
    // 初始化conv16数组
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            slot.conv16[i][j] = 0.0;
        }
    }
    
    slot.V = 1.0;
    
    // 测试存储功能
    uint64_t slot_id = storage.store(slot, 1.0);
    std::cout << "Stored slot with ID: " << slot_id << std::endl;
    
    // 测试获取功能
    KFE_STM_Slot retrieved_slot;
    if (storage.fetch(slot_id, retrieved_slot)) {
        std::cout << "Successfully retrieved slot" << std::endl;
        std::cout << "Ulocal: " << retrieved_slot.Ulocal << std::endl;
        std::cout << "Rcycles: " << retrieved_slot.Rcycles << std::endl;
        std::cout << "Icore: " << retrieved_slot.Icore << std::endl;
    } else {
        std::cout << "Failed to retrieve slot" << std::endl;
    }
    
    return 0;
}