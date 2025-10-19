#include <iostream>
#include "deviceQueue.cpp"
#include "structs.h"

// 测试DeviceQueue的基本功能
void test_device_queue() {
    DeviceQueue<int, 10> queue;
    queue.init();
    
    // 测试push和pop
    bool success = queue.push(42);
    if (success) {
        std::cout << "Push successful" << std::endl;
    }
    
    int value;
    success = queue.pop(value);
    if (success) {
        std::cout << "Pop successful, value: " << value << std::endl;
    }
}

int main() {
    std::cout << "Testing DeviceQueue implementation..." << std::endl;
    test_device_queue();
    std::cout << "Test completed" << std::endl;
    return 0;
}