// 简单的ExternalStorage API测试程序

#include "isw.hpp"
#include "external_storage_api.h"
#include <iostream>
#include <memory>
#include <string>

// 简单的测试数据结构
struct TestEntry {
    std::string id;
    std::string content;
    
    TestEntry() = default;
    TestEntry(const std::string& i, const std::string& c) : id(i), content(c) {}
    
    // 实现hash方法
    std::string hash() const {
        return id;
    }
    
    // 实现getFeature方法
    FeatureVector<float> getFeature() const {
        return FeatureVector<float>();
    }
};

int main() {
    std::cout << "开始测试ExternalStorage API..." << std::endl;
    
    // 创建ExternalStorage实例
    auto storage = std::make_shared<ExternalStorage<TestEntry>>();
    
    // 创建ExternalStorageAPI实例
    ExternalStorageAPI<TestEntry> api(storage);
    
    // 创建一些测试数据
    TestEntry entry1("001", "这是第一条测试数据");
    TestEntry entry2("002", "这是第二条测试数据");
    TestEntry entry3("003", "这是第三条测试数据");
    
    // 插入数据
    uint64_t slot1 = api.insertKnowledgeEntry(entry1);
    uint64_t slot2 = api.insertKnowledgeEntry(entry2);
    uint64_t slot3 = api.insertKnowledgeEntry(entry3);
    
    std::cout << "插入数据完成:" << std::endl;
    std::cout << "  entry1 slot_id: " << slot1 << std::endl;
    std::cout << "  entry2 slot_id: " << slot2 << std::endl;
    std::cout << "  entry3 slot_id: " << slot3 << std::endl;
    
    // 获取存储统计信息
    auto stats = storage->getStatistics();
    std::cout << "存储统计信息:" << std::endl;
    std::cout << "  L2内存池大小: " << stats.l2_size << std::endl;
    std::cout << "  总数据条数: " << stats.total_size << std::endl;
    
    // 获取最热的数据
    auto hottest = storage->getHottestK(3);
    std::cout << "最热的数据数量: " << hottest.size() << std::endl;
    
    // 获取最冷的数据
    auto coldest = storage->getColdestK(3);
    std::cout << "最冷的数据数量: " << coldest.size() << std::endl;
    
    std::cout << "测试完成!" << std::endl;
    
    return 0;
}