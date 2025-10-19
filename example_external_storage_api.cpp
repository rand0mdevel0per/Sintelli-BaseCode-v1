// ExternalStorage API使用示例

#include "external_storage_api.h"
#include "isw.hpp"
#include "rag_knowledge_loader.h"
#include <iostream>
#include <memory>

int main() {
    // 创建ExternalStorage实例
    auto storage = std::make_shared<ExternalStorage<KnowledgeEntry>>();
    
    // 创建ExternalStorageAPI实例
    ExternalStorageAPI<KnowledgeEntry> api(storage);
    
    // 创建RAG知识库加载器
    RAGKnowledgeBaseLoader loader("", "", 100, 0.3, 2000);
    
    // 加载一些预定义的知识
    auto cs_knowledge = PredefinedKnowledgeLoader::loadComputerScienceKnowledge();
    for (const auto& entry : cs_knowledge) {
        loader.addKnowledgeEntry(entry);
    }
    
    auto math_knowledge = PredefinedKnowledgeLoader::loadMathematicsKnowledge();
    for (const auto& entry : math_knowledge) {
        loader.addKnowledgeEntry(entry);
    }
    
    // 将知识库内容插入到ExternalStorage中
    bool success = api.insertFromRAGLoader(loader);
    
    if (success) {
        std::cout << "✅ 成功将知识库内容插入到ExternalStorage中" << std::endl;
    } else {
        std::cout << "❌ 插入知识库内容失败" << std::endl;
    }
    
    // 获取存储统计信息
    auto stats = storage->getStatistics();
    std::cout << "存储统计信息:" << std::endl;
    std::cout << "  L2内存池大小: " << stats.l2_size << std::endl;
    std::cout << "  总数据条数: " << stats.total_size << std::endl;
    std::cout << "  平均热度: " << stats.avg_heat << std::endl;
    
    // 获取最热的3个数据
    auto hottest = storage->getHottestK(3);
    std::cout << "最热的3个数据:" << std::endl;
    for (size_t i = 0; i < hottest.size(); ++i) {
        KnowledgeEntry entry;
        if (storage->fetch(hottest[i], entry)) {
            auto desc_opt = storage->getDescriptor(hottest[i]);
            if (desc_opt.has_value()) {
                std::cout << "  " << (i+1) << ". " << entry.title << " (热度: " << 
                         desc_opt.value().heat << ")" << std::endl;
            } else {
                std::cout << "  " << (i+1) << ". " << entry.title << " (热度: 未知)" << std::endl;
            }
        }
    }
    
    return 0;
}