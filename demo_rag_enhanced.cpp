#include "rag_knowledge_loader.h"
#include "isw.hpp"
#include "semantic_query_engine.h"
#include <iostream>
#include <memory>

using namespace std;

int main() {
    cout << "🚀 RAG知识库加载器增强版演示" << endl;
    cout << "============================" << endl;
    
    // 1. 创建RAG知识库加载器
    RAGKnowledgeBaseLoader loader;
    
    // 2. 创建外部存储
    auto external_storage = make_shared<ExternalStorage<KnowledgeEntry>>();
    loader.setExternalStorage(external_storage);
    
    // 3. 设置最大存储大小
    loader.setMaxStorageSize(1000); // 限制为1000个条目
    
    // 4. 演示HuggingFace数据集流式解析
    cout << "\n1️⃣ 演示HuggingFace数据集流式解析:" << endl;
    loader.streamHuggingFaceDataset("HuggingFaceFW/fineweb", "sample-10BT", "train", 5, "huggingface_demo");
    
    // 5. 演示HuggingFace数据集查询
    cout << "\n2️⃣ 演示HuggingFace数据集查询:" << endl;
    loader.queryAndLoadFromHFDataset("神经网络", "HuggingFaceFW/fineweb", "sample-10BT", 3, "huggingface_query");
    
    // 6. 演示Logic匹配不足时自动获取数据
    cout << "\n3️⃣ 演示Logic匹配不足时自动获取数据:" << endl;
    loader.autoFetchDataWhenLogicInsufficient("机器学习", 10, "HuggingFaceFW/fineweb", "sample-10BT");
    
    // 7. 获取知识库统计信息
    auto stats = loader.getKnowledgeStats();
    cout << "\n4️⃣ 知识库统计信息:" << endl;
    cout << "   总条目数: " << stats.total_entries << endl;
    cout << "   类别数: " << stats.unique_categories << endl;
    cout << "   平均相关性分数: " << stats.avg_relevance_score << endl;
    
    // 8. 演示外部存储功能
    cout << "\n5️⃣ 演示外部存储功能:" << endl;
    auto entries = loader.getAllEntries();
    if (!entries.empty()) {
        // 插入前5个条目到外部存储
        int count = 0;
        for (const auto& entry : entries) {
            if (count++ >= 5) break;
            loader.insertToExternalStorage(entry);
        }
    }
    
    // 9. 检查并清理存储
    cout << "\n6️⃣ 检查并清理存储:" << endl;
    loader.checkAndCleanupStorage();
    
    // 10. 获取外部存储统计信息
    if (external_storage) {
        auto storage_stats = external_storage->getStatistics();
        cout << "\n7️⃣ 外部存储统计信息:" << endl;
        cout << "   L2内存池大小: " << storage_stats.l2_size << endl;
        cout << "   L3磁盘存储大小: " << storage_stats.l3_size << endl;
        cout << "   总大小: " << storage_stats.total_size << endl;
        cout << "   平均热度: " << storage_stats.avg_heat << endl;
    }
    
    cout << "\n🎉 演示完成!" << endl;
    cout << "\n💡 主要功能特点:" << endl;
    cout << "   • 支持HuggingFace数据集流式解析" << endl;
    cout << "   • 支持HuggingFace数据集查询" << endl;
    cout << "   • Logic匹配不足时自动获取数据" << endl;
    cout << "   • 外部存储集成 (L2内存 + L3磁盘)" << endl;
    cout << "   • 智能存储管理 (自动清理)" << endl;
    cout << "   • 语义查询支持" << endl;
    
    return 0;
}