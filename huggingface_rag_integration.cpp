#include "rag_knowledge_loader.h"
#include "isw.hpp"
#include "semantic_query_engine.h"
#include <iostream>
#include <memory>

using namespace std;

int main() {
    cout << "🚀 HuggingFace RAG集成演示" << endl;
    cout << "========================" << endl;
    
    // 1. 创建RAG知识库加载器
    RAGKnowledgeBaseLoader loader;
    
    // 2. 创建外部存储
    auto external_storage = make_shared<ExternalStorage<KnowledgeEntry>>();
    loader.setExternalStorage(external_storage);
    
    // 3. 设置最大存储大小
    loader.setMaxStorageSize(5000); // 限制为5000个条目
    
    // 4. 演示HuggingFace数据集流式解析
    cout << "\n1️⃣ 演示HuggingFace数据集流式解析:" << endl;
    bool stream_success = loader.streamHuggingFaceDataset(
        "HuggingFaceFW/fineweb", 
        "sample-10BT", 
        "train", 
        10,  // 获取10个条目
        "web_content"
    );
    
    if (stream_success) {
        cout << "✅ 流式解析成功" << endl;
    } else {
        cout << "❌ 流式解析失败" << endl;
    }
    
    // 5. 演示HuggingFace数据集查询
    cout << "\n2️⃣ 演示HuggingFace数据集查询:" << endl;
    bool query_success = loader.queryAndLoadFromHFDataset(
        "人工智能", 
        "HuggingFaceFW/fineweb", 
        "sample-10BT", 
        5,  // 获取5个相关条目
        "ai_research"
    );
    
    if (query_success) {
        cout << "✅ 数据集查询成功" << endl;
    } else {
        cout << "❌ 数据集查询失败" << endl;
    }
    
    // 6. 演示Logic匹配不足时自动获取数据
    cout << "\n3️⃣ 演示Logic匹配不足时自动获取数据:" << endl;
    bool auto_fetch_success = loader.autoFetchDataWhenLogicInsufficient(
        "深度学习", 
        20,  // 需要至少20个匹配项
        "HuggingFaceFW/fineweb", 
        "sample-10BT"
    );
    
    if (auto_fetch_success) {
        cout << "✅ 自动获取数据成功" << endl;
    } else {
        cout << "❌ 自动获取数据失败" << endl;
    }
    
    // 7. 获取知识库统计信息
    auto stats = loader.getKnowledgeStats();
    cout << "\n4️⃣ 知识库统计信息:" << endl;
    cout << "   总条目数: " << stats.total_entries << endl;
    cout << "   类别数: " << stats.unique_categories << endl;
    cout << "   平均相关性分数: " << stats.avg_relevance_score << endl;
    
    // 显示各类别条目数
    cout << "   各类别条目数:" << endl;
    for (const auto& pair : stats.category_counts) {
        cout << "     " << pair.first << ": " << pair.second << endl;
    }
    
    // 8. 演示外部存储功能
    cout << "\n5️⃣ 演示外部存储功能:" << endl;
    auto entries = loader.getAllEntries();
    if (!entries.empty()) {
        cout << "   将前10个条目插入外部存储..." << endl;
        int count = 0;
        vector<KnowledgeEntry> first_entries;
        for (const auto& entry : entries) {
            if (count++ >= 10) break;
            first_entries.push_back(entry);
        }
        
        if (loader.insertToExternalStorage(first_entries)) {
            cout << "✅ 成功插入 " << first_entries.size() << " 个条目到外部存储" << endl;
        } else {
            cout << "❌ 插入外部存储失败" << endl;
        }
    }
    
    // 9. 检查并清理存储
    cout << "\n6️⃣ 检查并清理存储:" << endl;
    bool cleanup_triggered = loader.checkAndCleanupStorage();
    if (cleanup_triggered) {
        cout << "✅ 存储清理已触发" << endl;
    } else {
        cout << "ℹ️  存储大小在限制范围内，无需清理" << endl;
    }
    
    // 10. 获取外部存储统计信息
    if (external_storage) {
        auto storage_stats = external_storage->getStatistics();
        cout << "\n7️⃣ 外部存储统计信息:" << endl;
        cout << "   L2内存池大小: " << storage_stats.l2_size << endl;
        cout << "   L3磁盘存储大小: " << storage_stats.l3_size << endl;
        cout << "   总大小: " << storage_stats.total_size << endl;
        cout << "   平均热度: " << storage_stats.avg_heat << endl;
        cout << "   最大热度: " << storage_stats.max_heat << endl;
        cout << "   最小热度: " << storage_stats.min_heat << endl;
    }
    
    // 11. 演示语义搜索功能
    cout << "\n8️⃣ 演示语义搜索功能:" << endl;
    if (!entries.empty()) {
        // 创建语义查询引擎
        SemanticQueryEngine semantic_engine("/models/e5/e5_large.onnx");
        
        if (semantic_engine.isInitialized()) {
            cout << "✅ 语义查询引擎初始化成功" << endl;
            
            // 对前几个条目进行语义搜索
            vector<string> candidates;
            int candidate_count = 0;
            for (const auto& entry : entries) {
                if (candidate_count++ >= 5) break;
                candidates.push_back(entry.content);
            }
            
            // 执行语义搜索
            auto search_results = semantic_engine.semanticSearch(
                "机器学习算法", 
                candidates, 
                3,  // 返回前3个最相似的
                0.1 // 相似度阈值
            );
            
            cout << "   语义搜索结果:" << endl;
            for (const auto& result : search_results) {
                cout << "     相似度: " << result.second 
                     << ", 条目索引: " << result.first << endl;
            }
        } else {
            cout << "⚠️  语义查询引擎初始化失败，使用模拟实现" << endl;
        }
    }
    
    cout << "\n🎉 HuggingFace RAG集成演示完成!" << endl;
    cout << "\n💡 主要功能特点:" << endl;
    cout << "   • 🌊 HuggingFace数据集流式解析" << endl;
    cout << "   • 🔍 HuggingFace数据集语义查询" << endl;
    cout << "   • 🤖 Logic匹配不足时自动获取数据" << endl;
    cout << "   • 💾 外部存储集成 (L2内存 + L3磁盘)" << endl;
    cout << "   • 🧹 智能存储管理 (自动清理)" << endl;
    cout << "   • 🧠 语义查询支持 (E5模型)" << endl;
    cout << "   • 📊 实时统计和监控" << endl;
    
    return 0;
}