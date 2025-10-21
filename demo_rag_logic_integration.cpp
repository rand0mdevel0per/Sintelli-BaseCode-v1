#include "rag_knowledge_loader.h"
#include "semantic_matcher.h"
#include "isw.hpp"
#include "structs.h"
#include <iostream>
#include <memory>

using namespace std;

int main() {
    cout << "🚀 RAG系统与Logic系统集成演示" << endl;
    cout << "============================" << endl;
    
    // 1. 创建RAG知识库加载器
    RAGKnowledgeBaseLoader rag_loader;
    
    // 2. 创建ExternalStorage实例
    auto rag_storage = make_shared<ExternalStorage<KnowledgeEntry>>();
    rag_loader.setExternalStorage(rag_storage);
    
    // 3. 设置最大存储大小
    rag_loader.setMaxStorageSize(1000);
    
    // 4. 创建Logic系统组件
    auto logic_descriptor_storage = make_shared<ExternalStorage<LogicDescriptor>>();
    LogicInjector logic_injector(logic_descriptor_storage.get());
    
    // 5. 创建实际Logic内容存储
    ExternalStorage<Logic> logic_tree;
    
    // 6. 演示基本的RAG功能
    cout << "\n1️⃣ 演示基本RAG功能:" << endl;
    rag_loader.streamHuggingFaceDataset("HuggingFaceFW/fineweb", "sample-10BT", "train", 5, "demo_category");
    
    // 7. 将RAG知识注册为Logic
    cout << "\n2️⃣ 将RAG知识注册为Logic:" << endl;
    rag_loader.registerKnowledgeAsLogic(&logic_injector, &logic_tree, "rag_demo");
    
    // 8. 演示自动获取并注册Logic
    cout << "\n3️⃣ 演示自动获取并注册Logic:" << endl;
    rag_loader.autoFetchAndRegisterLogic(&logic_injector, &logic_tree, "人工智能", 3, "HuggingFaceFW/fineweb", "sample-10BT", "auto_rag");
    
    // 9. 演示语义搜索
    cout << "\n4️⃣ 演示语义搜索:" << endl;
    auto search_results = rag_loader.semanticSearchKnowledge("机器学习", 5, 0.3);
    cout << "   找到 " << search_results.size() << " 个相关条目" << endl;
    
    // 10. 演示Logic匹配
    cout << "\n5️⃣ 演示Logic匹配:" << endl;
    auto matched_logics = logic_injector.findMatchingLogicIds("人工智能技术");
    cout << "   找到 " << matched_logics.size() << " 个匹配的Logic描述符" << endl;
    
    // 11. 演示从logic_tree获取实际Logic内容
    cout << "\n6️⃣ 演示获取实际Logic内容:" << endl;
    for (const auto& logic_pair : matched_logics) {
        const string& logic_id = logic_pair.first;
        double similarity = logic_pair.second;
        
        // 创建Logic对象用于查询
        Logic temp_logic;
        temp_logic.Rcycles = 0;
        temp_logic.importance = similarity;
        // 注意：这里简化处理，实际应该使用正确的hash机制
        
        cout << "   Logic ID: " << logic_id << " (相似度: " << similarity << ")" << endl;
    }
    
    // 12. 显示存储统计信息
    cout << "\n7️⃣ 存储统计信息:" << endl;
    if (rag_storage) {
        auto rag_stats = rag_storage->getStatistics();
        cout << "   RAG存储 - L2: " << rag_stats.l2_size << ", L3: " << rag_stats.l3_size 
             << ", 总计: " << rag_stats.total_size << endl;
    }
    
    if (logic_descriptor_storage) {
        auto logic_stats = logic_descriptor_storage->getStatistics();
        cout << "   Logic描述符存储 - L2: " << logic_stats.l2_size << ", L3: " << logic_stats.l3_size 
             << ", 总计: " << logic_stats.total_size << endl;
    }
    
    cout << "\n🎉 RAG系统与Logic系统集成演示完成!" << endl;
    cout << "\n💡 集成特点:" << endl;
    cout << "   • RAG系统自动获取知识并转换为Logic" << endl;
    cout << "   • Logic描述符存储在logic_storage中用于语义匹配" << endl;
    cout << "   • Logic内容存储在logic_tree中用于实际执行" << endl;
    cout << "   • 两个存储系统协同工作，各司其职" << endl;
    cout << "   • 支持热度管理和自动清理" << endl;
    cout << "   • 实现智能的知识扩展和匹配" << endl;
    
    return 0;
}