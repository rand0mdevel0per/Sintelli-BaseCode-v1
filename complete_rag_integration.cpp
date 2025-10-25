// 完整的RAG知识库集成示例
#include "rag_knowledge_loader.h"
#include "semantic_query_interface.h"
#include "math_reasoning_loader.cpp"
#include <iostream>
#include <vector>
#include <memory>

using namespace std;

// 完整的RAG集成管理器
class CompleteRAGManager {
private:
    std::unique_ptr<RAGKnowledgeBaseLoader> base_loader;
    std::unique_ptr<MathReasoningLoader> math_loader;
    std::unique_ptr<LogicSemanticMatcher> logic_matcher;
    std::unique_ptr<SemanticQueryInterface> query_interface;
    
public:
    CompleteRAGManager() 
        : base_loader(std::make_unique<RAGKnowledgeBaseLoader>())
        , math_loader(std::make_unique<MathReasoningLoader>())
        , logic_matcher(std::make_unique<LogicSemanticMatcher>("/models/e5/e5_large.onnx"))
        , query_interface(std::make_unique<SemanticQueryInterface>(*logic_matcher)) {}
    
    // 加载所有知识库
    bool loadAllKnowledgeBases() {
        cout << "🧠 开始加载所有知识库..." << endl;
        
        bool success = true;
        
        // 1. 加载预定义知识库
        cout << "\n📚 加载预定义知识库..." << endl;
        auto ml_knowledge = PredefinedKnowledgeLoader::loadMachineLearningKnowledge();
        auto cs_knowledge = PredefinedKnowledgeLoader::loadComputerScienceKnowledge();
        
        for (const auto& entry : ml_knowledge) {
            base_loader->addKnowledgeEntry(entry);
        }
        for (const auto& entry : cs_knowledge) {
            base_loader->addKnowledgeEntry(entry);
        }
        cout << "  ✅ 预定义知识库加载完成" << endl;
        
        // 2. 从JSON文件加载
        cout << "\n📁 从JSON文件加载..." << endl;
        if (base_loader->loadFromFile("predefined_knowledge.json", "技术知识")) {
            cout << "  ✅ JSON知识库加载成功" << endl;
        } else {
            cout << "  ⚠️  JSON文件加载失败，使用内置数据" << endl;
        }
        
        // 3. 加载数学推理数据集
        cout << "\n🧮 加载数学推理数据集..." << endl;
        if (math_loader->loadMathReasoningDataset("math_dataset_example.json", "Deepseek-Math-RL-7B")) {
            cout << "  ✅ 数学数据集加载成功" << endl;
        } else {
            cout << "  ⚠️  数学数据集加载失败，使用模拟数据" << endl;
        }
        
        return success;
    }
    
    // 生成完整的Logic树
    std::vector<LogicDescriptor> generateCompleteLogicTree() {
        cout << "\n🌳 生成完整的Logic树..." << endl;
        
        std::vector<LogicDescriptor> all_logics;
        
        // 从基础知识库生成Logic
        auto ml_logics = base_loader->generateLogicTreeFromCategory("机器学习", 10, 0.6);
        auto cs_logics = base_loader->generateLogicTreeFromCategory("计算机科学", 10, 0.6);
        auto tech_logics = base_loader->generateLogicTreeFromCategory("技术知识", 5, 0.5);
        
        // 从数学数据集生成Logic
        auto math_logics = math_loader->generateMathLogicTree(15, 0.7);
        auto reasoning_logics = math_loader->generateReasoningLogicTree(20, 0.8);
        
        // 合并所有Logic
        all_logics.insert(all_logics.end(), ml_logics.begin(), ml_logics.end());
        all_logics.insert(all_logics.end(), cs_logics.begin(), cs_logics.end());
        all_logics.insert(all_logics.end(), tech_logics.begin(), tech_logics.end());
        all_logics.insert(all_logics.end(), math_logics.begin(), math_logics.end());
        all_logics.insert(all_logics.end(), reasoning_logics.begin(), reasoning_logics.end());
        
        cout << "  生成 Logic 统计:" << endl;
        cout << "  - 机器学习: " << ml_logics.size() << " 个" << endl;
        cout << "  - 计算机科学: " << cs_logics.size() << " 个" << endl;
        cout << "  - 技术知识: " << tech_logics.size() << " 个" << endl;
        cout << "  - 数学推理: " << math_logics.size() << " 个" << endl;
        cout << "  - 推理路径: " << reasoning_logics.size() << " 个" << endl;
        cout << "  📊 总计: " << all_logics.size() << " 个Logic" << endl;
        
        return all_logics;
    }
    
    // 注册Logic到匹配器
    void registerAllLogics() {
        auto all_logics = generateCompleteLogicTree();
        
        // 注册到语义匹配器
        base_loader->registerLogicTree(*logic_matcher, all_logics);
        
        cout << "\n🔗 所有Logic已注册到语义匹配器" << endl;
    }
    
    // 智能查询接口
    std::vector<std::pair<LogicDescriptor, double>> 
    intelligentQuery(const std::string& query, 
                    double neuron_confidence = 0.8,
                    double similarity_threshold = 0.4) {
        
        cout << "\n🤖 智能查询: \"" << query << "\"" << endl;
        cout << "  置信度: " << neuron_confidence << ", 相似度阈值: " << similarity_threshold << endl;
        
        auto results = query_interface->query(query, neuron_confidence, similarity_threshold, true, true);
        
        if (results.empty()) {
            cout << "  ❌ 未找到相关Logic" << endl;
        } else {
            cout << "  ✅ 找到 " << results.size() << " 个相关Logic:" << endl;
            
            // 按类别分组显示
            std::map<std::string, std::vector<std::pair<LogicDescriptor, double>>> categorized_results;
            for (const auto& result : results) {
                categorized_results[result.first.category].push_back(result);
            }
            
            for (const auto& [category, category_results] : categorized_results) {
                cout << "  📂 " << category << " (" << category_results.size() << " 个):" << endl;
                for (const auto& [logic, similarity] : category_results) {
                    cout << "    - " << logic.logic_id << " (相似度: " << similarity << ")" << endl;
                    if (similarity >= logic.activation_threshold) {
                        cout << "      💡 激活阈值已满足，可注入神经元" << endl;
                    }
                }
            }
        }
        
        return results;
    }
    
    // 批量测试查询
    void batchQueryTest() {
        cout << "\n🧪 批量查询测试" << endl;
        cout << "================" << endl;
        
        std::vector<std::tuple<std::string, std::string, double>> test_cases = {
            {"神经网络", "机器学习", 0.8},
            {"解方程", "数学推理", 0.7},
            {"CUDA编程", "技术知识", 0.6},
            {"概率计算", "数学推理", 0.9},
            {"操作系统", "计算机科学", 0.5}
        };
        
        for (const auto& [query, expected_category, confidence] : test_cases) {
            intelligentQuery(query, confidence, 0.3);
        }
    }
    
    // 获取完整统计信息
    void showCompleteStats() {
        cout << "\n📊 完整的RAG系统统计" << endl;
        cout << "====================" << endl;
        
        // 基础知识库统计
        auto base_stats = base_loader->getKnowledgeStats();
        cout << "📚 基础知识库:" << endl;
        cout << "  总条目数: " << base_stats.total_entries << endl;
        cout << "  类别数量: " << base_stats.unique_categories << endl;
        for (const auto& [category, count] : base_stats.category_counts) {
            cout << "  - " << category << ": " << count << " 个条目" << endl;
        }
        
        // 数学数据集统计
        auto math_stats = math_loader->getStats();
        cout << "\n🧮 数学推理数据集:" << endl;
        cout << "  总条目数: " << math_stats.total_entries << endl;
        cout << "  类别数量: " << math_stats.unique_categories << endl;
        for (const auto& [category, count] : math_stats.category_counts) {
            cout << "  - " << category << ": " << count << " 个条目" << endl;
        }
        
        // 查询系统统计
        auto query_stats = query_interface->getStats();
        cout << "\n🔍 查询系统:" << endl;
        cout << "  缓存命中率: " << (query_stats.cache_stats.hit_rate * 100) << "%" << endl;
        cout << "  缓存条目数: " << query_stats.cache_stats.total_entries << endl;
        cout << "  队列大小: " << query_stats.queue_size << endl;
    }
    
    // 重置系统
    void resetSystem() {
        base_loader = std::make_unique<RAGKnowledgeBaseLoader>();
        math_loader = std::make_unique<MathReasoningLoader>();
        query_interface->reset();
        cout << "🔄 系统已重置" << endl;
    }
};

int main() {
    cout << "🚀 完整的RAG知识库集成演示" << endl;
    cout << "==========================" << endl;
    
    try {
        CompleteRAGManager rag_manager;
        
        // 1. 加载所有知识库
        if (!rag_manager.loadAllKnowledgeBases()) {
            cout << "❌ 知识库加载失败" << endl;
            return 1;
        }
        
        // 2. 注册Logic
        rag_manager.registerAllLogics();
        
        // 3. 显示统计信息
        rag_manager.showCompleteStats();
        
        // 4. 批量查询测试
        rag_manager.batchQueryTest();
        
        // 5. 交互式查询演示
        cout << "\n💬 交互式查询演示 (输入'quit'退出)" << endl;
        cout << "================================" << endl;
        
        string user_query;
        while (true) {
            cout << "\n请输入查询: ";
            getline(cin, user_query);
            
            if (user_query == "quit" || user_query == "退出") {
                break;
            }
            
            if (!user_query.empty()) {
                rag_manager.intelligentQuery(user_query, 0.8, 0.3);
            }
        }
        
        cout << "\n🎉 RAG系统演示完成!" << endl;
        cout << "\n💡 使用建议:" << endl;
        cout << "1. 可以下载真实的MATH数据集JSON文件替换示例文件" << endl;
        cout << "2. 根据需要调整相似度阈值和激活阈值" << endl;
        cout << "3. 可以集成更多开源数据集如CORAL、Open RAG Benchmark等" << endl;
        cout << "4. 通过查询统计优化缓存策略" << endl;
        
    } catch (const exception& e) {
        cerr << "❌ 系统错误: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}