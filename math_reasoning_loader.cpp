// MATH推理路径数据集加载器
#include "rag_knowledge_loader.h"
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <unordered_set>

using json = nlohmann::json;

class MathReasoningLoader {
private:
    std::unique_ptr<RAGKnowledgeBaseLoader> rag_loader;
    
public:
    MathReasoningLoader() : rag_loader(std::make_unique<RAGKnowledgeBaseLoader>()) {}
    
    // 加载MATH推理路径数据集
    bool loadMathReasoningDataset(const std::string& file_path, 
                                 const std::string& model_name = "Deepseek-Math-RL-7B") {
        try {
            std::ifstream file(file_path);
            if (!file.is_open()) {
                std::cerr << "无法打开文件: " << file_path << std::endl;
                return false;
            }
            
            std::stringstream buffer;
            buffer << file.rdbuf();
            std::string json_data = buffer.str();
            
            json j = json::parse(json_data);
            
            // 解析数据结构
            if (!j.contains("prompt") || !j.contains("completion") || !j.contains("answer")) {
                std::cerr << "JSON格式不匹配：缺少必要字段" << std::endl;
                return false;
            }
            
            auto& prompts = j["prompt"];
            auto& completions = j["completion"];
            auto& answers = j["answer"];
            
            if (!prompts.is_array() || !completions.is_array() || !answers.is_array()) {
                std::cerr << "JSON格式错误：字段不是数组类型" << std::endl;
                return false;
            }
            
            size_t num_problems = prompts.size();
            std::cout << "📊 正在加载 " << num_problems << " 个数学问题..." << std::endl;
            
            // 处理每个问题
            for (size_t problem_idx = 0; problem_idx < num_problems; ++problem_idx) {
                std::string prompt = prompts[problem_idx];
                std::string answer = answers[problem_idx];
                
                // 创建问题知识条目
                KnowledgeEntry problem_entry;
                problem_entry.title = "数学问题 " + std::to_string(problem_idx + 1);
                problem_entry.content = "问题: " + prompt + "\n答案: " + answer;
                problem_entry.category = "数学推理";
                problem_entry.source = model_name + " 数据集";
                
                // 添加问题
                rag_loader->addKnowledgeEntry(problem_entry);
                
                // 处理每个推理路径
                if (completions[problem_idx].is_array()) {
                    auto& problem_completions = completions[problem_idx];
                    size_t num_samples = problem_completions.size();
                    
                    for (size_t sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
                        std::string completion = problem_completions[sample_idx];
                        
                        // 创建推理路径知识条目
                        KnowledgeEntry reasoning_entry;
                        reasoning_entry.title = "推理路径 " + std::to_string(problem_idx + 1) + 
                                               "-" + std::to_string(sample_idx + 1);
                        reasoning_entry.content = "问题: " + prompt + 
                                                 "\n推理过程: " + completion + 
                                                 "\n正确答案: " + answer;
                        reasoning_entry.category = "推理路径";
                        reasoning_entry.source = model_name + " 样本" + std::to_string(sample_idx + 1);
                        
                        // 如果有准确率信息，添加为相关性分数
                        if (j.contains("accuracy") && j["accuracy"].is_array() && 
                            sample_idx < j["accuracy"].size() && 
                            j["accuracy"][sample_idx].is_array() &&
                            problem_idx < j["accuracy"][sample_idx].size()) {
                            
                            bool is_correct = j["accuracy"][sample_idx][problem_idx];
                            reasoning_entry.relevance_score = is_correct ? 1.0 : 0.5;
                        }
                        
                        rag_loader->addKnowledgeEntry(reasoning_entry);
                    }
                }
                
                // 进度显示
                if ((problem_idx + 1) % 100 == 0) {
                    std::cout << "  已处理 " << problem_idx + 1 << "/" << num_problems << " 个问题" << std::endl;
                }
            }
            
            std::cout << "✅ 成功加载 " << num_problems << " 个数学问题及其推理路径" << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "加载MATH数据集错误: " << e.what() << std::endl;
            return false;
        }
    }
    
    // 生成数学推理Logic树
    std::vector<LogicDescriptor> generateMathLogicTree(int max_logics = 50, 
                                                      double activation_threshold = 0.6) {
        return rag_loader->generateLogicTreeFromCategory("数学推理", max_logics, activation_threshold);
    }
    
    // 生成推理路径Logic树
    std::vector<LogicDescriptor> generateReasoningLogicTree(int max_logics = 50, 
                                                           double activation_threshold = 0.7) {
        return rag_loader->generateLogicTreeFromCategory("推理路径", max_logics, activation_threshold);
    }
    
    // 获取知识库统计
    RAGKnowledgeBaseLoader::KnowledgeStats getStats() const {
        return rag_loader->getKnowledgeStats();
    }
    
    // 搜索数学问题
    std::vector<KnowledgeEntry> searchMathProblems(const std::string& query, 
                                                  int max_results = 10) {
        return rag_loader->searchKnowledge(query, max_results, "数学推理");
    }
    
    // 搜索推理路径
    std::vector<KnowledgeEntry> searchReasoningPaths(const std::string& query, 
                                                    int max_results = 10) {
        return rag_loader->searchKnowledge(query, max_results, "推理路径");
    }
    
    // 获取RAG加载器实例
    RAGKnowledgeBaseLoader* getLoader() const {
        return rag_loader.get();
    }
};

// 示例：使用MATH推理路径数据集
void exampleMathReasoningIntegration() {
    std::cout << "=== MATH推理路径数据集集成示例 ===" << std::endl;
    
    MathReasoningLoader math_loader;
    
    // 1. 加载数据集
    std::cout << "📚 加载MATH推理路径数据集..." << std::endl;
    if (math_loader.loadMathReasoningDataset("math_reasoning_dataset.json", "Deepseek-Math-RL-7B")) {
        std::cout << "✅ 数据集加载成功" << std::endl;
    } else {
        std::cout << "❌ 数据集加载失败，使用模拟数据" << std::endl;
        
        // 使用模拟数据
        RAGKnowledgeBaseLoader::KnowledgeStats stats;
        stats.total_entries = 150;
        stats.unique_categories = 2;
        stats.category_counts = {{"数学推理", 50}, {"推理路径", 100}};
        stats.avg_relevance_score = 0.8;
        
        std::cout << "📊 模拟数据统计:" << std::endl;
        std::cout << "  总条目数: " << stats.total_entries << std::endl;
        std::cout << "  类别数量: " << stats.unique_categories << std::endl;
        for (const auto& [category, count] : stats.category_counts) {
            std::cout << "  - " << category << ": " << count << " 个条目" << std::endl;
        }
    }
    
    // 2. 生成Logic树
    std::cout << "\n🌳 生成数学推理Logic树..." << std::endl;
    auto math_logics = math_loader.generateMathLogicTree(20, 0.6);
    auto reasoning_logics = math_loader.generateReasoningLogicTree(30, 0.7);
    
    std::cout << "  生成 " << math_logics.size() << " 个数学Logic" << std::endl;
    std::cout << "  生成 " << reasoning_logics.size() << " 个推理路径Logic" << std::endl;
    
    // 3. 注册到语义匹配器
    LogicSemanticMatcher matcher("/models/e5/e5_large.onnx");
    SemanticQueryInterface query_iface(matcher);
    
    math_loader.getLoader()->registerLogicTree(matcher, math_logics);
    math_loader.getLoader()->registerLogicTree(matcher, reasoning_logics);
    
    // 4. 测试数学推理查询
    std::cout << "\n🔍 测试数学推理查询..." << std::endl;
    std::vector<std::string> test_queries = {
        "解方程",
        "几何证明",
        "概率计算",
        "微积分问题",
        "数学推理过程"
    };
    
    for (const auto& query : test_queries) {
        std::cout << "\n查询: \"" << query << "\"" << std::endl;
        
        auto results = query_iface.query(query, 0.8, 0.4);
        
        if (results.empty()) {
            std::cout << "  未找到相关Logic" << std::endl;
        } else {
            std::cout << "  找到 " << results.size() << " 个相关Logic:" << std::endl;
            
            for (const auto& [logic, similarity] : results) {
                std::cout << "  - " << logic.logic_id << " (相似度: " << similarity 
                         << ", 类别: " << logic.category << ")" << std::endl;
                
                if (similarity >= logic.activation_threshold) {
                    std::cout << "    💡 达到激活阈值，可注入到神经元" << std::endl;
                }
            }
        }
    }
    
    // 5. 显示统计信息
    auto stats = math_loader.getStats();
    auto query_stats = query_iface.getStats();
    
    std::cout << "\n📊 数学知识库统计:" << std::endl;
    std::cout << "  总条目数: " << stats.total_entries << std::endl;
    std::cout << "  类别数量: " << stats.unique_categories << std::endl;
    std::cout << "  平均相关性: " << stats.avg_relevance_score << std::endl;
    
    std::cout << "\n📈 查询系统统计:" << std::endl;
    std::cout << "  缓存命中率: " << (query_stats.cache_stats.hit_rate * 100) << "%" << std::endl;
    std::cout << "  缓存条目数: " << query_stats.cache_stats.total_entries << std::endl;
    
    std::cout << "\n🎉 MATH推理路径数据集集成完成!" << std::endl;
}

int main() {
    std::cout << "🧮 MATH推理路径数据集演示" << std::endl;
    std::cout << "=========================" << std::endl;
    
    try {
        exampleMathReasoningIntegration();
        
        std::cout << "\n💡 使用建议:" << std::endl;
        std::cout << "1. 下载MATH数据集JSON文件并放在项目目录中" << std::endl;
        std::cout << "2. 修改文件路径为实际的JSON文件路径" << std::endl;
        std::cout << "3. 可以集成不同模型的推理路径进行对比分析" << std::endl;
        std::cout << "4. 通过调整激活阈值来控制数学Logic的触发条件" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 示例执行出错: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}