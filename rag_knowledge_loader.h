// RAG知识库加载器 - 用于从外部知识库初始化Logic树
// 支持多种数据源格式和自动Logic生成

#ifndef RAG_KNOWLEDGE_LOADER_H
#define RAG_KNOWLEDGE_LOADER_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <optional>
#include "openai_client.h"
#include "semantic_query_interface.h"
#include "semantic_matcher.h"
#include "isw.hpp"
#include "semantic_query_engine.h"

// 知识库条目结构
struct KnowledgeEntry {
    std::string title;           // 标题
    std::string content;         // 内容
    std::string category;        // 类别
    std::string source;          // 数据源
    double relevance_score;      // 相关性分数
    std::vector<std::string> tags; // 标签
    
    KnowledgeEntry(const std::string& t = "", const std::string& c = "", 
                  const std::string& cat = "", const std::string& src = "")
        : title(t), content(c), category(cat), source(src), relevance_score(0.0) {}
};

// RAG知识库加载器类
class RAGKnowledgeBaseLoader {
private:
    std::unique_ptr<OpenAIClient::HttpClient> openai_client;
    std::unique_ptr<LogicSemanticMatcher> logic_matcher;
    std::unique_ptr<SemanticQueryEngine> semantic_engine;  // 语义查询引擎
    std::vector<KnowledgeEntry> knowledge_base;
    std::map<std::string, std::vector<KnowledgeEntry>> category_to_entries;
    
    // ExternalStorage集成
    std::shared_ptr<ExternalStorage<KnowledgeEntry>> external_storage;
    
    // 配置参数
    int max_entries_per_category;
    double min_relevance_threshold;
    int max_content_length;
    size_t max_storage_size;  // 最大存储大小限制
    
public:
    RAGKnowledgeBaseLoader(const std::string& openai_api_key = "",
                          const std::string& base_url = "https://api.openai.com/v1",
                          int max_entries = 100,
                          double min_relevance = 0.3,
                          int max_length = 2000);
    
    // === 数据源加载方法 ===
    
    // 从URL加载知识库（支持Wikipedia、arXiv等）
    bool loadFromURL(const std::string& url, const std::string& category = "general");
    
    // 从本地文件加载知识库
    bool loadFromFile(const std::string& file_path, const std::string& category = "general");
    
    // 从API端点加载知识库
    bool loadFromAPI(const std::string& api_endpoint, 
                    const std::string& query, 
                    const std::string& category = "general",
                    const std::map<std::string, std::string>& params = {});
    
    // arXiv API专用方法
    bool fetchArxivPapers(const std::string& query, 
                         const std::string& category = "学术论文",
                         int max_results = 10);
    
    // 从JSON数据加载知识库
    bool loadFromJSON(const std::string& json_data, const std::string& category = "general");
    
    // 从CSV数据加载知识库
    bool loadFromCSV(const std::string& csv_data, const std::string& category = "general");
    
    // === Hugging Face数据集流式解析方法 ===
    
    // 流式解析Hugging Face数据集
    bool streamHuggingFaceDataset(const std::string& dataset_name, 
                                 const std::string& subset = "default",
                                 const std::string& split = "train",
                                 int max_entries = 100,
                                 const std::string& category = "huggingface");
    
    // 从Hugging Face数据集查询并加载相关条目
    bool queryAndLoadFromHFDataset(const std::string& query,
                                  const std::string& dataset_name = "HuggingFaceFW/fineweb",
                                  const std::string& subset = "sample-10BT",
                                  int max_results = 50,
                                  const std::string& category = "huggingface_query");
    
    // 在Logic匹配不足时自动获取数据
    bool autoFetchDataWhenLogicInsufficient(const std::string& query,
                                          int min_required_matches = 5,
                                          const std::string& dataset_name = "HuggingFaceFW/fineweb",
                                          const std::string& subset = "sample-10BT");
    
    // === Logic生成方法 ===
    
    // 自动生成Logic树
    std::vector<LogicDescriptor> generateLogicTreeFromKnowledge(
        const std::string& knowledge_text, 
        const std::string& category = "general",
        double activation_threshold = 0.5);
    
    // 批量生成Logic树
    std::vector<LogicDescriptor> generateLogicTreeFromCategory(
        const std::string& category,
        int max_logics = 50,
        double activation_threshold = 0.5);
    
    // 使用AI辅助生成Logic描述
    std::string generateLogicDescriptionWithAI(const std::string& knowledge_content);
    
    // 注册Logic到语义匹配器
    bool registerLogicTree(LogicSemanticMatcher& matcher, 
                          const std::vector<LogicDescriptor>& logics);
    
    // === 知识库管理 ===
    
    // 添加知识条目
    bool addKnowledgeEntry(const KnowledgeEntry& entry);
    
    // 搜索知识库
    std::vector<KnowledgeEntry> searchKnowledge(const std::string& query, 
                                               int max_results = 10,
                                               const std::string& category = "");
    
    // 获取知识库统计信息
    struct KnowledgeStats {
        size_t total_entries;
        std::map<std::string, size_t> category_counts;
        size_t unique_categories;
        double avg_relevance_score;
    };
    
    KnowledgeStats getKnowledgeStats() const;
    
    // 清空知识库
    void clearKnowledgeBase();
    
    // 导出知识库到文件
    bool exportToFile(const std::string& file_path, const std::string& format = "json");
    
    // 获取所有知识条目
    const std::vector<KnowledgeEntry>& getAllEntries() const { return knowledge_base; }
    
    // 按类别获取知识条目
    std::vector<KnowledgeEntry> getEntriesByCategory(const std::string& category) const;
    
    // === 配置方法 ===
    
    // 设置OpenAI客户端
    void setOpenAIClient(std::unique_ptr<OpenAIClient::HttpClient> client);
    
    // 设置Logic匹配器
    void setLogicMatcher(std::unique_ptr<LogicSemanticMatcher> matcher);
    
    // 配置参数
    void setMaxEntriesPerCategory(int max_entries);
    void setMinRelevanceThreshold(double threshold);
    void setMaxContentLength(int max_length);
    void setMaxStorageSize(size_t max_size);
    
    // ExternalStorage相关方法
    void setExternalStorage(std::shared_ptr<ExternalStorage<KnowledgeEntry>> storage);
    std::shared_ptr<ExternalStorage<KnowledgeEntry>> getExternalStorage() const;
    bool insertToExternalStorage(const KnowledgeEntry& entry);
    bool insertToExternalStorage(const std::vector<KnowledgeEntry>& entries);
    bool checkAndCleanupStorage();
    void cleanupL3Cache(int num_entries_to_remove = 10);
    
private:
    // 内部辅助方法
    std::string extractKeyPoints(const std::string& content);
    std::string summarizeContent(const std::string& content, int max_length = 200);
    double calculateRelevance(const std::string& content, const std::string& category);
    std::vector<std::string> extractTags(const std::string& content);
    
    // 数据源特定解析器
    bool parseWikipediaResponse(const std::string& response, KnowledgeEntry& entry);
    bool parseArxivResponse(const std::string& response, KnowledgeEntry& entry);
    bool parseJSONData(const std::string& json_data, std::vector<KnowledgeEntry>& entries);
    bool parseCSVData(const std::string& csv_data, std::vector<KnowledgeEntry>& entries);
    
    // OpenAI API调用
    std::string callOpenAICompletion(const std::string& prompt, 
                                    const std::string& model = "gpt-3.5-turbo",
                                    double temperature = 0.7);
};

// 预定义知识库加载器
class PredefinedKnowledgeLoader {
public:
    // 加载预定义的知识库
    static std::vector<KnowledgeEntry> loadComputerScienceKnowledge();
    static std::vector<KnowledgeEntry> loadMathematicsKnowledge();
    static std::vector<KnowledgeEntry> loadPhysicsKnowledge();
    static std::vector<KnowledgeEntry> loadBiologyKnowledge();
    static std::vector<KnowledgeEntry> loadPhilosophyKnowledge();
    static std::vector<KnowledgeEntry> loadCommonSenseKnowledge();
    
    // 加载技术文档
    static std::vector<KnowledgeEntry> loadProgrammingDocumentation();
    static std::vector<KnowledgeEntry> loadMachineLearningKnowledge();
    static std::vector<KnowledgeEntry> loadNeuroscienceKnowledge();
    
    // 加载语言模型相关的知识
    static std::vector<KnowledgeEntry> loadLanguageModelKnowledge();
    static std::vector<KnowledgeEntry> loadAITechniques();
};

#endif // RAG_KNOWLEDGE_LOADER_H