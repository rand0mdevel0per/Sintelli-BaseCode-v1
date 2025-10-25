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
#include "structs.h"  // 包含NeuronInput定义

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
    
    // === Data Source Loading Methods ===
    // Methods for loading knowledge from various data sources including URLs, files, and APIs
    
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
    
    // === Hugging Face Dataset Streaming Methods ===
    // Methods for streaming and processing Hugging Face datasets
    // Supports large dataset processing without memory overflow
    
    /**
     * @brief Stream and parse Hugging Face dataset.
     * 
     * Processes large Hugging Face datasets in streaming mode to avoid memory overflow.
     * Supports various dataset subsets and splits with configurable entry limits.
     * 
     * @param dataset_name Name of the Hugging Face dataset
     * @param subset Dataset subset (default: "default")
     * @param split Data split (default: "train")
     * @param max_entries Maximum number of entries to process
     * @param category Category for the processed entries
     * @return true if successful, false otherwise
     */
    bool streamHuggingFaceDataset(const std::string& dataset_name, 
                                 const std::string& subset = "default",
                                 const std::string& split = "train",
                                 int max_entries = 100,
                                 const std::string& category = "huggingface");
    
    /**
     * @brief Query and load relevant entries from Hugging Face dataset.
     * 
     * Searches Hugging Face dataset for entries matching the query text.
     * Useful for targeted knowledge retrieval and RAG enhancement.
     * 
     * @param query Search query text
     * @param dataset_name Name of the Hugging Face dataset
     * @param subset Dataset subset
     * @param max_results Maximum number of results to return
     * @param category Category for the results
     * @return true if successful, false otherwise
     */
    bool queryAndLoadFromHFDataset(const std::string& query,
                                  const std::string& dataset_name = "HuggingFaceFW/fineweb",
                                  const std::string& subset = "sample-10BT",
                                  int max_results = 50,
                                  const std::string& category = "huggingface_query");
    
    /**
     * @brief Automatically fetch data when Logic matches are insufficient.
     * 
     * Dynamically retrieves additional data from external sources when
     * existing Logic matches fall below the minimum required threshold.
     * 
     * @param query Search query to find relevant data
     * @param min_required_matches Minimum required Logic matches
     * @param dataset_name Source dataset name
     * @param subset Dataset subset
     * @return true if data was successfully fetched and integrated
     */
    bool autoFetchDataWhenLogicInsufficient(const std::string& query,
                                          int min_required_matches = 5,
                                          const std::string& dataset_name = "HuggingFaceFW/fineweb",
                                          const std::string& subset = "sample-10BT");
    
    // === Logic Generation Methods ===
    // Methods for generating Logic trees from knowledge entries
    // Integrates with AI models for enhanced Logic creation
    
    /**
     * @brief Automatically generate Logic tree from knowledge text.
     * 
     * Processes knowledge text to create structured Logic descriptors
     * with semantic embeddings for matching and retrieval.
     * 
     * @param knowledge_text Input knowledge text
     * @param category Category for the generated Logic
     * @param activation_threshold Activation threshold for the Logic
     * @return Vector of generated Logic descriptors
     */
    std::vector<LogicDescriptor> generateLogicTreeFromKnowledge(
        const std::string& knowledge_text, 
        const std::string& category = "general",
        double activation_threshold = 0.5);
    
    /**
     * @brief Batch generate Logic trees from a category.
     * 
     * Creates multiple Logic descriptors from all knowledge entries
     * belonging to the specified category.
     * 
     * @param category Category of knowledge entries to process
     * @param max_logics Maximum number of Logic descriptors to generate
     * @param activation_threshold Activation threshold for generated Logic
     * @return Vector of generated Logic descriptors
     */
    std::vector<LogicDescriptor> generateLogicTreeFromCategory(
        const std::string& category,
        int max_logics = 50,
        double activation_threshold = 0.5);
    
    // 使用AI辅助生成Logic描述
    std::string generateLogicDescriptionWithAI(const std::string& knowledge_content);
    
    // 注册Logic到语义匹配器
    bool registerLogicTree(LogicSemanticMatcher& matcher, 
                          const std::vector<LogicDescriptor>& logics);
    
    // === Knowledge Base Management ===
    // Methods for managing and querying the knowledge base
    // Includes search, statistics, and export functionality
    
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
    
    // === Configuration Methods ===
    // Methods for configuring the RAG knowledge base loader
    // Allows customization of loading parameters and behavior
    
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
    bool insertToExternalStorageWithSemanticFeatures(const KnowledgeEntry& entry);
    bool checkAndCleanupStorage();
    void cleanupL3Cache(int num_entries_to_remove = 10);
    
    // Logic系统集成方法
    bool registerKnowledgeAsLogic(LogicInjector* logic_injector, 
                                 ExternalStorage<Logic>* logic_tree,
                                 const std::string& category = "rag_knowledge");
    bool autoFetchAndRegisterLogic(LogicInjector* logic_injector,
                                  ExternalStorage<Logic>* logic_tree,
                                  const std::string& query,
                                  int min_logics = 10,
                                  const std::string& dataset_name = "HuggingFaceFW/fineweb",
                                  const std::string& subset = "sample-10BT",
                                  const std::string& category = "rag_knowledge");
    
    // 语义搜索方法
    std::vector<KnowledgeEntry> semanticSearchKnowledge(const std::string& query, 
                                                       int top_k = 10,
                                                       double similarity_threshold = 0.3);
    
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