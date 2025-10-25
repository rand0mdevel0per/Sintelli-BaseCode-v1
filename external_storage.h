#ifndef EXTERNAL_STORAGE_H
#define EXTERNAL_STORAGE_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "rag_knowledge_loader.h"

namespace ExternalStorage {

// HuggingFace数据集配置
struct HFDatasetConfig {
    std::string repo_id;           // 数据集仓库ID
    std::string subset;            // 子集名称（可选）
    std::string split;             // 分割（train/validation/test）
    bool streaming;                // 是否流式加载
    int batch_size;                // 批量大小
    
    HFDatasetConfig(const std::string& repo = "", 
                    const std::string& sub = "", 
                    const std::string& sp = "train", 
                    bool stream = true, 
                    int batch = 1000)
        : repo_id(repo), subset(sub), split(sp), streaming(stream), batch_size(batch) {}
};

// 数据集记录结构
struct DatasetRecord {
    std::string id;                // 记录ID
    std::string text;              // 文本内容
    std::string source;            // 数据源
    std::map<std::string, std::string> metadata; // 元数据
    std::map<std::string, double> scores;        // 评分
    
    DatasetRecord(const std::string& id_ = "", 
                  const std::string& text_ = "", 
                  const std::string& src = "")
        : id(id_), text(text_), source(src) {}
};

// 数据集解析回调函数类型
using RecordCallback = std::function<void(const DatasetRecord&)>;
using BatchCallback = std::function<void(const std::vector<DatasetRecord>&)>;

class ExternalStorage {
private:
    std::unique_ptr<RAGKnowledgeBaseLoader> rag_loader;
    std::string cache_dir;
    
public:
    ExternalStorage(const std::string& openai_key = "", 
                    const std::string& cache_path = "./cache");
    ~ExternalStorage() = default;
    
    // === 数据集流式解析方法 ===
    
    // MATH推理路径数据集解析
    bool parseMathReasoningDataset(const HFDatasetConfig& config, 
                                   RecordCallback callback,
                                   int max_samples = -1);
    
    // ArXiv论文数据集解析
    bool parseArXivDataset(const HFDatasetConfig& config,
                          RecordCallback callback,
                          int max_samples = -1);
    
    // GitHub代码数据集解析
    bool parseGitHubDataset(const HFDatasetConfig& config,
                           RecordCallback callback,
                           const std::string& category = "编程");
    
    // NIST网络安全数据集解析
    bool parseNISTDataset(const HFDatasetConfig& config,
                         RecordCallback callback,
                         int max_samples = -1);
    
    // FinePDFs数据集解析
    bool parseFinePDFsDataset(const HFDatasetConfig& config,
                             RecordCallback callback,
                             const std::string& language = "eng_Latn");
    
    // FineWeb数据集解析
    bool parseFineWebDataset(const HFDatasetConfig& config,
                            RecordCallback callback,
                            int max_samples = -1);
    
    // === 通用数据集方法 ===
    
    // 通用JSONL解析
    bool parseJSONLDataset(const std::string& file_path,
                          RecordCallback callback,
                          const std::string& category = "general");
    
    // 通用Parquet解析
    bool parseParquetDataset(const std::string& file_path,
                            RecordCallback callback,
                            const std::string& category = "general");
    
    // 流式加载到RAG知识库
    bool streamToRAG(const HFDatasetConfig& config,
                    const std::string& category = "general",
                    int max_entries = 1000);
    
    // 批量加载到RAG知识库
    bool batchToRAG(const std::vector<DatasetRecord>& records,
                   const std::string& category = "general");
    
    // === 工具方法 ===
    
    // 设置缓存目录
    void setCacheDir(const std::string& path);
    
    // 获取缓存目录
    std::string getCacheDir() const;
    
    // 清理缓存
    bool clearCache();
    
    // 获取数据集统计信息
    struct DatasetStats {
        size_t total_records;
        size_t categories;
        double avg_text_length;
        std::map<std::string, size_t> source_counts;
    };
    
    DatasetStats getDatasetStats() const;
    
private:
    // 内部方法
    bool downloadDataset(const HFDatasetConfig& config, 
                        const std::string& local_path);
    bool parseJSONLFile(const std::string& file_path, 
                       RecordCallback callback);
    bool parseParquetFile(const std::string& file_path, 
                         RecordCallback callback);
    
    // 数据集特定解析器
    bool parseMathReasoningRecord(const std::map<std::string, std::string>& data,
                                 RecordCallback callback);
    bool parseArXivRecord(const std::map<std::string, std::string>& data,
                         RecordCallback callback);
    bool parseGitHubRecord(const std::map<std::string, std::string>& data,
                          RecordCallback callback);
    bool parseNISTRecord(const std::map<std::string, std::string>& data,
                        RecordCallback callback);
    bool parseFinePDFsRecord(const std::map<std::string, std::string>& data,
                            RecordCallback callback);
    bool parseFineWebRecord(const std::map<std::string, std::string>& data,
                           RecordCallback callback);
    
    // 数据转换
    KnowledgeEntry recordToKnowledgeEntry(const DatasetRecord& record,
                                         const std::string& category);
    
    std::vector<DatasetRecord> loaded_records;
    std::map<std::string, std::vector<DatasetRecord>> category_records;
};

} // namespace ExternalStorage

#endif // EXTERNAL_STORAGE_H