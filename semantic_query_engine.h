#ifndef SEMANTIC_QUERY_ENGINE_H
#define SEMANTIC_QUERY_ENGINE_H

#include <string>
#include <vector>
#include <memory>

#include "isw.hpp"

// 语义查询引擎 - 独立的语义处理类，使用smry.cpp中的E5模型
class SemanticQueryEngine {
private:
    void *e5_model_; // E5模型实例指针
    bool is_initialized_;

public:
    SemanticQueryEngine(const std::string &model_path = "/models/e5/e5_large.onnx",
                        const std::string &vocab_path = "/models/e5/vocab.json",
                        const std::string &merges_path = "/models/e5/merges.txt",
                        const std::string &special_tokens_path = "/models/e5/special_tokens.json");

    ~SemanticQueryEngine();

    // 初始化E5模型
    bool initialize(const std::string &model_path,
                    const std::string &vocab_path,
                    const std::string &merges_path,
                    const std::string &special_tokens_path);

    // 获取文本的语义向量
    bool getTextEmbedding(const std::string &text, ::FeatureVector<float> &feature);

    // 获取语义向量的维度
    size_t getEmbeddingDimension() const;

    // 检查模型是否已初始化
    bool isInitialized() const;

    // 批量获取文本嵌入
    bool getBatchEmbeddings(const std::vector<std::string> &texts,
                            std::vector<::FeatureVector<float> > &features);

    // 计算两个文本之间的语义相似度
    double getSemanticSimilarity(const std::string &text1, const std::string &text2);

    // 语义搜索 - 在给定的文本集合中搜索与查询最相似的文本
    std::vector<std::pair<size_t, double> > semanticSearch(const std::string &query,
                                                           const std::vector<std::string> &candidates,
                                                           int top_k = 10,
                                                           double similarity_threshold = 0.0);
};

// 全局语义查询引擎实例（可选）
extern std::unique_ptr<SemanticQueryEngine> g_semantic_engine;

// 初始化全局语义查询引擎
bool initializeGlobalSemanticEngine(const std::string &model_path = "/models/e5/e5_large.onnx",
                                    const std::string &vocab_path = "/models/e5/vocab.json",
                                    const std::string &merges_path = "/models/e5/merges.txt",
                                    const std::string &special_tokens_path = "/models/e5/special_tokens.json");

// 获取全局语义查询引擎实例
SemanticQueryEngine *getGlobalSemanticEngine();

#endif // SEMANTIC_QUERY_ENGINE_H
