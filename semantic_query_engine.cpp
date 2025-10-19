#include "semantic_query_engine.h"
#include "smry.cpp"  // 包含E5模型实现
#include <algorithm>
#include <limits>
#include <cmath>

// 全局语义查询引擎实例
std::unique_ptr<SemanticQueryEngine> g_semantic_engine;

SemanticQueryEngine::SemanticQueryEngine(const std::string& model_path,
                                       const std::string& vocab_path,
                                       const std::string& merges_path,
                                       const std::string& special_tokens_path)
    : e5_model_(nullptr), is_initialized_(false) {
    if (!model_path.empty()) {
        initialize(model_path, vocab_path, merges_path, special_tokens_path);
    }
}

SemanticQueryEngine::~SemanticQueryEngine() {
    // 清理E5模型资源
    if (e5_model_) {
        // 注意：实际清理需要根据E5模型的具体实现来处理
        e5_model_ = nullptr;
    }
}

bool SemanticQueryEngine::initialize(const std::string& model_path,
                                   const std::string& vocab_path,
                                   const std::string& merges_path,
                                   const std::string& special_tokens_path) {
    // 初始化E5模型
    is_initialized_ = initUnifiedSystem(model_path.c_str(), 
                                       vocab_path.c_str(), 
                                       merges_path.c_str(), 
                                       special_tokens_path.c_str());
    return is_initialized_;
}

bool SemanticQueryEngine::getTextEmbedding(const std::string& text, FeatureVector<float>& feature) {
    if (!is_initialized_) {
        return false;
    }
    
    // 这里应该调用E5模型获取文本嵌入
    // 实际实现会调用smry.cpp中的E5模型接口
    
    // 临时实现：生成一个伪特征向量用于演示
    std::vector<float> embedding(1024, 0.0f);
    
    // 简单的文本hash作为特征
    std::hash<std::string> hasher;
    size_t hash_val = hasher(text);
    
    // 基于hash生成伪特征
    for (size_t i = 0; i < embedding.size(); ++i) {
        embedding[i] = static_cast<float>((hash_val + i) % 100) / 100.0f;
    }
    
    // 归一化
    float norm = 0.0f;
    for (float val : embedding) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
        for (float& val : embedding) {
            val /= norm;
        }
    }
    
    feature = FeatureVector<float>(embedding, "semantic");
    return true;
}

size_t SemanticQueryEngine::getEmbeddingDimension() const {
    return 1024; // E5模型通常是1024维
}

bool SemanticQueryEngine::isInitialized() const {
    return is_initialized_;
}

bool SemanticQueryEngine::getBatchEmbeddings(const std::vector<std::string>& texts, 
                                           std::vector<FeatureVector<float>>& features) {
    if (!is_initialized_) {
        return false;
    }
    
    features.clear();
    features.reserve(texts.size());
    
    for (const auto& text : texts) {
        FeatureVector<float> feature;
        if (getTextEmbedding(text, feature)) {
            features.push_back(std::move(feature));
        } else {
            return false; // 如果任何一个文本处理失败，则返回失败
        }
    }
    
    return true;
}

double SemanticQueryEngine::getSemanticSimilarity(const std::string& text1, const std::string& text2) {
    if (!is_initialized_) {
        return -1.0; // 表示错误
    }
    
    FeatureVector<float> feature1, feature2;
    if (!getTextEmbedding(text1, feature1) || !getTextEmbedding(text2, feature2)) {
        return -1.0; // 表示错误
    }
    
    // 计算余弦相似度
    return feature1.cosineSimilarity(feature2);
}

std::vector<std::pair<size_t, double>> SemanticQueryEngine::semanticSearch(const std::string& query,
                                                                         const std::vector<std::string>& candidates,
                                                                         int top_k,
                                                                         double similarity_threshold) {
    std::vector<std::pair<size_t, double>> results;
    
    if (!is_initialized_ || candidates.empty()) {
        return results;
    }
    
    FeatureVector<float> query_feature;
    if (!getTextEmbedding(query, query_feature)) {
        return results;
    }
    
    for (size_t i = 0; i < candidates.size(); ++i) {
        FeatureVector<float> candidate_feature;
        if (getTextEmbedding(candidates[i], candidate_feature)) {
            double similarity = query_feature.cosineSimilarity(candidate_feature);
            if (similarity >= similarity_threshold) {
                results.emplace_back(i, similarity);
            }
        }
    }
    
    // 按相似度降序排序
    std::sort(results.begin(), results.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // 保留前top_k个结果
    if (results.size() > static_cast<size_t>(top_k)) {
        results.resize(top_k);
    }
    
    return results;
}

// 初始化全局语义查询引擎
bool initializeGlobalSemanticEngine(const std::string& model_path,
                                  const std::string& vocab_path,
                                  const std::string& merges_path,
                                  const std::string& special_tokens_path) {
    if (!g_semantic_engine) {
        g_semantic_engine = std::make_unique<SemanticQueryEngine>();
    }
    
    return g_semantic_engine->initialize(model_path, vocab_path, merges_path, special_tokens_path);
}

// 获取全局语义查询引擎实例
SemanticQueryEngine* getGlobalSemanticEngine() {
    return g_semantic_engine.get();
}