// 字符串特征提取器实现
// 基于E5模型的语义特征提取和匹配

#include "feature_extractor.h"
#include <algorithm>
#include <functional>
#include <numeric>

FeatureExtractor::FeatureExtractor(const std::string& model_path,
                                 const std::string& vocab_path,
                                 const std::string& merges_path,
                                 const std::string& special_tokens_path)
    : e5_model(nullptr), text_processor(nullptr) {
    initializeModel(model_path, vocab_path, merges_path, special_tokens_path);
}

FeatureExtractor::~FeatureExtractor() {
    if (text_processor) {
        delete text_processor;
        text_processor = nullptr;
    }
    if (e5_model) {
        delete e5_model;
        e5_model = nullptr;
    }
}

bool FeatureExtractor::initializeModel(const std::string& model_path,
                                     const std::string& vocab_path,
                                     const std::string& merges_path,
                                     const std::string& special_tokens_path) {
    try {
        // 初始化E5模型
        e5_model = new E5LargeModel(
            model_path.c_str(),
            vocab_path.c_str(), 
            merges_path.c_str(),
            special_tokens_path.c_str()
        );
        
        // 初始化文本处理器
        text_processor = createTextProcessor();
        
        return (e5_model != nullptr && text_processor != nullptr);
    } catch (...) {
        if (e5_model) {
            delete e5_model;
            e5_model = nullptr;
        }
        if (text_processor) {
            delete text_processor;
            text_processor = nullptr;
        }
        return false;
    }
}

FeatureVector<float> FeatureExtractor::extractTextFeature(const std::string& text) {
    FeatureVector<float> feature;
    
    if (!isInitialized() || text.empty()) {
        return feature;
    }
    
    try {
        // 转换为宽字符
        std::wstring wtext = stringToWstring(text);
        
        // 处理文本并获取全局语义向量
        if (text_processor->processText(wtext.c_str(), wtext.length())) {
            Matrix256* attention_matrix = text_processor->getGlobalAttentionMatrix();
            if (attention_matrix) {
                // 从注意力矩阵提取特征向量
                std::vector<float> embedding(EMBED_DIM, 0.0f);
                
                // 将256x256矩阵压缩为1024维向量
                int steps = MAT_ELEMENTS / EMBED_DIM;
                for (int i = 0; i < EMBED_DIM; ++i) {
                    float sum = 0.0f;
                    for (int j = 0; j < steps && (i * steps + j) < MAT_ELEMENTS; ++j) {
                        sum += static_cast<float>(attention_matrix->data[i * steps + j]);
                    }
                    embedding[i] = sum / steps;
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
                delete attention_matrix;
            }
        }
    } catch (...) {
        // 异常处理
    }
    
    return feature;
}

FeatureVector<float> FeatureExtractor::extractTextFeature(const std::wstring& text) {
    FeatureVector<float> feature;
    
    if (!isInitialized() || text.empty()) {
        return feature;
    }
    
    try {
        if (text_processor->processText(text.c_str(), text.length())) {
            Matrix256* attention_matrix = text_processor->getGlobalAttentionMatrix();
            if (attention_matrix) {
                std::vector<float> embedding(EMBED_DIM, 0.0f);
                
                int steps = MAT_ELEMENTS / EMBED_DIM;
                for (int i = 0; i < EMBED_DIM; ++i) {
                    float sum = 0.0f;
                    for (int j = 0; j < steps && (i * steps + j) < MAT_ELEMENTS; ++j) {
                        sum += static_cast<float>(attention_matrix->data[i * steps + j]);
                    }
                    embedding[i] = sum / steps;
                }
                
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
                delete attention_matrix;
            }
        }
    } catch (...) {
        // 异常处理
    }
    
    return feature;
}

std::vector<FeatureVector<float>> FeatureExtractor::batchExtractTextFeatures(
    const std::vector<std::string>& texts) {
    
    std::vector<FeatureVector<float>> features;
    features.reserve(texts.size());
    
    for (const auto& text : texts) {
        features.push_back(extractTextFeature(text));
    }
    
    return features;
}

double FeatureExtractor::calculateSimilarity(const FeatureVector<float>& vec1, 
                                           const FeatureVector<float>& vec2,
                                           const std::string& metric) {
    
    if (vec1.dimension != vec2.dimension || vec1.data.empty() || vec2.data.empty()) {
        return -1.0; // 无效相似度
    }
    
    if (metric == "cosine") {
        return vec1.cosineSimilarity(vec2);
    } else if (metric == "euclidean") {
        double distance = vec1.euclideanDistance(vec2);
        return 1.0 / (1.0 + distance); // 转换为相似度
    } else if (metric == "manhattan") {
        double distance = vec1.manhattanDistance(vec2);
        return 1.0 / (1.0 + distance); // 转换为相似度
    }
    
    return -1.0; // 无效度量
}

std::vector<std::pair<std::string, double>> FeatureExtractor::semanticSearch(
    const std::string& query_text,
    const std::vector<std::string>& candidate_texts,
    int top_k,
    double similarity_threshold,
    const std::string& metric) {
    
    std::vector<std::pair<std::string, double>> results;
    
    if (!isInitialized() || query_text.empty() || candidate_texts.empty()) {
        return results;
    }
    
    // 提取查询文本的特征
    FeatureVector<float> query_feature = extractTextFeature(query_text);
    if (query_feature.data.empty()) {
        return results;
    }
    
    // 提取候选文本的特征
    std::vector<FeatureVector<float>> candidate_features = batchExtractTextFeatures(candidate_texts);
    
    // 计算相似度
    for (size_t i = 0; i < candidate_texts.size(); ++i) {
        if (candidate_features[i].data.empty()) continue;
        
        double similarity = calculateSimilarity(query_feature, candidate_features[i], metric);
        if (similarity >= similarity_threshold) {
            results.emplace_back(candidate_texts[i], similarity);
        }
    }
    
    // 按相似度排序
    std::sort(results.begin(), results.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // 限制结果数量
    if (results.size() > static_cast<size_t>(top_k)) {
        results.resize(top_k);
    }
    
    return results;
}

bool FeatureExtractor::isInitialized() const {
    return (e5_model != nullptr && text_processor != nullptr);
}

size_t FeatureExtractor::getEmbeddingDimension() const {
    return EMBED_DIM;
}

// 全局便捷函数实现
namespace FeatureUtils {
    
    static FeatureExtractor* g_default_extractor = nullptr;
    
    FeatureExtractor* createDefaultExtractor() {
        if (!g_default_extractor) {
            g_default_extractor = new FeatureExtractor();
        }
        return g_default_extractor;
    }
    
    FeatureVector<float> quickExtract(const std::string& text) {
        FeatureExtractor* extractor = createDefaultExtractor();
        return extractor->extractTextFeature(text);
    }
    
    double quickSimilarity(const std::string& text1, const std::string& text2) {
        FeatureExtractor* extractor = createDefaultExtractor();
        FeatureVector<float> vec1 = extractor->extractTextFeature(text1);
        FeatureVector<float> vec2 = extractor->extractTextFeature(text2);
        
        if (vec1.data.empty() || vec2.data.empty()) {
            return -1.0;
        }
        
        return vec1.cosineSimilarity(vec2);
    }
    
    std::string findMostSimilar(const std::string& query, 
                               const std::vector<std::string>& candidates) {
        FeatureExtractor* extractor = createDefaultExtractor();
        auto results = extractor->semanticSearch(query, candidates, 1);
        
        if (!results.empty()) {
            return results[0].first;
        }
        
        return "";
    }
}