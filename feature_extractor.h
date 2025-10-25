// 字符串特征提取器头文件
// 基于现有E5模型的语义特征提取接口

#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <string>
#include <vector>
#include "smry.cpp"
#include "isw.hpp"

class FeatureExtractor {
private:
    E5LargeModel* e5_model;
    UnifiedInputProcessor* text_processor;
    
public:
    FeatureExtractor(const std::string& model_path = "/models/e5/e5_large.onnx",
                    const std::string& vocab_path = "/models/vocab.json",
                    const std::string& merges_path = "/models/merges.txt",
                    const std::string& special_tokens_path = "/models/special_tokens.json");
    
    ~FeatureExtractor();
    
    // 提取文本的语义特征向量
    FeatureVector<float> extractTextFeature(const std::string& text);
    
    // 提取文本的语义特征向量(宽字符版本)
    FeatureVector<float> extractTextFeature(const std::wstring& text);
    
    // 批量提取文本特征
    std::vector<FeatureVector<float>> batchExtractTextFeatures(const std::vector<std::string>& texts);
    
    // 计算两个特征向量的相似度
    double calculateSimilarity(const FeatureVector<float>& vec1, const FeatureVector<float>& vec2, 
                              const std::string& metric = "cosine");
    
    // 语义搜索：在特征库中查找最相似的文本
    std::vector<std::pair<std::string, double>> semanticSearch(
        const std::string& query_text,
        const std::vector<std::string>& candidate_texts,
        int top_k = 10,
        double similarity_threshold = 0.0,
        const std::string& metric = "cosine");
    
    // 检查模型是否已初始化
    bool isInitialized() const;
    
    // 获取嵌入维度
    size_t getEmbeddingDimension() const;
    
private:
    // 初始化模型
    bool initializeModel(const std::string& model_path,
                        const std::string& vocab_path,
                        const std::string& merges_path,
                        const std::string& special_tokens_path);
};

// 便捷的全局函数
namespace FeatureUtils {
    // 创建默认特征提取器
    FeatureExtractor* createDefaultExtractor();
    
    // 快速特征提取
    FeatureVector<float> quickExtract(const std::string& text);
    
    // 快速相似度计算
    double quickSimilarity(const std::string& text1, const std::string& text2);
    
    // 寻找最相似的文本
    std::string findMostSimilar(const std::string& query, 
                               const std::vector<std::string>& candidates);
}

#endif // FEATURE_EXTRACTOR_H