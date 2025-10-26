// 语义匹配器 - 基于特征向量的语义搜索和匹配
// 集成到NeuronModel中的语义匹配功能

#ifndef SEMANTIC_MATCHER_H
#define SEMANTIC_MATCHER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include "feature_extractor.h"
#include "isw.hpp"
#include "structs.h"

// 语义匹配结果
struct SemanticMatch {
    std::string text; // 匹配的文本内容
    double similarity; // 相似度得分 (0-1)
    uint64_t slot_id; // 在外部存储中的slot_id
    std::string category; // 类别标签

    SemanticMatch(const std::string &t = "", double s = 0.0,
                  uint64_t id = 0, const std::string &cat = "")
        : text(t), similarity(s), slot_id(id), category(cat) {
    }

    bool operator<(const SemanticMatch &other) const {
        return similarity > other.similarity; // 降序排列
    }

    std::string hash() const {
        return sha256_hash<SemanticMatch>(*this);
    }

    std::string getText() const {
        return text;
    }
};

class SemanticMatcher {
private:
    std::unique_ptr<FeatureExtractor> extractor;
    std::unordered_map<uint64_t, std::string> id_to_text; // slot_id到文本的映射
    std::unordered_map<uint64_t, std::string> id_to_category; // slot_id到类别的映射

public:
    SemanticMatcher(const std::string &model_path = "/models/e5/e5_large.onnx",
                    const std::string &vocab_path = "/models/vocab.json",
                    const std::string &merges_path = "/models/merges.txt",
                    const std::string &special_tokens_path = "/models/special_tokens.json");

    // 注册文本到语义匹配系统
    uint64_t registerText(const std::string &text, const std::string &category = "");

    // 批量注册文本
    std::vector<uint64_t> batchRegisterTexts(const std::vector<std::string> &texts,
                                             const std::string &category = "");

    // 语义搜索：查找最相似的文本
    std::vector<SemanticMatch> findSimilarTexts(const std::string &query_text,
                                                int top_k = 10,
                                                double similarity_threshold = 0.5,
                                                const std::string &metric = "cosine");

    // 基于类别过滤的语义搜索
    std::vector<SemanticMatch> findSimilarTextsByCategory(const std::string &query_text,
                                                          const std::string &category,
                                                          int top_k = 10,
                                                          double similarity_threshold = 0.5);

    // 获取所有已注册的文本
    std::vector<std::string> getAllRegisteredTexts() const;

    // 根据slot_id获取文本
    std::string getTextById(uint64_t slot_id) const;

    // 根据文本获取slot_id
    uint64_t getIdByText(const std::string &text) const;

    // 移除文本
    bool removeText(uint64_t slot_id);

    // 清空所有文本
    void clearAll();

    // 获取统计信息
    struct Stats {
        size_t total_texts;
        std::unordered_map<std::string, size_t> category_counts;
        size_t unique_categories;
    };

    Stats getStats() const;

    // 获取特征提取器实例
    FeatureExtractor *getFeatureExtractor() const;

private:
    // 生成唯一的slot_id
    uint64_t generateSlotId() const;

    // 内部存储管理
    mutable uint64_t next_slot_id = 1;
};

// 与ExternalStorage集成的语义匹配器
class IntegratedSemanticMatcher {
private:
    std::unique_ptr<SemanticMatcher> matcher;
    ExternalStorage<SemanticMatch> *storage; // 使用ExternalStorage存储匹配结果

public:
    IntegratedSemanticMatcher(ExternalStorage<SemanticMatch> *ext_storage = nullptr,
                              const std::string &model_path = "/models/e5/e5_large.onnx");

    // 注册文本并存储到外部存储
    uint64_t registerAndStore(const std::string &text, const std::string &category = "");

    // 语义搜索并存储结果
    std::vector<SemanticMatch> searchAndStore(const std::string &query_text,
                                              int top_k = 10,
                                              double similarity_threshold = 0.5,
                                              const std::string &metric = "cosine");

    // 从外部存储中加载语义匹配结果
    bool loadFromStorage(uint64_t slot_id, SemanticMatch &match);

    // 获取最热的语义匹配结果
    std::vector<SemanticMatch> getHottestMatches(int k = 10);

    // 设置外部存储
    void setStorage(ExternalStorage<SemanticMatch> *ext_storage);

    // 获取语义匹配器
    SemanticMatcher *getMatcher() const;
};

// ===== Logic语义匹配接口 =====

// Logic描述符结构 - 用于语义匹配和注入
struct LogicDescriptor {
    std::string logic_id; // Logic的唯一标识
    std::string description; // Logic的描述文本
    std::string category; // Logic的类别
    FeatureVector<float> feature; // Logic的语义特征
    double activation_threshold; // 激活阈值

    // 回调函数：根据Logic ID生成对应的NeuronInput
    std::function<void(const std::string &, NeuronInput &)> generate_input_callback;

    LogicDescriptor(const std::string &id = "",
                    const std::string &desc = "",
                    const std::string &cat = "",
                    double threshold = 0.5)
        : logic_id(id), description(desc), category(cat),
          activation_threshold(threshold) {
    }

    // 创建默认的NeuronInput生成器
    static std::function<void(const std::string &, NeuronInput &)> createDefaultGenerator(
        const std::string &logic_content,
        double activity = 1.0,
        double weight = 1.0,
        ll from_x = 0, ll from_y = 0, ll from_z = 0) {
        return [logic_content, activity, weight, from_x, from_y, from_z]
        (const std::string &logic_id, NeuronInput &input) {
            // 初始化NeuronInput
            memset(&input, 0, sizeof(NeuronInput));

            // 设置基本参数
            input.activity = activity;
            input.weight = weight;
            input.from_coord[0] = from_x;
            input.from_coord[1] = from_y;
            input.from_coord[2] = from_z;

            // 将logic_content转换为矩阵数据
            // 这里可以根据logic_id和content生成特定的输入数据
            // 例如：基于Logic的语义特征生成特定的激活模式

            // 简单的示例：将logic_id的hash值映射到矩阵中
            std::hash<std::string> hasher;
            size_t hash_val = hasher(logic_id);

            for (int i = 0; i < 256; ++i) {
                for (int j = 0; j < 256; ++j) {
                    // 基于hash值生成伪随机模式
                    input.array[i][j] = static_cast<double>((hash_val + i * 256 + j) % 100) / 100.0;
                }
            }
        };
    }

    // Hash函数，用于ExternalStorage
    std::string hash() const {
        std::hash<std::string> hasher;
        return std::to_string(hasher(logic_id + description + category));
    }

    // 获取文本内容，用于ExternalStorage的语义查询
    std::string getText() const {
        return description;
    }

    // 获取特征向量，用于ExternalStorage的特征匹配
    FeatureVector<float> getFeature() const {
        return feature;
    }
};

// Logic语义匹配器 - 专门用于Logic的召回和注入
class LogicSemanticMatcher {
private:
    std::unique_ptr<FeatureExtractor> extractor;
    std::unordered_map<std::string, LogicDescriptor> logic_descriptors;
    std::unordered_map<std::string, std::vector<std::string> > category_to_logics;

public:
    LogicSemanticMatcher(const std::string &model_path = "/models/e5/e5_large.onnx",
                         const std::string &vocab_path = "/models/vocab.json",
                         const std::string &merges_path = "/models/merges.txt",
                         const std::string &special_tokens_path = "/models/special_tokens.json");

    // 注册Logic描述符
    bool registerLogic(const LogicDescriptor &logic_desc);

    // 批量注册Logic
    bool batchRegisterLogics(const std::vector<LogicDescriptor> &logics);

    // 根据文本查询匹配的Logic
    std::vector<std::pair<LogicDescriptor, double> > findMatchingLogics(
        const std::string &query_text,
        int top_k = 5,
        double similarity_threshold = 0.3,
        const std::string &category = "");

    // 根据特征向量查询匹配的Logic
    std::vector<std::pair<LogicDescriptor, double> > findMatchingLogicsByFeature(
        const FeatureVector<float> &query_feature,
        int top_k = 5,
        double similarity_threshold = 0.3,
        const std::string &category = "");

    // 激活匹配的Logic（执行回调函数）
    std::vector<std::pair<LogicDescriptor, NeuronInput> > activateMatchingLogics(const std::string &query_text,
        int top_k = 3,
        double similarity_threshold = 0.5);

    // 获取Logic描述符
    LogicDescriptor *getLogicDescriptor(const std::string &logic_id);

    // 移除Logic
    bool removeLogic(const std::string &logic_id);

    // 清空所有Logic
    void clearAllLogics();

    // 获取统计信息
    struct LogicStats {
        size_t total_logics;
        std::unordered_map<std::string, size_t> category_counts;
        size_t unique_categories;
    };

    LogicStats getLogicStats() const;

    // 获取特征提取器
    FeatureExtractor *getFeatureExtractor() const;

    // 根据Logic ID获取描述文本
    std::string getLogicDescription(const std::string &logic_id) const;

    // 设置Logic激活回调
    bool setLogicCallback(const std::string &logic_id,
                          std::function<void(const std::string &, NeuronInput &)> callback);

    // 设置简单Logic激活回调（不需要参数的版本）
    bool setSimpleLogicCallback(const std::string &logic_id, std::function<void()> callback);
};

// Logic注入器接口
class LogicInjector {
private:
    LogicSemanticMatcher logic_matcher;
    ExternalStorage<LogicDescriptor> *logic_storage;

public:
    LogicInjector(ExternalStorage<LogicDescriptor> *storage = nullptr,
                  const std::string &model_path = "/models/e5/e5_large.onnx");

    // 注册Logic到存储系统
    bool registerLogicWithStorage(const LogicDescriptor &logic_desc);

    // 仅返回匹配的logic_id，不执行注入
    std::vector<std::pair<std::string, double> > findMatchingLogicIds(
        const std::string &query_text,
        int top_k = 3,
        double similarity_threshold = 0.4);

    // 根据查询文本注入匹配的Logic
    std::vector<std::pair<std::string, NeuronInput> > injectMatchingLogics(const std::string &query_text,
                                                                           int top_k = 3,
                                                                           double similarity_threshold = 0.4);

    // 从存储中加载Logic
    bool loadLogicFromStorage(const std::string &logic_id, LogicDescriptor &logic_desc);

    // 根据logic_id从存储中获取LogicDescriptor（不执行匹配）
    LogicDescriptor *getLogicById(const std::string &logic_id);

    // 根据logic_id执行注入（单独注入指定Logic）
    bool injectLogicById(const std::string &logic_id, NeuronInput &input);

    // 获取最活跃的Logic
    std::vector<LogicDescriptor> getMostActiveLogics(int k = 10);

    // 设置存储
    void setStorage(ExternalStorage<LogicDescriptor> *storage);

    // 获取Logic匹配器
    LogicSemanticMatcher *getLogicMatcher();
};

#endif // SEMANTIC_MATCHER_H
