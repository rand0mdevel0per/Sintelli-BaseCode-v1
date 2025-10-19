// Logic语义匹配器实现
// 专门用于Logic的召回和注入

#include "semantic_matcher.h"
#include <algorithm>
#include <functional>

// LogicSemanticMatcher 实现
LogicSemanticMatcher::LogicSemanticMatcher(const std::string& model_path,
                                         const std::string& vocab_path,
                                         const std::string& merges_path,
                                         const std::string& special_tokens_path) {
    extractor = std::make_unique<FeatureExtractor>(model_path, vocab_path, 
                                                  merges_path, special_tokens_path);
}

bool LogicSemanticMatcher::registerLogic(const LogicDescriptor& logic_desc) {
    if (logic_desc.logic_id.empty() || logic_desc.description.empty()) {
        return false;
    }
    
    // 提取Logic描述的特征
    FeatureVector<float> feature = extractor->extractTextFeature(logic_desc.description);
    if (feature.data.empty()) {
        return false;
    }
    
    // 创建新的Logic描述符副本并设置特征
    LogicDescriptor new_logic = logic_desc;
    new_logic.feature = feature;
    
    // 存储到映射中
    logic_descriptors[logic_desc.logic_id] = new_logic;
    
    // 更新类别索引
    if (!logic_desc.category.empty()) {
        category_to_logics[logic_desc.category].push_back(logic_desc.logic_id);
    }
    
    return true;
}

bool LogicSemanticMatcher::batchRegisterLogics(const std::vector<LogicDescriptor>& logics) {
    bool all_success = true;
    for (const auto& logic : logics) {
        if (!registerLogic(logic)) {
            all_success = false;
        }
    }
    return all_success;
}

std::vector<std::pair<LogicDescriptor, double>> LogicSemanticMatcher::findMatchingLogics(
    const std::string& query_text,
    int top_k,
    double similarity_threshold,
    const std::string& category) {
    
    std::vector<std::pair<LogicDescriptor, double>> results;
    
    if (!extractor || !extractor->isInitialized() || query_text.empty()) {
        return results;
    }
    
    // 提取查询文本的特征
    FeatureVector<float> query_feature = extractor->extractTextFeature(query_text);
    if (query_feature.data.empty()) {
        return results;
    }
    
    return findMatchingLogicsByFeature(query_feature, top_k, similarity_threshold, category);
}

std::vector<std::pair<LogicDescriptor, double>> LogicSemanticMatcher::findMatchingLogicsByFeature(
    const FeatureVector<float>& query_feature,
    int top_k,
    double similarity_threshold,
    const std::string& category) {
    
    std::vector<std::pair<LogicDescriptor, double>> results;
    
    if (logic_descriptors.empty() || query_feature.data.empty()) {
        return results;
    }
    
    // 获取候选Logic列表
    std::vector<std::string> candidate_ids;
    if (category.empty()) {
        // 所有Logic
        for (const auto& pair : logic_descriptors) {
            candidate_ids.push_back(pair.first);
        }
    } else {
        // 指定类别的Logic
        auto it = category_to_logics.find(category);
        if (it != category_to_logics.end()) {
            candidate_ids = it->second;
        }
    }
    
    if (candidate_ids.empty()) {
        return results;
    }
    
    // 计算相似度
    for (const auto& logic_id : candidate_ids) {
        const auto& logic_desc = logic_descriptors[logic_id];
        if (logic_desc.feature.data.empty()) continue;
        
        double similarity = extractor->calculateSimilarity(query_feature, logic_desc.feature, "cosine");
        
        if (similarity >= similarity_threshold) {
            results.emplace_back(logic_desc, similarity);
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

std::vector<std::pair<LogicDescriptor, NeuronInput>> LogicSemanticMatcher::activateMatchingLogics(
    const std::string &query_text,
    int top_k,
    double similarity_threshold) {
    
    auto matching_logics = findMatchingLogics(query_text, top_k, similarity_threshold);
    
    std::vector<std::pair<LogicDescriptor, NeuronInput>> activated_logics;
    
    for (const auto& [logic_desc, similarity] : matching_logics) {
        if (logic_desc.generate_input_callback && similarity >= logic_desc.activation_threshold) {
            NeuronInput input{};
            logic_desc.generate_input_callback(logic_desc.logic_id, input);
            activated_logics.emplace_back(logic_desc, input);
        }
    }
    
    return activated_logics;
}


LogicDescriptor* LogicSemanticMatcher::getLogicDescriptor(const std::string& logic_id) {
    auto it = logic_descriptors.find(logic_id);
    return (it != logic_descriptors.end()) ? &it->second : nullptr;
}

bool LogicSemanticMatcher::removeLogic(const std::string& logic_id) {
    auto it = logic_descriptors.find(logic_id);
    if (it == logic_descriptors.end()) {
        return false;
    }
    
    // 从类别索引中移除
    const std::string& category = it->second.category;
    if (!category.empty()) {
        auto cat_it = category_to_logics.find(category);
        if (cat_it != category_to_logics.end()) {
            auto& logic_ids = cat_it->second;
            logic_ids.erase(std::remove(logic_ids.begin(), logic_ids.end(), logic_id), logic_ids.end());
            if (logic_ids.empty()) {
                category_to_logics.erase(cat_it);
            }
        }
    }
    
    // 从主映射中移除
    logic_descriptors.erase(it);
    return true;
}

void LogicSemanticMatcher::clearAllLogics() {
    logic_descriptors.clear();
    category_to_logics.clear();
}

LogicSemanticMatcher::LogicStats LogicSemanticMatcher::getLogicStats() const {
    LogicStats stats;
    stats.total_logics = logic_descriptors.size();
    
    for (const auto& pair : logic_descriptors) {
        if (!pair.second.category.empty()) {
            stats.category_counts[pair.second.category]++;
        }
    }
    
    stats.unique_categories = stats.category_counts.size();
    return stats;
}

FeatureExtractor* LogicSemanticMatcher::getFeatureExtractor() const {
    return extractor.get();
}

std::string LogicSemanticMatcher::getLogicDescription(const std::string& logic_id) const {
    auto it = logic_descriptors.find(logic_id);
    return (it != logic_descriptors.end()) ? it->second.description : "";
}

// 设置Logic回调函数
bool LogicSemanticMatcher::setLogicCallback(const std::string& logic_id, std::function<void(const std::string&, NeuronInput&)> callback) {
    auto it = logic_descriptors.find(logic_id);
    if (it == logic_descriptors.end()) {
        return false;
    }
    
    it->second.generate_input_callback = callback;
    return true;
}

// 设置简单Logic激活回调（不需要参数的版本）
bool LogicSemanticMatcher::setSimpleLogicCallback(const std::string& logic_id, std::function<void()> callback) {
    auto it = logic_descriptors.find(logic_id);
    if (it == logic_descriptors.end()) {
        return false;
    }
    
    // 创建一个包装函数，忽略参数并执行原始回调
    it->second.generate_input_callback = [callback](const std::string&, NeuronInput&) {
        callback();
    };
    
    return true;
}

// LogicInjector 实现
LogicInjector::LogicInjector(ExternalStorage<LogicDescriptor>* storage,
                           const std::string& model_path)
    : logic_storage(storage), logic_matcher(model_path) {
}

bool LogicInjector::registerLogicWithStorage(const LogicDescriptor& logic_desc) {
    if (!logic_matcher.registerLogic(logic_desc)) {
        return false;
    }
    
    if (logic_storage) {
        // 存储Logic描述符
        logic_storage->storeWithFeature(logic_desc, logic_desc.feature, 1.0);
    }
    
    return true;
}

// 仅返回匹配的logic_id，不执行注入
std::vector<std::pair<std::string, double>> LogicInjector::findMatchingLogicIds(
    const std::string& query_text,
    int top_k,
    double similarity_threshold) {
    
    std::vector<std::pair<std::string, double>> matched_ids;
    
    // 使用findMatchingLogics获取匹配结果
    auto matching_results = logic_matcher.findMatchingLogics(query_text, top_k, similarity_threshold);
    
    // 仅返回logic_id和相似度，不执行注入
    for (const auto& [logic_desc, similarity] : matching_results) {
        // 检查激活阈值
        if (similarity >= logic_desc.activation_threshold) {
            matched_ids.emplace_back(logic_desc.logic_id, similarity);
            
            // 更新存储中的热度（即使不注入也要更新热度）
            if (logic_storage) {
                double heat_increase = similarity * 5.0; // 比注入时的热度增加少一些
                std::string hash_val = logic_desc.hash();
                LogicDescriptor temp_desc;
                if (!logic_storage->fetchByHash(hash_val, temp_desc)) {
                    logic_storage->store(logic_desc, logic_desc.feature, 1.0);
                } else {
                    logic_storage->store(logic_desc, logic_desc.feature, heat_increase);
                }
            }
        }
    }
    
    return matched_ids;
}

std::vector<std::pair<std::string, NeuronInput>> LogicInjector::injectMatchingLogics(
    const std::string& query_text,
    int top_k,
    double similarity_threshold) {
    
    std::vector<std::pair<std::string, NeuronInput>> activated_logics;
    
    // 先使用findMatchingLogics获取包含相似度的匹配结果
    auto matching_results = logic_matcher.findMatchingLogics(query_text, top_k, similarity_threshold);
    
    // 激活匹配的Logic
    for (const auto& [logic_desc, similarity] : matching_results) {
        // 检查激活阈值
        if (logic_desc.generate_input_callback && similarity >= logic_desc.activation_threshold) {
            NeuronInput neuron_input{};
            logic_desc.generate_input_callback(logic_desc.logic_id, neuron_input);
            activated_logics.emplace_back(logic_desc.logic_id, neuron_input);
            
            // 更新存储中的热度
            if (logic_storage) {
                // 根据相似度更新热度：相似度越高，热度增加越多
                double heat_increase = similarity * 10.0; // 相似度乘以权重
                
                // 获取LogicDescriptor的hash值
                std::string hash_val = logic_desc.hash();
                
                // 查找对应的slot_id
                // 注意：这里假设LogicDescriptor已经存储在ExternalStorage中
                // 如果没有存储，则重新存储
                LogicDescriptor temp_desc;
                if (!logic_storage->fetchByHash(hash_val, temp_desc)) {
                    // 如果不存在，则存储
                    logic_storage->store(logic_desc, logic_desc.feature, 1.0);
                } else {
                    // 如果存在，更新热度
                    // 注意：ExternalStorage的adjustHeat需要slot_id，但这里只有hash
                    // 需要先通过hash找到slot_id，然后调整热度
                    // 这里简化处理：重新存储以更新热度
                    logic_storage->store(logic_desc, logic_desc.feature, heat_increase);
                }
            }
        }
    }
    
    return activated_logics;
}

bool LogicInjector::loadLogicFromStorage(const std::string& logic_id, LogicDescriptor& logic_desc) {
    if (!logic_storage) {
        return false;
    }
    
    // 使用LogicDescriptor的hash方法查找
    LogicDescriptor temp_desc;
    temp_desc.logic_id = logic_id;
    std::string hash_val = temp_desc.hash();
    
    // 通过hash值从ExternalStorage加载
    if (logic_storage->fetchByHash(hash_val, logic_desc)) {
        // 成功加载，注册到matcher中
        return logic_matcher.registerLogic(logic_desc);
    }
    
    return false;
}

// 根据logic_id从存储中获取LogicDescriptor
LogicDescriptor* LogicInjector::getLogicById(const std::string& logic_id) {
    // 先从内存中的匹配器查找
    auto* logic_desc = logic_matcher.getLogicDescriptor(logic_id);
    if (logic_desc) {
        return logic_desc;
    }
    
    // 如果内存中没有，尝试从ISW存储中加载
    if (logic_storage) {
        LogicDescriptor temp_desc;
        temp_desc.logic_id = logic_id;
        std::string hash_val = temp_desc.hash();
        
        LogicDescriptor loaded_desc;
        if (logic_storage->fetchByHash(hash_val, loaded_desc)) {
            // 注册到匹配器中并返回指针
            if (logic_matcher.registerLogic(loaded_desc)) {
                return logic_matcher.getLogicDescriptor(logic_id);
            }
        }
    }
    
    return nullptr;
}

// 根据logic_id执行注入
bool LogicInjector::injectLogicById(const std::string& logic_id, NeuronInput& input) {
    auto* logic_desc = getLogicById(logic_id);
    if (!logic_desc) {
        return false;
    }
    
    // 检查是否有回调函数
    if (!logic_desc->generate_input_callback) {
        return false;
    }
    
    // 执行注入回调
    logic_desc->generate_input_callback(logic_id, input);
    
    // 更新热度
    if (logic_storage) {
        double heat_increase = 5.0; // 固定热度增加
        std::string hash_val = logic_desc->hash();
        logic_storage->store(*logic_desc, logic_desc->feature, heat_increase);
    }
    
    return true;
}

std::vector<LogicDescriptor> LogicInjector::getMostActiveLogics(int k) {
    std::vector<LogicDescriptor> results;
    
    if (!logic_storage) {
        return results;
    }
    
    // 获取最热的k个Logic的slot_id
    auto hottest_ids = logic_storage->getHottestK(k);
    
    // 根据slot_id加载LogicDescriptor
    for (uint64_t slot_id : hottest_ids) {
        LogicDescriptor logic_desc;
        if (logic_storage->fetch(slot_id, logic_desc)) {
            results.push_back(logic_desc);
        }
    }
    
    return results;
}

void LogicInjector::setStorage(ExternalStorage<LogicDescriptor>* storage) {
    logic_storage = storage;
}

LogicSemanticMatcher* LogicInjector::getLogicMatcher() {
    return &logic_matcher;
}