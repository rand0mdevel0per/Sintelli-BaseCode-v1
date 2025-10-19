// 语义匹配器实现
// 集成语义匹配功能到神经元系统

#include "semantic_matcher.h"
#include <algorithm>
#include <random>

SemanticMatcher::SemanticMatcher(const std::string& model_path,
                               const std::string& vocab_path,
                               const std::string& merges_path,
                               const std::string& special_tokens_path) {
    extractor = std::make_unique<FeatureExtractor>(model_path, vocab_path, merges_path, special_tokens_path);
}

uint64_t SemanticMatcher::registerText(const std::string& text, const std::string& category) {
    if (text.empty() || !extractor || !extractor->isInitialized()) {
        return 0; // 无效的slot_id
    }
    
    uint64_t slot_id = generateSlotId();
    id_to_text[slot_id] = text;
    if (!category.empty()) {
        id_to_category[slot_id] = category;
    }
    
    return slot_id;
}

std::vector<uint64_t> SemanticMatcher::batchRegisterTexts(const std::vector<std::string>& texts, 
                                                         const std::string& category) {
    std::vector<uint64_t> slot_ids;
    slot_ids.reserve(texts.size());
    
    for (const auto& text : texts) {
        uint64_t slot_id = registerText(text, category);
        if (slot_id != 0) {
            slot_ids.push_back(slot_id);
        }
    }
    
    return slot_ids;
}

std::vector<SemanticMatch> SemanticMatcher::findSimilarTexts(const std::string& query_text, 
                                                            int top_k, 
                                                            double similarity_threshold,
                                                            const std::string& metric) {
    std::vector<SemanticMatch> results;
    
    if (!extractor || !extractor->isInitialized() || query_text.empty() || id_to_text.empty()) {
        return results;
    }
    
    // 获取所有已注册的文本
    std::vector<std::string> candidate_texts;
    std::vector<uint64_t> candidate_ids;
    
    for (const auto& pair : id_to_text) {
        candidate_texts.push_back(pair.second);
        candidate_ids.push_back(pair.first);
    }
    
    // 执行语义搜索
    auto similarity_results = extractor->semanticSearch(query_text, candidate_texts, 
                                                       top_k, similarity_threshold, metric);
    
    // 转换为SemanticMatch结果
    for (const auto& result : similarity_results) {
        size_t index = std::distance(candidate_texts.begin(), 
                                   std::find(candidate_texts.begin(), candidate_texts.end(), result.first));
        if (index < candidate_ids.size()) {
            uint64_t slot_id = candidate_ids[index];
            std::string category;
            auto cat_it = id_to_category.find(slot_id);
            if (cat_it != id_to_category.end()) {
                category = cat_it->second;
            }
            results.emplace_back(result.first, result.second, slot_id, category);
        }
    }
    
    // 按相似度排序
    std::sort(results.begin(), results.end());
    
    return results;
}

std::vector<SemanticMatch> SemanticMatcher::findSimilarTextsByCategory(const std::string& query_text,
                                                                      const std::string& category,
                                                                      int top_k,
                                                                      double similarity_threshold) {
    std::vector<SemanticMatch> results;
    
    if (!extractor || !extractor->isInitialized() || query_text.empty()) {
        return results;
    }
    
    // 获取指定类别的文本
    std::vector<std::string> candidate_texts;
    std::vector<uint64_t> candidate_ids;
    
    for (const auto& pair : id_to_text) {
        auto cat_it = id_to_category.find(pair.first);
        if (cat_it != id_to_category.end() && cat_it->second == category) {
            candidate_texts.push_back(pair.second);
            candidate_ids.push_back(pair.first);
        }
    }
    
    if (candidate_texts.empty()) {
        return results;
    }
    
    // 执行语义搜索
    auto similarity_results = extractor->semanticSearch(query_text, candidate_texts, 
                                                       top_k, similarity_threshold, "cosine");
    
    // 转换为SemanticMatch结果
    for (const auto& result : similarity_results) {
        size_t index = std::distance(candidate_texts.begin(), 
                                   std::find(candidate_texts.begin(), candidate_texts.end(), result.first));
        if (index < candidate_ids.size()) {
            results.emplace_back(result.first, result.second, candidate_ids[index], category);
        }
    }
    
    // 按相似度排序
    std::sort(results.begin(), results.end());
    
    return results;
}

std::vector<std::string> SemanticMatcher::getAllRegisteredTexts() const {
    std::vector<std::string> texts;
    texts.reserve(id_to_text.size());
    
    for (const auto& pair : id_to_text) {
        texts.push_back(pair.second);
    }
    
    return texts;
}

std::string SemanticMatcher::getTextById(uint64_t slot_id) const {
    auto it = id_to_text.find(slot_id);
    return (it != id_to_text.end()) ? it->second : "";
}

uint64_t SemanticMatcher::getIdByText(const std::string& text) const {
    for (const auto& pair : id_to_text) {
        if (pair.second == text) {
            return pair.first;
        }
    }
    return 0;
}

bool SemanticMatcher::removeText(uint64_t slot_id) {
    auto text_it = id_to_text.find(slot_id);
    if (text_it == id_to_text.end()) {
        return false;
    }
    
    id_to_text.erase(text_it);
    id_to_category.erase(slot_id);
    return true;
}

void SemanticMatcher::clearAll() {
    id_to_text.clear();
    id_to_category.clear();
    next_slot_id = 1;
}

SemanticMatcher::Stats SemanticMatcher::getStats() const {
    Stats stats;
    stats.total_texts = id_to_text.size();
    
    for (const auto& pair : id_to_category) {
        stats.category_counts[pair.second]++;
    }
    
    stats.unique_categories = stats.category_counts.size();
    return stats;
}

FeatureExtractor* SemanticMatcher::getFeatureExtractor() const {
    return extractor.get();
}

uint64_t SemanticMatcher::generateSlotId() const {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis(1, UINT64_MAX);
    
    uint64_t slot_id;
    do {
        slot_id = dis(gen);
    } while (id_to_text.find(slot_id) != id_to_text.end());
    
    return slot_id;
}

// IntegratedSemanticMatcher 实现
IntegratedSemanticMatcher::IntegratedSemanticMatcher(ExternalStorage<SemanticMatch>* ext_storage,
                                                   const std::string& model_path)
    : storage(ext_storage) {
    matcher = std::make_unique<SemanticMatcher>(model_path);
}

uint64_t IntegratedSemanticMatcher::registerAndStore(const std::string& text, const std::string& category) {
    if (!matcher) return 0;
    
    uint64_t slot_id = matcher->registerText(text, category);
    if (slot_id == 0 || !storage) return slot_id;
    
    // 创建语义匹配结果并存储
    SemanticMatch match(text, 1.0, slot_id, category); // 初始相似度为1.0（自相似）
    storage->store(match, 1.0); // 初始热度1.0
    
    return slot_id;
}

std::vector<SemanticMatch> IntegratedSemanticMatcher::searchAndStore(const std::string& query_text,
                                                                    int top_k,
                                                                    double similarity_threshold,
                                                                    const std::string& metric) {
    if (!matcher) return {};
    
    auto results = matcher->findSimilarTexts(query_text, top_k, similarity_threshold, metric);
    
    // 存储匹配结果
    if (storage) {
        for (const auto& match : results) {
            // 根据相似度设置热度
            double heat = match.similarity * 100.0; // 相似度转换为热度
            storage->store(match, heat);
        }
    }
    
    return results;
}

bool IntegratedSemanticMatcher::loadFromStorage(uint64_t slot_id, SemanticMatch& match) {
    if (!storage) return false;
    return storage->fetch(slot_id, match);
}

std::vector<SemanticMatch> IntegratedSemanticMatcher::getHottestMatches(int k) {
    if (!storage) return {};
    
    std::vector<SemanticMatch> results;
    auto hottest_ids = storage->getHottestK(k);
    
    for (uint64_t slot_id : hottest_ids) {
        SemanticMatch match;
        if (storage->fetch(slot_id, match)) {
            results.push_back(match);
        }
    }
    
    return results;
}

void IntegratedSemanticMatcher::setStorage(ExternalStorage<SemanticMatch>* ext_storage) {
    storage = ext_storage;
}

SemanticMatcher* IntegratedSemanticMatcher::getMatcher() const {
    return matcher.get();
}