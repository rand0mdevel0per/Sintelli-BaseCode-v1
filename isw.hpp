//
// Created by ASUS on 10/1/2025.
//

#ifndef GENERIC_EXTERNAL_STORAGE_H
#define GENERIC_EXTERNAL_STORAGE_H

#include <cuda_runtime.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include <cmath>
#include <memory>
#include <mutex>
#include <thread>
#include <functional>
#include <fstream>
#include <string>
#include <type_traits>
#include <algorithm>
#include <limits>
#include <codecvt>
#include <locale>
#include <optional>
#include "smry.cpp"
#include "semantic_query_engine.h"

// ===== 存储层级枚举 =====
enum StorageTier {
    TIER_L2_HOST_MEMORY,        // L2: Host内存池(热数据)
    TIER_L3_NVME_DISK           // L3: NVMe持久化(冷数据)
};

struct Statistics {
    size_t l2_size;
    size_t l3_size;
    size_t total_size;
    double avg_heat;
    double max_heat;
    double min_heat;
    uint32_t current_time;
    size_t total_memory_bytes;
};

// ===== 特征向量类型定义 =====
template<typename T>
struct FeatureVector {
    std::vector<T> data;
    size_t dimension;
    std::string feature_type;
    
    FeatureVector() : dimension(0), feature_type("generic") {}
    
    FeatureVector(const std::vector<T>& vec, const std::string& type = "generic")
        : data(vec), dimension(vec.size()), feature_type(type) {}
    
    // 计算欧几里得距离
    double euclideanDistance(const FeatureVector<T>& other) const {
        if (dimension != other.dimension) {
            return std::numeric_limits<double>::max();
        }
        
        double sum = 0.0;
        for (size_t i = 0; i < dimension; ++i) {
            double diff = static_cast<double>(data[i]) - static_cast<double>(other.data[i]);
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
    
    // 计算余弦相似度
    double cosineSimilarity(const FeatureVector<T>& other) const {
        if (dimension != other.dimension) {
            return -1.0;
        }
        
        double dot_product = 0.0;
        double norm_a = 0.0;
        double norm_b = 0.0;
        
        for (size_t i = 0; i < dimension; ++i) {
            dot_product += static_cast<double>(data[i]) * static_cast<double>(other.data[i]);
            norm_a += static_cast<double>(data[i]) * static_cast<double>(data[i]);
            norm_b += static_cast<double>(other.data[i]) * static_cast<double>(other.data[i]);
        }
        
        if (norm_a == 0.0 || norm_b == 0.0) {
            return 0.0;
        }
        
        return dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }
    
    // 计算曼哈顿距离
    double manhattanDistance(const FeatureVector<T>& other) const {
        if (dimension != other.dimension) {
            return std::numeric_limits<double>::max();
        }
        
        double sum = 0.0;
        for (size_t i = 0; i < dimension; ++i) {
            sum += std::abs(static_cast<double>(data[i]) - static_cast<double>(other.data[i]));
        }
        return sum;
    }
    
    // 序列化用于hash计算
    std::string serialize() const {
        std::string result = feature_type + "|" + std::to_string(dimension) + "|";
        for (const auto& val : data) {
            result += std::to_string(val) + ",";
        }
        return result;
    }
};

// ===== 数据描述符 =====
struct DataDescriptor {
    uint64_t slot_id;
    std::string hash_value;            // 数据的hash值
    double heat;                    // ISW热度
    uint32_t last_access_time;
    uint32_t access_count;
    StorageTier tier;
    size_t data_size;               // 数据大小(字节)
    std::string feature_hash;       // 特征向量hash值
    std::string feature_type;       // 特征类型
    size_t feature_dimension;       // 特征维度

    // ISW热度计算喵
    double computeHeat(uint32_t current_time) const {
        double heat_sum = 0.0;
        uint32_t delta_t = current_time > last_access_time ?
                          (current_time - last_access_time) : 1;

        // ISW: 热度 = Σ(1/Δt²) × 访问次数
        heat_sum = 1.0 / (delta_t * delta_t);
        return heat_sum * access_count;
    }

    // 检查是否有特征向量 喵
    bool hasFeature() const {
        return !feature_hash.empty() && feature_dimension > 0;
    }
};

// ===== Van Emde Boas树(简化但高效的实现) =====
template<typename ValueType>
class vEB_Tree {
private:
    struct Node {
        uint32_t min_key, max_key;
        bool has_min, has_max;
        std::shared_ptr<ValueType> min_val, max_val;
        std::shared_ptr<Node> summary;
        std::vector<std::shared_ptr<Node>> clusters;
        uint32_t universe_size;

        Node(uint32_t u) : min_key(0), max_key(0), universe_size(u), has_min(false), has_max(false) {
            if (u > 2) {
                uint32_t sqrt_u = 1 << ((int) log2(u) / 2);
                summary = std::make_shared<Node>(sqrt_u);
                clusters.resize(sqrt_u);
                for (auto &c: clusters) {
                    c = std::make_shared<Node>(sqrt_u);
                }
            }
        }

        uint32_t high(uint32_t x) const {
            uint32_t sqrt_u = 1 << ((int)log2(universe_size) / 2);
            return x / sqrt_u;
        }

        uint32_t low(uint32_t x) const {
            uint32_t sqrt_u = 1 << ((int)log2(universe_size) / 2);
            return x % sqrt_u;
        }

        uint32_t index(uint32_t h, uint32_t l) const {
            uint32_t sqrt_u = 1 << ((int)log2(universe_size) / 2);
            return h * sqrt_u + l;
        }
    };

    std::shared_ptr<Node> root;
    std::unordered_map<uint32_t, std::shared_ptr<ValueType>> value_map;

public:
    vEB_Tree(uint32_t universe_size) {
        uint32_t u = 1;
        while (u < universe_size) u <<= 1;
        root = std::make_shared<Node>(u);
    }

    // 插入 O(log log U) 喵
    void insert(uint32_t key, const ValueType& value) {
        value_map[key] = std::make_shared<ValueType>(value);
        insert_recursive(root, key);
    }

    // 删除 O(log log U) 喵
    void remove(uint32_t key) {
        value_map.erase(key);
        remove_recursive(root, key);
    }

    // 查找后继 O(log log U) 喵
    uint32_t successor(uint32_t key) {
        return successor_recursive(root, key);
    }

    // 查找前驱 O(log log U) 喵
    uint32_t predecessor(uint32_t key) {
        return predecessor_recursive(root, key);
    }

    // 获取最小key
    uint32_t minimum() {
        if (root->has_min) return root->min_key;
        return UINT32_MAX;
    }

    // 获取最大key
    uint32_t maximum() {
        if (root->has_max) return root->max_key;
        return 0;
    }

    // 获取值
    std::shared_ptr<ValueType> get(uint32_t key) {
        auto it = value_map.find(key);
        return (it != value_map.end()) ? it->second : nullptr;
    }

    // 获取所有key(按序)
    std::vector<uint32_t> getAllKeys() {
        std::vector<uint32_t> keys;
        if (!root->has_min) return keys;

        uint32_t key = minimum();
        keys.push_back(key);

        while (key != maximum()) {
            key = successor(key);
            if (key != UINT32_MAX) {
                keys.push_back(key);
            } else {
                break;
            }
        }
        return keys;
    }

private:
    void insert_recursive(std::shared_ptr<Node> v, uint32_t x) {
        if (!v->has_min) {
            v->min_key = v->max_key = x;
            v->has_min = v->has_max = true;
            return;
        }

        if (x < v->min_key) {
            std::swap(x, v->min_key);
        }

        if (v->universe_size > 2) {
            uint32_t h = v->high(x);
            uint32_t l = v->low(x);

            if (!v->clusters[h]->has_min) {
                insert_recursive(v->summary, h);
            }
            insert_recursive(v->clusters[h], l);
        }

        if (x > v->max_key) {
            v->max_key = x;
        }
    }

    void remove_recursive(std::shared_ptr<Node> v, uint32_t x) {
        if (!v->has_min || !v->has_max) return;

        if (v->min_key == v->max_key && v->min_key == x) {
            v->has_min = v->has_max = false;
            return;
        }

        if (v->universe_size == 2) {
            if (x == 0) {
                v->min_key = 1;
            } else {
                v->max_key = 0;
            }
            if (v->min_key > v->max_key) {
                v->has_min = v->has_max = false;
            }
            return;
        }

        // 简化实现
        if (x == v->min_key) {
            if (v->summary && v->summary->has_min) {
                uint32_t first_cluster = v->summary->min_key;
                x = v->index(first_cluster, v->clusters[first_cluster]->min_key);
                v->min_key = x;
            } else {
                v->has_min = v->has_max = false;
                return;
            }
        }

        uint32_t h = v->high(x);
        uint32_t l = v->low(x);
        remove_recursive(v->clusters[h], l);

        if (!v->clusters[h]->has_min && v->summary) {
            remove_recursive(v->summary, h);
        }
    }

    uint32_t successor_recursive(std::shared_ptr<Node> v, uint32_t x) {
        if (v->universe_size == 2) {
            if (x == 0 && v->has_max && v->max_key == 1) {
                return 1;
            }
            return UINT32_MAX;
        }

        if (v->has_min && x < v->min_key) {
            return v->min_key;
        }

        uint32_t h = v->high(x);
        uint32_t l = v->low(x);

        if (h < v->clusters.size() && v->clusters[h]->has_max &&
            l < v->clusters[h]->max_key) {
            uint32_t offset = successor_recursive(v->clusters[h], l);
            return v->index(h, offset);
        }

        if (v->summary) {
            uint32_t succ_cluster = successor_recursive(v->summary, h);
            if (succ_cluster != UINT32_MAX && succ_cluster < v->clusters.size()) {
                uint32_t offset = v->clusters[succ_cluster]->min_key;
                return v->index(succ_cluster, offset);
            }
        }

        return UINT32_MAX;
    }

    uint32_t predecessor_recursive(std::shared_ptr<Node> v, uint32_t x) {
        if (v->universe_size == 2) {
            if (x == 1 && v->has_min && v->min_key == 0) {
                return 0;
            }
            return UINT32_MAX;
        }

        if (v->has_max && x > v->max_key) {
            return v->max_key;
        }

        uint32_t h = v->high(x);
        uint32_t l = v->low(x);

        if (h < v->clusters.size() && v->clusters[h]->has_min &&
            l > v->clusters[h]->min_key) {
            uint32_t offset = predecessor_recursive(v->clusters[h], l);
            return v->index(h, offset);
        }

        if (v->summary) {
            uint32_t pred_cluster = predecessor_recursive(v->summary, h);
            if (pred_cluster != UINT32_MAX && pred_cluster < v->clusters.size()) {
                uint32_t offset = v->clusters[pred_cluster]->max_key;
                return v->index(pred_cluster, offset);
            }
        }

        if (v->has_min && x > v->min_key) {
            return v->min_key;
        }

        return UINT32_MAX;
    }
};

// ===== 泛型外部存储系统 =====
template<typename T>
class ExternalStorage {
private:
    // L2内存池
    std::unordered_map<uint64_t, T> l2_memory_pool;

    // 热度索引(vEB树)
    std::unique_ptr<vEB_Tree<DataDescriptor>> heat_index;

    // 描述符映射
    std::unordered_map<uint64_t, DataDescriptor> descriptor_map;

    // Hash到slot_id的映射(支持按hash查找)
    std::unordered_map<std::string, uint64_t> hash_to_slot;

    // 特征索引(特征hash到slot_id的映射)
    std::unordered_map<std::string, std::vector<uint64_t>> feature_index;

    // 配置参数
    size_t l2_max_size;
    double promote_threshold;
    double demote_threshold;

    // 统计信息
    uint32_t current_time;
    uint64_t next_slot_id;

    // 线程安全
    mutable std::mutex storage_mutex;

    // 持久化路径
    std::string persistence_path;

public:
    // 构造函数喵
    ExternalStorage(size_t max_l2_size = 1024,
                    double promote_thresh = 100.0,
                    double demote_thresh = 1.0,
                    const std::string& persist_path = "./storage_cache/")
        : l2_max_size(max_l2_size),
          promote_threshold(promote_thresh),
          demote_threshold(demote_thresh),
          current_time(0),
          next_slot_id(0),
          persistence_path(persist_path)
    {
        // 初始化vEB树(假设热度范围0-65535)
        heat_index = std::make_unique<vEB_Tree<DataDescriptor>>(65536);
    }

    // ===== 核心API =====

    // 存储数据(需要数据提供hash()函数) 喵
    template<typename = std::enable_if_t<
        std::is_invocable_r_v<std::string, decltype(&T::hash), T>>>
    uint64_t store(const T& data, double initial_heat = 1.0) {
        return storeWithFeature(data, FeatureVector<float>(), initial_heat);
    }

    // 存储数据(带特征向量) 喵
    template<typename = std::enable_if_t<
        std::is_invocable_r_v<std::string, decltype(&T::hash), T>>>
    uint64_t store(const T& data, const FeatureVector<float>& feature, double initial_heat = 1.0) {
        return storeWithFeature(data, feature, initial_heat);
    }

    // 通过slot_id获取数据 喵
    bool fetch(uint64_t slot_id, T& out_data) {
        std::lock_guard<std::mutex> lock(storage_mutex);
        current_time++;

        auto it = descriptor_map.find(slot_id);
        if (it == descriptor_map.end()) {
            return false;
        }

        return fetchInternal(it->second, out_data);
    }

    // 通过hash值获取数据 喵
    template<typename = std::enable_if_t<
        std::is_invocable_r_v<std::string, decltype(&T::hash), T>>>
    bool fetchByHash(std::string hash_val, T& out_data) {
        std::lock_guard<std::mutex> lock(storage_mutex);

        auto hash_it = hash_to_slot.find(hash_val);
        if (hash_it == hash_to_slot.end()) {
            return false;
        }

        current_time++;
        auto desc_it = descriptor_map.find(hash_it->second);
        if (desc_it == descriptor_map.end()) {
            return false;
        }

        return fetchInternal(desc_it->second, out_data);
    }

    // 批量获取(优化传输) 喵
    std::vector<T> fetchBatch(const std::vector<uint64_t>& slot_ids) {
        std::vector<T> results;
        results.reserve(slot_ids.size());

        for (uint64_t id : slot_ids) {
            T data;
            if (fetch(id, data)) {
                results.push_back(std::move(data));
            }
        }

        return results;
    }

    // 获取最热的K个数据的slot_id 喵
    std::vector<uint64_t> getHottestK(int k) {
        std::lock_guard<std::mutex> lock(storage_mutex);

        std::vector<uint64_t> result;
        result.reserve(k);

        uint32_t key = heat_index->maximum();
        for (int i = 0; i < k && key != 0; i++) {
            auto desc = heat_index->get(key);
            if (desc) {
                result.push_back(desc->slot_id);
            }
            key = heat_index->predecessor(key);
            if (key == UINT32_MAX) break;
        }

        return result;
    }

    // 获取最冷的K个数据的slot_id 喵
    std::vector<uint64_t> getColdestK(int k) {
        std::lock_guard<std::mutex> lock(storage_mutex);

        std::vector<uint64_t> result;
        result.reserve(k);

        uint32_t key = heat_index->minimum();
        for (int i = 0; i < k && key != UINT32_MAX; i++) {
            auto desc = heat_index->get(key);
            if (desc) {
                result.push_back(desc->slot_id);
            }
            key = heat_index->successor(key);
        }

        return result;
    }

    // ===== 特征匹配API =====

    // 存储数据并关联特征向量 喵
    template<typename = std::enable_if_t<
        std::is_invocable_r_v<std::string, decltype(&T::hash), T>>>
    uint64_t storeWithFeature(const T& data, const FeatureVector<float>& feature, double initial_heat = 1.0) {
        std::lock_guard<std::mutex> lock(storage_mutex);

        // 获取数据的hash值
        std::string hash_val = data.hash();
        std::string feature_hash = computeFeatureHash(feature);

        // 检查是否已存在
        auto hash_it = hash_to_slot.find(hash_val);
        if (hash_it != hash_to_slot.end()) {
            // 已存在,更新并返回
            return updateExistingWithFeature(hash_it->second, data, feature, initial_heat);
        }

        // 分配新的slot_id
        uint64_t slot_id = next_slot_id++;

        // 创建描述符
        DataDescriptor desc;
        desc.slot_id = slot_id;
        desc.hash_value = hash_val;
        desc.feature_hash = feature_hash;
        desc.feature_type = feature.feature_type;
        desc.feature_dimension = feature.dimension;
        desc.heat = initial_heat;
        desc.last_access_time = current_time;
        desc.access_count = 1;
        desc.tier = TIER_L2_HOST_MEMORY;
        desc.data_size = sizeof(T);

        // 存储到L2
        l2_memory_pool[slot_id] = data;
        descriptor_map[slot_id] = desc;
        hash_to_slot[hash_val] = slot_id;

        // 更新特征索引
        feature_index[feature_hash].push_back(slot_id);

        // 插入热度索引
        uint32_t heat_key = heatToKey(initial_heat);
        heat_index->insert(heat_key, desc);

        // 检查L2容量
        if (l2_memory_pool.size() > l2_max_size) {
            evictColdest();
        }

        return slot_id;
    }

    // 通过特征向量查找最相似的K个数据 喵
    std::vector<std::pair<uint64_t, double>> findSimilarByFeature(
        const FeatureVector<float>& query_feature, 
        int k = 10, 
        double similarity_threshold = 0.0,
        const std::string& distance_metric = "cosine") {
        
        std::lock_guard<std::mutex> lock(storage_mutex);
        std::vector<std::pair<uint64_t, double>> results;
        
        for (const auto& pair : descriptor_map) {
            const DataDescriptor& desc = pair.second;
            
            // 跳过没有特征的数据
            if (!desc.hasFeature()) continue;
            
            // 获取数据来计算相似度
            T data;
            if (fetchInternal(const_cast<DataDescriptor&>(desc), data)) {
                // 这里需要数据对象有getFeature()方法来获取特征向量
                // 假设T类型有getFeature()方法返回FeatureVector<float>
                FeatureVector<float> stored_feature = data.getFeature();
                
                double similarity = 0.0;
                if (distance_metric == "cosine") {
                    similarity = stored_feature.cosineSimilarity(query_feature);
                } else if (distance_metric == "euclidean") {
                    // 转换为相似度: 1 / (1 + distance)
                    double distance = stored_feature.euclideanDistance(query_feature);
                    similarity = 1.0 / (1.0 + distance);
                } else if (distance_metric == "manhattan") {
                    double distance = stored_feature.manhattanDistance(query_feature);
                    similarity = 1.0 / (1.0 + distance);
                }
                
                if (similarity >= similarity_threshold) {
                    results.emplace_back(desc.slot_id, similarity);
                }
            }
        }
        
        // 按相似度排序
        std::sort(results.begin(), results.end(), 
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // 返回前K个结果
        if (results.size() > static_cast<size_t>(k)) {
            results.resize(k);
        }
        
        return results;
    }

    // 通过特征hash查找数据 喵
    std::vector<uint64_t> findByFeatureHash(const std::string& feature_hash) {
        std::lock_guard<std::mutex> lock(storage_mutex);
        
        auto it = feature_index.find(feature_hash);
        if (it != feature_index.end()) {
            return it->second;
        }
        return {};
    }

    // 批量存储特征向量 喵
    template<typename = std::enable_if_t<
        std::is_invocable_r_v<std::string, decltype(&T::hash), T>>>
    std::vector<uint64_t> batchStoreWithFeatures(
        const std::vector<std::pair<T, FeatureVector<float>>>& data_features,
        double initial_heat = 1.0) {
        
        std::vector<uint64_t> slot_ids;
        slot_ids.reserve(data_features.size());
        
        for (const auto& pair : data_features) {
            uint64_t slot_id = storeWithFeature(pair.first, pair.second, initial_heat);
            slot_ids.push_back(slot_id);
        }
        
        return slot_ids;
    }

    // 获取特征统计信息 喵
    struct FeatureStatistics {
        size_t total_features;
        size_t unique_feature_types;
        std::unordered_map<std::string, size_t> feature_type_counts;
        std::unordered_map<size_t, size_t> dimension_counts;
    };

    FeatureStatistics getFeatureStatistics() const {
        std::lock_guard<std::mutex> lock(storage_mutex);
        
        FeatureStatistics stats;
        stats.total_features = 0;
        
        for (const auto& pair : descriptor_map) {
            if (pair.second.hasFeature()) {
                stats.total_features++;
                stats.feature_type_counts[pair.second.feature_type]++;
                stats.dimension_counts[pair.second.feature_dimension]++;
            }
        }
        
        stats.unique_feature_types = stats.feature_type_counts.size();
        return stats;
    }

    // 根据特征类型查找数据 喵
    std::vector<uint64_t> findByFeatureType(const std::string& feature_type) {
        std::lock_guard<std::mutex> lock(storage_mutex);
        
        std::vector<uint64_t> result;
        for (const auto& pair : descriptor_map) {
            if (pair.second.feature_type == feature_type) {
                result.push_back(pair.first);
            }
        }
        return result;
    }

    // 清空特征索引 喵
    void clearFeatureIndex() {
        std::lock_guard<std::mutex> lock(storage_mutex);
        feature_index.clear();
    }

    // 重建特征索引(用于数据恢复等情况) 喵
    void rebuildFeatureIndex() {
        std::lock_guard<std::mutex> lock(storage_mutex);
        
        feature_index.clear();
        for (const auto& pair : descriptor_map) {
            if (pair.second.hasFeature()) {
                feature_index[pair.second.feature_hash].push_back(pair.first);
            }
        }
    }

    // 获取所有特征类型 喵
    std::vector<std::string> getAllFeatureTypes() const {
        std::lock_guard<std::mutex> lock(storage_mutex);
        
        std::unordered_set<std::string> types;
        for (const auto& pair : descriptor_map) {
            if (!pair.second.feature_type.empty()) {
                types.insert(pair.second.feature_type);
            }
        }
        return std::vector<std::string>(types.begin(), types.end());
    }

    // ===== 语义查询API =====

    // 语义查询接口(需要T类型实现getText()方法) 喵
    template<typename = std::enable_if_t<
        std::is_invocable_r_v<std::string, decltype(&T::getText), T>>>
    std::vector<std::pair<uint64_t, double>> semanticSearch(
        const std::string& query_text,
        int k = 10,
        double similarity_threshold = 0.0,
        const std::string& model_path = "",
        const std::string& vocab_path = "",
        const std::string& merges_path = "",
        const std::string& special_tokens_path = "") {
        
        std::lock_guard<std::mutex> lock(storage_mutex);
        
        // 如果E5模型未初始化，尝试初始化
        static std::unique_ptr<SemanticQueryEngine> semantic_engine;
        if (!semantic_engine) {
            semantic_engine = std::make_unique<SemanticQueryEngine>(
                model_path, vocab_path, merges_path, special_tokens_path);
        }
        
        // 获取查询文本的语义向量
        FeatureVector<float> query_feature;
        if (!semantic_engine->getTextEmbedding(query_text, query_feature)) {
            return {};
        }
        
        // 进行相似度搜索
        return findSimilarByFeature(query_feature, k, similarity_threshold, "cosine");
    }

    // 语义查询接口(宽字符版本) 喵
    template<typename = std::enable_if_t<
        std::is_invocable_r_v<std::string, decltype(&T::getText), T>>>
    std::vector<std::pair<uint64_t, double>> semanticSearch(
        const std::wstring& query_text,
        int k = 10,
        double similarity_threshold = 0.0,
        const std::string& model_path = "",
        const std::string& vocab_path = "",
        const std::string& merges_path = "",
        const std::string& special_tokens_path = "") {
        
        // 宽字符转UTF-8
        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        std::string utf8_query = converter.to_bytes(query_text);
        
        return semanticSearch(utf8_query, k, similarity_threshold, 
                             model_path, vocab_path, merges_path, special_tokens_path);
    }

    // 存储数据并自动计算语义特征 喵
    template<typename = std::enable_if_t<
        std::is_invocable_r_v<std::string, decltype(&T::hash), T> &&
        std::is_invocable_r_v<std::string, decltype(&T::getText), T>>>
    uint64_t storeWithSemanticFeature(const T& data, 
                                     double initial_heat = 1.0,
                                     const std::string& model_path = "",
                                     const std::string& vocab_path = "",
                                     const std::string& merges_path = "",
                                     const std::string& special_tokens_path = "") {
        
        std::lock_guard<std::mutex> lock(storage_mutex);
        
        // 初始化语义引擎
        static std::unique_ptr<SemanticQueryEngine> semantic_engine;
        if (!semantic_engine) {
            semantic_engine = std::make_unique<SemanticQueryEngine>(
                model_path, vocab_path, merges_path, special_tokens_path);
        }
        
        // 获取文本内容的语义特征
        std::string text = data.getText();
        FeatureVector<float> semantic_feature;
        
        if (!semantic_engine->getTextEmbedding(text, semantic_feature)) {
            // 如果语义特征提取失败，使用空特征
            semantic_feature = FeatureVector<float>();
        }
        
        // 存储数据并关联语义特征
        return storeWithFeature(data, semantic_feature, initial_heat);
    }

    // 批量存储数据并自动计算语义特征 喵
    template<typename = std::enable_if_t<
        std::is_invocable_r_v<std::string, decltype(&T::hash), T> &&
        std::is_invocable_r_v<std::string, decltype(&T::getText), T>>>
    std::vector<uint64_t> batchStoreWithSemanticFeatures(
        const std::vector<T>& data_list,
        double initial_heat = 1.0,
        const std::string& model_path = "",
        const std::string& vocab_path = "",
        const std::string& merges_path = "",
        const std::string& special_tokens_path = "") {
        
        std::vector<uint64_t> slot_ids;
        slot_ids.reserve(data_list.size());
        
        for (const auto& data : data_list) {
            uint64_t slot_id = storeWithSemanticFeature(data, initial_heat, 
                                                       model_path, vocab_path, 
                                                       merges_path, special_tokens_path);
            slot_ids.push_back(slot_id);
        }
        
        return slot_ids;
    }

    Statistics getStatistics() const {
        std::lock_guard<std::mutex> lock(storage_mutex);

        Statistics stats;
        stats.l2_size = l2_memory_pool.size();
        stats.total_size = descriptor_map.size();
        stats.l3_size = stats.total_size - stats.l2_size;
        stats.current_time = current_time;
        stats.total_memory_bytes = l2_memory_pool.size() * sizeof(T);

        double total_heat = 0.0;
        stats.max_heat = 0.0;
        stats.min_heat = 1e9;

        for (const auto& pair : descriptor_map) {
            double h = pair.second.heat;
            total_heat += h;
            stats.max_heat = std::max(stats.max_heat, h);
            stats.min_heat = std::min(stats.min_heat, h);
        }

        stats.avg_heat = stats.total_size > 0 ?
                        (total_heat / stats.total_size) : 0.0;

        return stats;
    }

private:

    // 删除数据 喵
    void remove(uint64_t slot_id) {
        std::lock_guard<std::mutex> lock(storage_mutex);

        auto it = descriptor_map.find(slot_id);
        if (it == descriptor_map.end()) return;

        // 从热度索引移除
        uint32_t heat_key = heatToKey(it->second.heat);
        heat_index->remove(heat_key);

        // 从hash映射移除
        hash_to_slot.erase(it->second.hash_value);

        // 从特征索引移除
        if (!it->second.feature_hash.empty()) {
            auto feature_it = feature_index.find(it->second.feature_hash);
            if (feature_it != feature_index.end()) {
                auto& slots = feature_it->second;
                slots.erase(std::remove(slots.begin(), slots.end(), slot_id), slots.end());
                if (slots.empty()) {
                    feature_index.erase(feature_it);
                }
            }
        }

        // 从L2移除
        l2_memory_pool.erase(slot_id);

        // 从L3移除(如果存在)
        if (it->second.tier == TIER_L3_NVME_DISK) {
            removeFromDisk(slot_id);
        }

        // 移除描述符
        descriptor_map.erase(it);
    }

    // ===== 热度管理 =====

    // 全局热度衰减(周期性调用) 喵
    void globalHeatDecay(double decay_factor = 0.95) {
        std::lock_guard<std::mutex> lock(storage_mutex);

        std::vector<uint64_t> to_demote;

        for (auto& pair : descriptor_map) {
            DataDescriptor& desc = pair.second;

            // 移除旧热度索引
            uint32_t old_key = heatToKey(desc.heat);
            heat_index->remove(old_key);

            // 衰减热度
            desc.heat *= decay_factor;

            // 插入新热度索引
            uint32_t new_key = heatToKey(desc.heat);
            heat_index->insert(new_key, desc);

            // 检查是否需要降级
            if (desc.heat < demote_threshold &&
                desc.tier == TIER_L2_HOST_MEMORY) {
                to_demote.push_back(pair.first);
                }
        }

        // 执行降级
        for (uint64_t slot_id : to_demote) {
            demoteToL3(slot_id);
        }
    }

    // 手动调整热度 喵
    void adjustHeat(uint64_t slot_id, double heat_delta) {
        std::lock_guard<std::mutex> lock(storage_mutex);

        auto it = descriptor_map.find(slot_id);
        if (it == descriptor_map.end()) return;

        DataDescriptor& desc = it->second;

        // 移除旧索引
        uint32_t old_key = heatToKey(desc.heat);
        heat_index->remove(old_key);

        // 调整热度
        desc.heat = std::max(0.0, desc.heat + heat_delta);

        // 插入新索引
        uint32_t new_key = heatToKey(desc.heat);
        heat_index->insert(new_key, desc);
    }

    // ===== 统计信息 =====





    // 获取描述符 喵
    bool getDescriptor(uint64_t slot_id, DataDescriptor& out_desc) const {
        std::lock_guard<std::mutex> lock(storage_mutex);

        auto it = descriptor_map.find(slot_id);
        if (it != descriptor_map.end()) {
            out_desc = it->second;
            return true;
        }
        return false;
    }

    // 获取描述符的可选版本 喵
    std::optional<DataDescriptor> getDescriptor(uint64_t slot_id) const {
        std::lock_guard<std::mutex> lock(storage_mutex);

        auto it = descriptor_map.find(slot_id);
        if (it != descriptor_map.end()) {
            return it->second;
        }
        return std::nullopt;
    }

private:
    // ===== 内部辅助函数 =====

    // 热度转换为vEB树的key(归一化到0-65535) 喵
    uint32_t heatToKey(double heat) const {
        // 使用log缩放,确保分布合理
        double normalized = std::log(heat + 1.0) / std::log(100000.0);
        normalized = std::max(0.0, std::min(1.0, normalized));
        return static_cast<uint32_t>(normalized * 65535.0);
    }

    // 计算特征向量hash 喵
    std::string computeFeatureHash(const FeatureVector<float>& feature) const {
        // 使用简单的hash算法,实际应用中可以使用更复杂的hash
        std::hash<std::string> hasher;
        return std::to_string(hasher(feature.serialize()));
    }

    // 更新已存在的数据(带特征) 喵
    uint64_t updateExistingWithFeature(uint64_t slot_id, const T& data, const FeatureVector<float>& feature, double heat) {
        auto it = descriptor_map.find(slot_id);
        if (it == descriptor_map.end()) return slot_id;

        // 移除旧特征索引
        if (!it->second.feature_hash.empty()) {
            auto feature_it = feature_index.find(it->second.feature_hash);
            if (feature_it != feature_index.end()) {
                auto& slots = feature_it->second;
                slots.erase(std::remove(slots.begin(), slots.end(), slot_id), slots.end());
                if (slots.empty()) {
                    feature_index.erase(feature_it);
                }
            }
        }

        // 更新数据
        if (it->second.tier == TIER_L2_HOST_MEMORY) {
            l2_memory_pool[slot_id] = data;
        } else {
            // 晋升回L2
            l2_memory_pool[slot_id] = data;
            it->second.tier = TIER_L2_HOST_MEMORY;
        }

        // 更新特征信息
        std::string new_feature_hash = computeFeatureHash(feature);
        it->second.feature_hash = new_feature_hash;
        it->second.feature_type = feature.feature_type;
        it->second.feature_dimension = feature.dimension;

        // 更新特征索引
        feature_index[new_feature_hash].push_back(slot_id);

        // 更新热度
        uint32_t old_key = heatToKey(it->second.heat);
        heat_index->remove(old_key);

        it->second.heat = std::max(it->second.heat, heat);
        it->second.access_count++;
        it->second.last_access_time = current_time;

        uint32_t new_key = heatToKey(it->second.heat);
        heat_index->insert(new_key, it->second);

        return slot_id;
    }

    // 更新已存在的数据 喵
    uint64_t updateExisting(uint64_t slot_id, const T& data, double heat) {
        return updateExistingWithFeature(slot_id, data, FeatureVector<float>(), heat);
    }

    // 更新已存在的数据(带特征) 喵
    uint64_t updateExisting(uint64_t slot_id, const T& data, const FeatureVector<float>& feature, double heat) {
        return updateExistingWithFeature(slot_id, data, feature, heat);
    }

    // 内部获取函数 喵
    bool fetchInternal(DataDescriptor& desc, T& out_data) {
        // 更新访问统计
        desc.access_count++;
        uint32_t old_access_time = desc.last_access_time;
        desc.last_access_time = current_time;

        // 重新计算热度
        uint32_t old_key = heatToKey(desc.heat);
        heat_index->remove(old_key);

        desc.heat = desc.computeHeat(current_time);

        uint32_t new_key = heatToKey(desc.heat);
        heat_index->insert(new_key, desc);

        // 根据tier获取数据
        if (desc.tier == TIER_L2_HOST_MEMORY) {
            auto data_it = l2_memory_pool.find(desc.slot_id);
            if (data_it != l2_memory_pool.end()) {
                out_data = data_it->second;
                return true;
            }
        } else if (desc.tier == TIER_L3_NVME_DISK) {
            // 从磁盘加载
            if (loadFromDisk(desc.slot_id, out_data)) {
                // 晋升到L2
                l2_memory_pool[desc.slot_id] = out_data;
                desc.tier = TIER_L2_HOST_MEMORY;

                // 检查L2容量
                if (l2_memory_pool.size() > l2_max_size) {
                    evictColdest();
                }

                return true;
            }
        }

        return false;
    }

    // 驱逐最冷的数据 喵
    void evictColdest() {
        uint32_t coldest_key = heat_index->minimum();
        auto desc = heat_index->get(coldest_key);

        if (desc) {
            demoteToL3(desc->slot_id);
        }
    }

    // 降级到L3 喵
    void demoteToL3(uint64_t slot_id) {
        auto it = descriptor_map.find(slot_id);
        if (it == descriptor_map.end()) return;

        auto data_it = l2_memory_pool.find(slot_id);
        if (data_it != l2_memory_pool.end()) {
            // 保存到磁盘
            saveToDisk(slot_id, data_it->second);

            // 从L2移除
            l2_memory_pool.erase(data_it);
        }

        it->second.tier = TIER_L3_NVME_DISK;
    }

    // 持久化到磁盘 喵
    void saveToDisk(uint64_t slot_id, const T& data) {
        std::string filename = persistence_path + std::to_string(slot_id) + ".bin";
        std::ofstream ofs(filename, std::ios::binary);
        if (ofs) {
            ofs.write(reinterpret_cast<const char*>(&data), sizeof(T));
        }
    }

    // 从磁盘加载 喵
    bool loadFromDisk(uint64_t slot_id, T& out_data) {
        std::string filename = persistence_path + std::to_string(slot_id) + ".bin";
        std::ifstream ifs(filename, std::ios::binary);
        if (ifs) {
            ifs.read(reinterpret_cast<char*>(&out_data), sizeof(T));
            return ifs.good();
        }
        return false;
    }

    // 从磁盘删除 喵
    void removeFromDisk(uint64_t slot_id) {
        std::string filename = persistence_path + std::to_string(slot_id) + ".bin";
        std::remove(filename.c_str());
    }

};

#endif // GENERIC_EXTERNAL_STORAGE_H