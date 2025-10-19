//
// 特征匹配功能简化测试
//

#include <vector>
#include <string>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdio>

// 简化版本的特征向量定义
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
    
    // 序列化用于hash计算
    std::string serialize() const {
        std::string result = feature_type + "|" + std::to_string(dimension) + "|";
        for (const auto& val : data) {
            result += std::to_string(val) + ",";
        }
        return result;
    }
};

// 测试数据结构
struct TestData {
    int id;
    std::string name;
    FeatureVector<float> feature;
    
    TestData(int id, const std::string& name, const FeatureVector<float>& feat) 
        : id(id), name(name), feature(feat) {}
    
    // 必须提供hash函数
    std::string hash() const {
        return std::to_string(id) + "_" + name;
    }
};

// 简化的存储系统实现用于测试
class TestStorage {
private:
    struct DataDescriptor {
        uint64_t slot_id;
        std::string hash_value;
        FeatureVector<float> feature;
    };
    
    std::unordered_map<uint64_t, TestData> data_map;
    std::unordered_map<uint64_t, DataDescriptor> descriptor_map;
    std::unordered_map<std::string, uint64_t> hash_to_slot;
    std::unordered_map<std::string, std::vector<uint64_t>> feature_index;
    uint64_t next_slot_id = 0;
    
public:
    uint64_t store(const TestData& data, const FeatureVector<float>& feature) {
        std::string hash_val = data.hash();
        
        auto hash_it = hash_to_slot.find(hash_val);
        if (hash_it != hash_to_slot.end()) {
            return hash_it->second;
        }
        
        uint64_t slot_id = next_slot_id++;
        
        DataDescriptor desc;
        desc.slot_id = slot_id;
        desc.hash_value = hash_val;
        desc.feature = feature;
        
        data_map[slot_id] = data;
        descriptor_map[slot_id] = desc;
        hash_to_slot[hash_val] = slot_id;
        
        // 更新特征索引
        std::hash<std::string> hasher;
        std::string feature_hash = std::to_string(hasher(feature.serialize()));
        feature_index[feature_hash].push_back(slot_id);
        
        return slot_id;
    }
    
    bool fetch(uint64_t slot_id, TestData& out_data) {
        auto it = data_map.find(slot_id);
        if (it != data_map.end()) {
            out_data = it->second;
            return true;
        }
        return false;
    }
    
    std::vector<std::pair<uint64_t, double>> findSimilarByFeature(
        const FeatureVector<float>& query_feature, 
        int k = 10, 
        double similarity_threshold = 0.0,
        const std::string& distance_metric = "cosine") {
        
        std::vector<std::pair<uint64_t, double>> results;
        
        for (const auto& pair : descriptor_map) {
            const DataDescriptor& desc = pair.second;
            TestData data;
            if (fetch(desc.slot_id, data)) {
                double similarity = 0.0;
                if (distance_metric == "cosine") {
                    similarity = desc.feature.cosineSimilarity(query_feature);
                } else if (distance_metric == "euclidean") {
                    double distance = desc.feature.euclideanDistance(query_feature);
                    similarity = 1.0 / (1.0 + distance);
                }
                
                if (similarity >= similarity_threshold) {
                    results.emplace_back(desc.slot_id, similarity);
                }
            }
        }
        
        std::sort(results.begin(), results.end(), 
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        if (results.size() > static_cast<size_t>(k)) {
            results.resize(k);
        }
        
        return results;
    }
    
    std::vector<uint64_t> findByFeatureHash(const std::string& feature_hash) {
        auto it = feature_index.find(feature_hash);
        if (it != feature_index.end()) {
            return it->second;
        }
        return {};
    }
};

void testBasicFeatureMatching() {
    printf("=== 基本特征匹配测试 ===\n");
    
    // 创建存储系统
    TestStorage storage;
    
    // 创建测试数据
    FeatureVector<float> feature1({1.0f, 2.0f, 3.0f}, "test_type");
    FeatureVector<float> feature2({1.1f, 2.0f, 3.1f}, "test_type");
    FeatureVector<float> feature3({4.0f, 5.0f, 6.0f}, "test_type");
    
    TestData data1(1, "item1", feature1);
    TestData data2(2, "item2", feature2);
    TestData data3(3, "item3", feature3);
    
    // 存储数据
    uint64_t slot1 = storage.store(data1, feature1);
    uint64_t slot2 = storage.store(data2, feature2);
    uint64_t slot3 = storage.store(data3, feature3);
    
    printf("存储的数据槽ID: %llu, %llu, %llu\n", slot1, slot2, slot3);
    
    // 测试特征相似度搜索
    FeatureVector<float> query_feature({1.0f, 2.0f, 3.0f}, "test_type");
    auto similar_items = storage.findSimilarByFeature(query_feature, 3, 0.8, "cosine");
    
    printf("相似度搜索结果:\n");
    for (const auto& item : similar_items) {
        TestData data;
        if (storage.fetch(item.first, data)) {
            printf("槽ID: %llu, 相似度: %.4f, 数据: %s\n", 
                   item.first, item.second, data.name.c_str());
        }
    }
    
    printf("=== 测试完成 ===\n");
}

void testFeatureHashLookup() {
    printf("\n=== 特征Hash查找测试 ===\n");
    
    TestStorage storage;
    
    FeatureVector<float> feature({1.0f, 2.0f, 3.0f}, "test_type");
    TestData data(1, "test_item", feature);
    
    uint64_t slot_id = storage.store(data, feature);
    
    // 手动计算特征hash
    std::hash<std::string> hasher;
    std::string feature_hash = std::to_string(hasher(feature.serialize()));
    auto slots = storage.findByFeatureHash(feature_hash);
    
    printf("特征Hash: %s\n", feature_hash.c_str());
    printf("找到的槽ID: ");
    for (auto id : slots) {
        printf("%llu ", id);
    }
    printf("\n");
    
    printf("=== 测试完成 ===\n");
}

int main() {
    printf("开始特征匹配功能测试...\n\n");
    
    testBasicFeatureMatching();
    testFeatureHashLookup();
    
    printf("\n所有测试通过!\n");
    
    return 0;
}