// 语义查询接口封装
// 提供缓存、上下文感知、频率限制等高级功能

#ifndef SEMANTIC_QUERY_INTERFACE_H
#define SEMANTIC_QUERY_INTERFACE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <chrono>
#include <functional>
#include <mutex>
#include "semantic_matcher.h"

// 缓存条目结构
struct CachedQueryResult {
    std::string query_text;
    FeatureVector<float> query_feature;
    std::vector<std::pair<LogicDescriptor, double>> results;
    std::chrono::steady_clock::time_point timestamp;
    uint32_t hit_count = 0;
    double confidence = 0.0; // 查询置信度
};

// 查询优先级结构
struct PriorityQuery {
    std::string query;
    double confidence;        // 基于神经元激活度的置信度
    std::chrono::steady_clock::time_point timestamp;
    
    bool operator<(const PriorityQuery& other) const {
        // 高置信度优先
        return confidence < other.confidence;
    }
};

// 语义查询缓存类
class SemanticQueryCache {
private:
    std::unordered_map<std::string, CachedQueryResult> cache;
    size_t max_cache_size;
    std::chrono::seconds cache_ttl;
    mutable std::mutex cache_mutex;
    
public:
    SemanticQueryCache(size_t max_size = 1000, std::chrono::seconds ttl = std::chrono::seconds(300))
        : max_cache_size(max_size), cache_ttl(ttl) {}
    
    // 尝试从缓存获取结果
    bool tryGetFromCache(const std::string& query, 
                        std::vector<std::pair<LogicDescriptor, double>>& results,
                        double& confidence) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        
        auto hash = std::to_string(std::hash<std::string>{}(query));
        auto it = cache.find(hash);
        
        if (it != cache.end()) {
            auto now = std::chrono::steady_clock::now();
            if (now - it->second.timestamp < cache_ttl) {
                it->second.hit_count++;
                results = it->second.results;
                confidence = it->second.confidence;
                return true;
            } else {
                cache.erase(it);
            }
        }
        return false;
    }
    
    // 添加到缓存
    void addToCache(const std::string& query, 
                   const FeatureVector<float>& feature,
                   const std::vector<std::pair<LogicDescriptor, double>>& results,
                   double confidence = 0.0) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        
        auto hash = std::to_string(std::hash<std::string>{}(query));
        
        // 缓存淘汰策略
        if (cache.size() >= max_cache_size) {
            auto oldest = cache.begin();
            for (auto it = cache.begin(); it != cache.end(); ++it) {
                if (it->second.hit_count < oldest->second.hit_count) {
                    oldest = it;
                }
            }
            cache.erase(oldest);
        }
        
        cache[hash] = {query, feature, results, std::chrono::steady_clock::now(), 0, confidence};
    }
    
    // 获取缓存统计
    struct CacheStats {
        size_t total_entries;
        size_t total_hits;
        double hit_rate;
    };
    
    CacheStats getStats() const {
        std::lock_guard<std::mutex> lock(cache_mutex);
        
        CacheStats stats{};
        stats.total_entries = cache.size();
        
        size_t total_hits = 0;
        for (const auto& entry : cache) {
            total_hits += entry.second.hit_count;
        }
        stats.total_hits = total_hits;
        stats.hit_rate = cache.empty() ? 0.0 : static_cast<double>(total_hits) / cache.size();
        
        return stats;
    }
    
    // 清除缓存
    void clearCache() {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache.clear();
    }
};

// 上下文感知查询处理器
class ContextAwareQueryProcessor {
private:
    SemanticQueryCache& cache;
    std::vector<std::string> recent_queries;
    size_t context_window;
    mutable std::mutex context_mutex;
    
public:
    ContextAwareQueryProcessor(SemanticQueryCache& query_cache, size_t window_size = 10)
        : cache(query_cache), context_window(window_size) {}
    
    // 带上下文的查询处理
    std::vector<std::pair<LogicDescriptor, double>> 
    processWithContext(const std::string& current_query,
                      LogicSemanticMatcher& matcher,
                      double neuron_confidence = 0.0,
                      double similarity_threshold = 0.3) {
        
        // 1. 先尝试缓存命中
        std::vector<std::pair<LogicDescriptor, double>> results;
        double cached_confidence;
        if (cache.tryGetFromCache(current_query, results, cached_confidence)) {
            std::cout << "🎯 缓存命中! 查询: \"" << current_query 
                      << "\" (置信度: " << cached_confidence << ")" << std::endl;
            return results;
        }
        
        // 2. 基于上下文的查询扩展
        std::string expanded_query = expandQueryWithContext(current_query);
        
        // 3. 执行语义查询
        results = matcher.findMatchingLogics(expanded_query, 5, similarity_threshold);
        
        // 4. 缓存结果
        auto feature = matcher.getFeatureExtractor()->extractTextFeature(current_query);
        cache.addToCache(current_query, feature, results, neuron_confidence);
        
        // 5. 更新上下文
        updateContext(current_query);
        
        std::cout << "🔍 新查询: \"" << current_query 
                  << "\" -> 找到 " << results.size() << " 个匹配Logic" << std::endl;
        
        return results;
    }
    
    // 获取上下文
    std::vector<std::string> getContext() const {
        std::lock_guard<std::mutex> lock(context_mutex);
        return recent_queries;
    }
    
    // 清除上下文
    void clearContext() {
        std::lock_guard<std::mutex> lock(context_mutex);
        recent_queries.clear();
    }
    
private:
    std::string expandQueryWithContext(const std::string& query) {
        std::lock_guard<std::mutex> lock(context_mutex);
        
        if (recent_queries.empty()) return query;
        
        // 合并最近查询作为上下文（取最后几个）
        std::string context_query = query;
        size_t start_idx = recent_queries.size() > 3 ? recent_queries.size() - 3 : 0;
        
        for (size_t i = start_idx; i < recent_queries.size(); ++i) {
            context_query += " " + recent_queries[i];
        }
        return context_query;
    }
    
    void updateContext(const std::string& query) {
        std::lock_guard<std::mutex> lock(context_mutex);
        
        recent_queries.push_back(query);
        if (recent_queries.size() > context_window) {
            recent_queries.erase(recent_queries.begin());
        }
    }
};

// 自适应采样控制器
class AdaptiveSamplingController {
private:
    double intensity_threshold;
    std::chrono::milliseconds base_interval;
    std::chrono::milliseconds current_interval;
    std::chrono::steady_clock::time_point last_sample_time;
    mutable std::mutex sampling_mutex;
    
public:
    AdaptiveSamplingController(double threshold = 0.7, 
                             std::chrono::milliseconds interval = std::chrono::milliseconds(100))
        : intensity_threshold(threshold), base_interval(interval), current_interval(interval) {}
    
    // 判断是否应该采样
    bool shouldSample(double neuron_activity, const std::string& current_content) {
        std::lock_guard<std::mutex> lock(sampling_mutex);
        
        auto now = std::chrono::steady_clock::now();
        
        // 检查时间间隔
        if (now - last_sample_time < current_interval) {
            return false;
        }
        
        // 高激活度 => 更频繁采样
        if (neuron_activity > intensity_threshold) {
            current_interval = std::chrono::milliseconds(50); // 50ms
        } else {
            current_interval = std::chrono::milliseconds(500); // 500ms
        }
        
        // 检查内容是否有意义
        bool meaningful = isMeaningfulContent(current_content);
        
        if (meaningful) {
            last_sample_time = now;
        }
        
        return meaningful;
    }
    
    // 获取当前采样间隔
    std::chrono::milliseconds getCurrentInterval() const {
        std::lock_guard<std::mutex> lock(sampling_mutex);
        return current_interval;
    }
    
    // 设置采样参数
    void setSamplingParameters(double threshold, std::chrono::milliseconds interval) {
        std::lock_guard<std::mutex> lock(sampling_mutex);
        intensity_threshold = threshold;
        base_interval = interval;
        current_interval = interval;
    }
    
    // 重置采样状态
    void reset() {
        std::lock_guard<std::mutex> lock(sampling_mutex);
        current_interval = base_interval;
        last_sample_time = std::chrono::steady_clock::time_point{};
    }
    
private:
    bool isMeaningfulContent(const std::string& content) {
        // 简单的有意义内容检测
        static std::string last_content;
        
        // 内容长度检查
        if (content.length() < 3) return false;
        
        // 内容变化检查
        if (content == last_content) return false;
        
        last_content = content;
        return true;
    }
};

// 查询优先级管理器
class QueryPriorityManager {
private:
    std::priority_queue<PriorityQuery> query_queue;
    size_t max_queue_size;
    mutable std::mutex queue_mutex;
    
public:
    QueryPriorityManager(size_t max_size = 100) : max_queue_size(max_size) {}
    
    // 添加查询到优先级队列
    void addQuery(const std::string& query, double confidence) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        
        PriorityQuery pq{query, confidence, std::chrono::steady_clock::now()};
        query_queue.push(pq);
        
        // 限制队列大小
        if (query_queue.size() > max_queue_size) {
            // 移除置信度最低的查询
            std::priority_queue<PriorityQuery> temp;
            while (query_queue.size() > max_queue_size - 1) {
                temp.push(query_queue.top());
                query_queue.pop();
            }
            query_queue = std::move(temp);
        }
    }
    
    // 获取最高优先级的查询
    bool getNextQuery(std::string& query, double& confidence) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        
        if (query_queue.empty()) return false;
        
        auto top_query = query_queue.top();
        query_queue.pop();
        
        query = top_query.query;
        confidence = top_query.confidence;
        return true;
    }
    
    // 获取队列大小
    size_t getQueueSize() const {
        std::lock_guard<std::mutex> lock(queue_mutex);
        return query_queue.size();
    }
    
    // 清空队列
    void clearQueue() {
        std::lock_guard<std::mutex> lock(queue_mutex);
        while (!query_queue.empty()) {
            query_queue.pop();
        }
    }
};

// 主查询接口类
class SemanticQueryInterface {
private:
    SemanticQueryCache cache;
    ContextAwareQueryProcessor context_processor;
    AdaptiveSamplingController sampling_controller;
    QueryPriorityManager priority_manager;
    LogicSemanticMatcher& logic_matcher;
    
public:
    SemanticQueryInterface(LogicSemanticMatcher& matcher,
                          size_t cache_size = 1000,
                          size_t context_window = 10,
                          double sampling_threshold = 0.7,
                          std::chrono::milliseconds sampling_interval = std::chrono::milliseconds(100))
        : cache(cache_size),
          context_processor(cache, context_window),
          sampling_controller(sampling_threshold, sampling_interval),
          logic_matcher(matcher) {}
    
    // 主查询接口
    std::vector<std::pair<LogicDescriptor, double>> 
    query(const std::string& query_text, 
          double neuron_confidence = 0.0,
          double similarity_threshold = 0.3,
          bool use_context = true,
          bool use_cache = true) {
        
        if (!use_cache) {
            // 直接查询
            return logic_matcher.findMatchingLogics(query_text, 5, similarity_threshold);
        }
        
        if (use_context) {
            return context_processor.processWithContext(query_text, logic_matcher, 
                                                       neuron_confidence, similarity_threshold);
        } else {
            // 仅使用缓存
            std::vector<std::pair<LogicDescriptor, double>> results;
            double confidence;
            if (cache.tryGetFromCache(query_text, results, confidence)) {
                return results;
            }
            
            // 缓存未命中，执行查询并缓存
            results = logic_matcher.findMatchingLogics(query_text, 5, similarity_threshold);
            auto feature = logic_matcher.getFeatureExtractor()->extractTextFeature(query_text);
            cache.addToCache(query_text, feature, results, neuron_confidence);
            return results;
        }
    }
    
    // 异步查询（添加到优先级队列）
    void asyncQuery(const std::string& query_text, double confidence = 0.0) {
        priority_manager.addQuery(query_text, confidence);
    }
    
    // 处理优先级队列中的查询
    void processPriorityQueries(double similarity_threshold = 0.3) {
        std::string query;
        double confidence;
        
        while (priority_manager.getNextQuery(query, confidence)) {
            auto results = this->query(query, confidence, similarity_threshold, true, true);
            
            // 处理结果（可以在这里添加回调或注入逻辑）
            if (!results.empty()) {
                std::cout << "🚀 处理优先级查询: \"" << query 
                          << "\" -> " << results.size() << " 个匹配" << std::endl;
            }
        }
    }
    
    // 智能采样查询（用于神经元输出）
    bool smartSampleQuery(double neuron_activity, 
                         const std::string& neuron_output,
                         double similarity_threshold = 0.3) {
        
        if (sampling_controller.shouldSample(neuron_activity, neuron_output)) {
            // 添加到优先级队列
            asyncQuery(neuron_output, neuron_activity);
            return true;
        }
        return false;
    }
    
    // 获取统计信息
    struct QueryStats {
        SemanticQueryCache::CacheStats cache_stats;
        size_t queue_size;
        std::chrono::milliseconds current_sampling_interval;
        std::vector<std::string> recent_context;
    };
    
    QueryStats getStats() const {
        QueryStats stats;
        stats.cache_stats = cache.getStats();
        stats.queue_size = priority_manager.getQueueSize();
        stats.current_sampling_interval = sampling_controller.getCurrentInterval();
        stats.recent_context = context_processor.getContext();
        return stats;
    }
    
    // 重置状态
    void reset() {
        cache.clearCache();
        context_processor.clearContext();
        sampling_controller.reset();
        priority_manager.clearQueue();
    }
    
    // 配置接口
    void configure(size_t new_cache_size, 
                  size_t new_context_window,
                  double new_sampling_threshold,
                  std::chrono::milliseconds new_sampling_interval) {
        cache = SemanticQueryCache(new_cache_size);
        context_processor = ContextAwareQueryProcessor(cache, new_context_window);
        sampling_controller.setSamplingParameters(new_sampling_threshold, new_sampling_interval);
    }
};

#endif // SEMANTIC_QUERY_INTERFACE_H