//
// Created by ASUS on 10/2/2025.
//

#ifndef SRC_SCT_H
#define SRC_SCT_H

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <queue>
#include "deviceQueue.cpp"

// ===== SCT节点定义 =====
struct SCT_Node {
    int turn_id;
    double embedding[128];      // V_text (语义向量)
    double timestamp;           // 时间戳
    double utility_score;       // U_score (效用评分)
    std::string raw_text;       // 原始文本

    ~SCT_Node() = default;

    // 默认的拷贝操作
    SCT_Node(const SCT_Node& other) = default;
    SCT_Node& operator=(const SCT_Node& other) = default;
    SCT_Node() = default;

    // 默认的移动操作
    SCT_Node(SCT_Node&& other) noexcept = default;
    SCT_Node& operator=(SCT_Node&& other) noexcept = default;

    // 计算三重评分
    double computeScore(const double query_vec[128],
                       double current_time,
                       double w_sim = 0.5,
                       double w_rec = 0.3,
                       double w_util = 0.2,
                       double lambda = 0.01) const {
        // 1. 语义相似度
        double sim = 0.0;
        double norm_q = 0.0, norm_t = 0.0;
        for (int i = 0; i < 128; i++) {
            sim += query_vec[i] * embedding[i];
            norm_q += query_vec[i] * query_vec[i];
            norm_t += embedding[i] * embedding[i];
        }
        sim /= (sqrt(norm_q) * sqrt(norm_t) + 1e-8);

        // 2. 时间衰减
        double delta_t = current_time - timestamp;
        double recency = exp(-lambda * delta_t);

        // 3. 效用评分(已存储)

        // 组合
        return w_sim * sim + w_rec * recency + w_util * utility_score;
    }
};

// ===== 语义对话树管理器 =====
class SemanticConversationTree {
private:
    std::vector<SCT_Node> nodes;
    int next_turn_id;
    int max_nodes;              // N_max硬限制
    double current_time;

    // 权重参数
    double w_sim, w_rec, w_util;
    double lambda;              // 时间衰减率
    double delta_u;             // 效用提升量
    double global_decay;        // 全局衰减率

public:
    SemanticConversationTree(int max_n = 10000){
        next_turn_id = 0;
        max_nodes = max_n;
        current_time = 0.0;
        w_sim = 0.5;
        w_rec = 0.3;
        w_util = 0.2;
        lambda = 0.01;
        delta_u = 0.1;
        global_decay = 0.99;
    }

    bool reset() {
        try {
            nodes.clear();
            next_turn_id = 1;
        } catch (...) {
            return false;
        }
        return true;
    }

    // 添加新对话回合
    void addTurn(const std::string& text, const double embedding[128]) {
        SCT_Node node{};
        node.turn_id = next_turn_id++;
        memcpy(node.embedding, embedding, 128 * sizeof(double));
        node.timestamp = current_time;
        node.utility_score = 1.0;  // 初始效用
        node.raw_text = text;

        nodes.push_back(node);

        // 检查是否超过硬限制
        if (nodes.size() > max_nodes) {
            pruneLowestScore();
        }
    }

    // 召回最相关的K个上下文
    std::vector<int> recallTopK(const double query_vec[128], int k) {
        struct ScoredNode {
            int index;
            double score;
            bool operator<(const ScoredNode& other) const {
                return score < other.score;  // 最大堆
            }
        };

        std::priority_queue<ScoredNode> heap;

        // 计算所有节点的得分
        for (size_t i = 0; i < nodes.size(); i++) {
            double score = nodes[i].computeScore(
                query_vec, current_time, w_sim, w_rec, w_util, lambda
            );
            heap.push({(int)i, score});
        }

        // 提取Top-K
        std::vector<int> top_k_indices;
        for (int i = 0; i < k && !heap.empty(); i++) {
            int idx = heap.top().index;
            top_k_indices.push_back(idx);
            heap.pop();

            // 效用提升!
            nodes[idx].utility_score += delta_u;
        }

        return top_k_indices;
    }

    // 获取召回的原始文本
    std::vector<std::string> getRecalledTexts(const std::vector<int>& indices) {
        std::vector<std::string> texts;
        for (int idx : indices) {
            if (idx >= 0 && idx < nodes.size()) {
                texts.push_back(nodes[idx].raw_text);
            }
        }
        return texts;
    }

    // 全局老化衰减
    void globalAging() {
        for (auto& node : nodes) {
            node.utility_score *= global_decay;
        }
    }

    // 清除最低分节点
    void pruneLowestScore() {
        if (nodes.empty()) return;

        // 找到得分最低的节点
        double dummy_query[128] = {0};  // 用于计算当前分数

        int worst_idx = 0;
        double worst_score = 1e9;

        for (size_t i = 0; i < nodes.size(); i++) {
            double score = nodes[i].computeScore(
                dummy_query, current_time, w_sim, w_rec, w_util, lambda
            );
            if (score < worst_score) {
                worst_score = score;
                worst_idx = i;
            }
        }

        // 移除
        nodes.erase(nodes.begin() + worst_idx);
    }

    // 时间推进(每个对话周期调用)
    void advanceTime(double dt = 1.0) {
        current_time += dt;
    }

    // 统计信息
    size_t size() const { return nodes.size(); }

    // 设置参数
    void setWeights(double ws, double wr, double wu) {
        w_sim = ws; w_rec = wr; w_util = wu;
    }
    void setDecayRate(double l) { lambda = l; }
    void setUtilityBoost(double du) { delta_u = du; }
    void setGlobalDecay(double gd) { global_decay = gd; }
};

#endif //SRC_SCT_H