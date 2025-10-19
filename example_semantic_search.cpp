//
// 语义搜索功能示例
// 演示如何使用ISW存储系统的语义查询功能
//

#include "isw.hpp"
#include <iostream>
#include <string>
#include <vector>

// 示例数据结构，包含文本内容
struct DocumentData {
    int id;
    std::string title;
    std::string content;
    std::string category;
    
    DocumentData(int id, const std::string& title, const std::string& content, const std::string& category)
        : id(id), title(title), content(content), category(category) {}
    
    // 必须提供hash函数
    std::string hash() const {
        return std::to_string(id) + "_" + title;
    }
    
    // 必须提供getText函数用于语义查询
    std::string getText() const {
        return title + ". " + content;
    }
    
    // 可选的getFeature函数(如果已有特征向量)
    FeatureVector<float> getFeature() const {
        // 如果没有预计算特征，返回空特征
        return FeatureVector<float>();
    }
};

void demoBasicSemanticSearch() {
    std::cout << "=== 基础语义搜索演示 ===" << std::endl;
    
    // 创建存储系统
    ExternalStorage<DocumentData> storage(100);
    
    // 创建一些示例文档
    std::vector<DocumentData> documents = {
        DocumentData(1, "人工智能技术", 
                    "人工智能是一门研究如何使计算机能够像人一样思考、学习和决策的科学。", "科技"),
        DocumentData(2, "机器学习算法", 
                    "机器学习是人工智能的重要分支，专注于开发能够从数据中学习的算法。", "科技"),
        DocumentData(3, "深度学习网络", 
                    "深度学习利用神经网络模型来处理复杂的模式识别和预测任务。", "科技"),
        DocumentData(4, "自然语言处理", 
                    "自然语言处理技术使计算机能够理解和生成人类语言。", "科技"),
        DocumentData(5, "计算机视觉", 
                    "计算机视觉研究如何让计算机从图像和视频中提取信息。", "科技")
    };
    
    // 存储文档(自动计算语义特征)
    std::cout << "存储文档..." << std::endl;
    for (const auto& doc : documents) {
        uint64_t slot_id = storage.storeWithSemanticFeature(doc);
        std::cout << "存储文档: " << doc.title << " -> 槽ID: " << slot_id << std::endl;
    }
    
    // 进行语义搜索
    std::cout << "\n=== 语义搜索测试 ===" << std::endl;
    
    // 搜索"AI技术"
    std::string query = "AI技术";
    auto results = storage.semanticSearch(query, 3, 0.0);
    
    std::cout << "查询: '" << query << "'" << std::endl;
    std::cout << "搜索结果:" << std::endl;
    for (const auto& result : results) {
        DocumentData doc;
        if (storage.fetch(result.first, doc)) {
            std::cout << "  相似度: " << result.second 
                      << ", 文档: " << doc.title 
                      << " (ID: " << doc.id << ")" << std::endl;
        }
    }
    
    // 搜索"神经网络"
    query = "神经网络";
    results = storage.semanticSearch(query, 3, 0.0);
    
    std::cout << "\n查询: '" << query << "'" << std::endl;
    std::cout << "搜索结果:" << std::endl;
    for (const auto& result : results) {
        DocumentData doc;
        if (storage.fetch(result.first, doc)) {
            std::cout << "  相似度: " << result.second 
                      << ", 文档: " << doc.title 
                      << " (ID: " << doc.id << ")" << std::endl;
        }
    }
    
    std::cout << "=== 演示完成 ===" << std::endl;
}

void demoHybridSearch() {
    std::cout << "\n=== 混合搜索演示 ===" << std::endl;
    
    ExternalStorage<DocumentData> storage(100);
    
    // 创建带有预定义特征向量的文档
    std::vector<std::pair<DocumentData, FeatureVector<float>>> docs_with_features = {
        {DocumentData(1, "向量数据库", "向量数据库专门用于存储和检索高维向量数据。", "数据库"),
         FeatureVector<float>({0.8f, 0.6f, 0.9f, 0.7f}, "vector_db")},
        {DocumentData(2, "图数据库", "图数据库用于存储节点和关系的网络结构。", "数据库"),
         FeatureVector<float>({0.7f, 0.9f, 0.5f, 0.8f}, "graph_db")},
        {DocumentData(3, "关系数据库", "关系数据库使用表格结构存储数据。", "数据库"),
         FeatureVector<float>({0.6f, 0.5f, 0.7f, 0.6f}, "relational_db")}
    };
    
    // 存储文档和特征
    std::cout << "存储文档和特征..." << std::endl;
    for (const auto& doc_feature : docs_with_features) {
        uint64_t slot_id = storage.store(doc_feature.first, doc_feature.second);
        std::cout << "存储: " << doc_feature.first.title << " -> 槽ID: " << slot_id << std::endl;
    }
    
    // 基于特征向量进行相似度搜索
    std::cout << "\n=== 特征相似度搜索 ===" << std::endl;
    FeatureVector<float> query_feature({0.8f, 0.7f, 0.8f, 0.7f}, "vector_db");
    auto feature_results = storage.findSimilarByFeature(query_feature, 3, 0.0, "cosine");
    
    std::cout << "特征查询结果:" << std::endl;
    for (const auto& result : feature_results) {
        DocumentData doc;
        if (storage.fetch(result.first, doc)) {
            std::cout << "  相似度: " << result.second 
                      << ", 文档: " << doc.title << std::endl;
        }
    }
    
    std::cout << "=== 混合搜索演示完成 ===" << std::endl;
}

void demoAdvancedFeatures() {
    std::cout << "\n=== 高级功能演示 ===" << std::endl;
    
    ExternalStorage<DocumentData> storage(100);
    
    // 批量存储
    std::vector<DocumentData> batch_docs = {
        DocumentData(10, "数据科学", "数据科学结合统计学和计算机科学来分析数据。", "数据科学"),
        DocumentData(11, "大数据技术", "大数据技术处理海量数据集。", "数据科学"),
        DocumentData(12, "数据分析", "数据分析是从数据中提取洞察的过程。", "数据科学")
    };
    
    auto slot_ids = storage.batchStoreWithSemanticFeatures(batch_docs);
    std::cout << "批量存储的槽ID: ";
    for (auto id : slot_ids) {
        std::cout << id << " ";
    }
    std::cout << std::endl;
    
    // 获取特征统计
    auto stats = storage.getFeatureStatistics();
    std::cout << "特征统计:" << std::endl;
    std::cout << "  总特征数: " << stats.total_features << std::endl;
    std::cout << "  独特特征类型数: " << stats.unique_feature_types << std::endl;
    
    for (const auto& type_count : stats.feature_type_counts) {
        std::cout << "  类型 '" << type_count.first << "': " << type_count.second << " 个" << std::endl;
    }
    
    // 获取所有特征类型
    auto feature_types = storage.getAllFeatureTypes();
    std::cout << "所有特征类型: ";
    for (const auto& type : feature_types) {
        std::cout << type << " ";
    }
    std::cout << std::endl;
    
    std::cout << "=== 高级功能演示完成 ===" << std::endl;
}

int main() {
    std::cout << "开始语义搜索功能演示...\n" << std::endl;
    
    try {
        demoBasicSemanticSearch();
        demoHybridSearch();
        demoAdvancedFeatures();
        
        std::cout << "\n所有演示完成!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "演示失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}