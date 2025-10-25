// 预定义知识库加载器 - 加载本地JSON知识库
#include "rag_knowledge_loader.h"
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// 加载预定义的知识库
std::vector<KnowledgeEntry> PredefinedKnowledgeLoader::loadComputerScienceKnowledge() {
    std::vector<KnowledgeEntry> entries;
    
    // 计算机科学基础知识
    entries.push_back({
        "数据结构与算法",
        "数据结构是计算机存储、组织数据的方式，包括数组、链表、树、图等。算法是解决问题的步骤和方法。",
        "计算机科学",
        "预定义知识库"
    });
    
    entries.push_back({
        "操作系统原理", 
        "操作系统是管理计算机硬件与软件资源的系统软件，负责进程管理、内存管理、文件系统等。",
        "计算机科学",
        "预定义知识库"
    });
    
    return entries;
}

std::vector<KnowledgeEntry> PredefinedKnowledgeLoader::loadMachineLearningKnowledge() {
    std::vector<KnowledgeEntry> entries;
    
    // 机器学习知识
    entries.push_back({
        "监督学习",
        "监督学习使用带标签的训练数据来训练模型，包括分类和回归任务。",
        "机器学习", 
        "预定义知识库"
    });
    
    entries.push_back({
        "神经网络",
        "神经网络是模仿生物神经网络结构和功能的数学模型，由输入层、隐藏层和输出层组成。",
        "机器学习",
        "预定义知识库"
    });
    
    entries.push_back({
        "深度学习",
        "深度学习是机器学习的一个分支，使用深层神经网络来学习数据的层次化表示。",
        "机器学习",
        "预定义知识库"
    });
    
    return entries;
}

std::vector<KnowledgeEntry> PredefinedKnowledgeLoader::loadCommonSenseKnowledge() {
    std::vector<KnowledgeEntry> entries;
    
    // 常识知识
    entries.push_back({
        "基本逻辑",
        "如果A等于B，B等于C，那么A等于C。这是逻辑推理的基本原理。",
        "常识",
        "预定义知识库"
    });
    
    return entries;
}

// 从JSON文件加载知识库
bool RAGKnowledgeBaseLoader::loadFromFile(const std::string& file_path, const std::string& category) {
    try {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << file_path << std::endl;
            return false;
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string json_data = buffer.str();
        
        return loadFromJSON(json_data, category);
        
    } catch (const std::exception& e) {
        std::cerr << "加载文件错误: " << e.what() << std::endl;
        return false;
    }
}

// 解析JSON数据
bool RAGKnowledgeBaseLoader::parseJSONData(const std::string& json_data, std::vector<KnowledgeEntry>& entries) {
    try {
        json j = json::parse(json_data);
        
        if (j.contains("knowledge_base") && j["knowledge_base"].is_array()) {
            for (const auto& item : j["knowledge_base"]) {
                KnowledgeEntry entry;
                
                if (item.contains("title")) entry.title = item["title"];
                if (item.contains("content")) entry.content = item["content"];
                if (item.contains("category")) entry.category = item["category"];
                if (item.contains("source")) entry.source = item["source"];
                if (item.contains("relevance_score")) entry.relevance_score = item["relevance_score"];
                
                if (item.contains("tags") && item["tags"].is_array()) {
                    for (const auto& tag : item["tags"]) {
                        entry.tags.push_back(tag);
                    }
                }
                
                entries.push_back(entry);
            }
            return true;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "JSON解析错误: " << e.what() << std::endl;
    }
    
    return false;
}

// 示例用法
void examplePredefinedKnowledge() {
    std::cout << "=== 预定义知识库示例 ===" << std::endl;
    
    RAGKnowledgeBaseLoader loader;
    
    // 加载预定义的机器学习知识
    auto ml_knowledge = PredefinedKnowledgeLoader::loadMachineLearningKnowledge();
    for (const auto& entry : ml_knowledge) {
        loader.addKnowledgeEntry(entry);
        std::cout << "✓ 添加: " << entry.title << std::endl;
    }
    
    // 从JSON文件加载
    if (loader.loadFromFile("predefined_knowledge.json", "技术知识")) {
        std::cout << "✓ 从JSON文件加载成功" << std::endl;
    }
    
    // 生成Logic树
    auto logics = loader.generateLogicTreeFromCategory("机器学习", 5, 0.6);
    std::cout << "生成 " << logics.size() << " 个机器学习Logic" << std::endl;
}