#ifndef EXTERNAL_STORAGE_API_H
#define EXTERNAL_STORAGE_API_H

#include "isw.hpp"
#include "rag_knowledge_loader.h"
#include <string>
#include <vector>
#include <memory>

// 外部存储API类模板
template<typename T>
class ExternalStorageAPI {
private:
    std::shared_ptr<ExternalStorage<T>> storage_;

public:
    // 构造函数，接受一个ExternalStorage实例
    explicit ExternalStorageAPI(std::shared_ptr<ExternalStorage<T>> storage) 
        : storage_(storage) {}

    // 将RAG知识库条目插入到外部存储中
    bool insertKnowledgeEntries(const std::vector<KnowledgeEntry>& entries) {
        if (!storage_) {
            return false;
        }

        bool success = true;
        for (const auto& entry : entries) {
            // 将KnowledgeEntry转换为T类型并存储
            T data;
            if (convertKnowledgeEntry(entry, data)) {
                // 存储数据到外部存储
                uint64_t slot_id = storage_->store(data);
                if (slot_id == 0) {
                    success = false;
                }
            } else {
                success = false;
            }
        }
        
        return success;
    }

    // 从RAG知识库加载器插入知识
    bool insertFromRAGLoader(RAGKnowledgeBaseLoader& loader) {
        if (!storage_) {
            return false;
        }

        // 获取所有知识条目
        // 注意：这里需要RAGKnowledgeBaseLoader提供一个方法来获取所有条目
        // 我们假设有一个getAllEntries方法
        auto entries = loader.getAllEntries();
        return insertKnowledgeEntries(entries);
    }

    // 通过类别插入知识
    bool insertKnowledgeByCategory(RAGKnowledgeBaseLoader& loader, const std::string& category) {
        if (!storage_) {
            return false;
        }

        // 获取特定类别的知识条目
        // 注意：这里需要RAGKnowledgeBaseLoader提供一个方法来按类别获取条目
        // 我们假设有一个getEntriesByCategory方法
        auto entries = loader.getEntriesByCategory(category);
        return insertKnowledgeEntries(entries);
    }

    // 插入单个知识条目
    uint64_t insertKnowledgeEntry(const T& data) {
        if (!storage_) {
            return 0;
        }

        return storage_->store(data);
    }

    // 从预定义知识库插入知识
    bool insertFromPredefinedLoader() {
        if (!storage_) {
            return false;
        }

        bool success = true;
        
        // 插入计算机科学知识
        auto cs_knowledge = PredefinedKnowledgeLoader::loadComputerScienceKnowledge();
        if (!insertKnowledgeEntries(cs_knowledge)) {
            success = false;
        }
        
        // 插入数学知识
        auto math_knowledge = PredefinedKnowledgeLoader::loadMathematicsKnowledge();
        if (!insertKnowledgeEntries(math_knowledge)) {
            success = false;
        }
        
        // 插入物理知识
        auto physics_knowledge = PredefinedKnowledgeLoader::loadPhysicsKnowledge();
        if (!insertKnowledgeEntries(physics_knowledge)) {
            success = false;
        }
        
        // 插入生物学知识
        auto biology_knowledge = PredefinedKnowledgeLoader::loadBiologyKnowledge();
        if (!insertKnowledgeEntries(biology_knowledge)) {
            success = false;
        }
        
        // 插入哲学知识
        auto philosophy_knowledge = PredefinedKnowledgeLoader::loadPhilosophyKnowledge();
        if (!insertKnowledgeEntries(philosophy_knowledge)) {
            success = false;
        }
        
        // 插入常识知识
        auto commonsense_knowledge = PredefinedKnowledgeLoader::loadCommonSenseKnowledge();
        if (!insertKnowledgeEntries(commonsense_knowledge)) {
            success = false;
        }
        
        // 插入编程文档
        auto programming_knowledge = PredefinedKnowledgeLoader::loadProgrammingDocumentation();
        if (!insertKnowledgeEntries(programming_knowledge)) {
            success = false;
        }
        
        // 插入机器学习知识
        auto ml_knowledge = PredefinedKnowledgeLoader::loadMachineLearningKnowledge();
        if (!insertKnowledgeEntries(ml_knowledge)) {
            success = false;
        }
        
        // 插入神经科学知识
        auto neuroscience_knowledge = PredefinedKnowledgeLoader::loadNeuroscienceKnowledge();
        if (!insertKnowledgeEntries(neuroscience_knowledge)) {
            success = false;
        }
        
        // 插入语言模型知识
        auto lm_knowledge = PredefinedKnowledgeLoader::loadLanguageModelKnowledge();
        if (!insertKnowledgeEntries(lm_knowledge)) {
            success = false;
        }
        
        // 插入AI技术知识
        auto ai_knowledge = PredefinedKnowledgeLoader::loadAITechniques();
        if (!insertKnowledgeEntries(ai_knowledge)) {
            success = false;
        }
        
        return success;
    }

    // 获取存储实例的引用
    std::shared_ptr<ExternalStorage<T>> getStorage() {
        return storage_;
    }

private:
    // 将KnowledgeEntry转换为T类型
    // 这个方法需要根据T的具体类型来实现
    bool convertKnowledgeEntry(const KnowledgeEntry& entry, T& data) {
        // 这是一个通用的实现，需要根据T的具体类型进行特化
        // 对于不同的T类型，可能需要不同的转换逻辑
        
        // 如果T是KnowledgeEntry类型，直接赋值
        if constexpr (std::is_same_v<T, KnowledgeEntry>) {
            data = entry;
            return true;
        }
        
        // 如果T有相应的方法来从KnowledgeEntry构造，可以调用这些方法
        // 这里只是一个示例，实际实现需要根据T的具体类型来定制
        
        // 默认返回false，表示转换失败
        return false;
    }
};

// 特化版本：针对KnowledgeEntry类型的ExternalStorage
template<>
bool ExternalStorageAPI<KnowledgeEntry>::convertKnowledgeEntry(
    const KnowledgeEntry& entry, KnowledgeEntry& data) {
    data = entry;
    return true;
}

// 特化版本：针对其他可能的类型，比如字符串类型
template<>
bool ExternalStorageAPI<std::string>::convertKnowledgeEntry(
    const KnowledgeEntry& entry, std::string& data) {
    // 将KnowledgeEntry转换为JSON格式的字符串
    data = "{\n";
    data += "  \"title\": \"" + entry.title + "\",\n";
    data += "  \"content\": \"" + entry.content + "\",\n";
    data += "  \"category\": \"" + entry.category + "\",\n";
    data += "  \"source\": \"" + entry.source + "\",\n";
    data += "  \"relevance_score\": " + std::to_string(entry.relevance_score) + "\n";
    data += "}";
    return true;
}

#endif // EXTERNAL_STORAGE_API_H