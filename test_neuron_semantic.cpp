// 测试NeuronModel中的语义匹配功能

#include <iostream>
#include <vector>
#include "NeuronModel.cu"

void testNeuronModelSemanticMatching() {
    std::cout << "=== 测试NeuronModel语义匹配功能 ===" << std::endl;
    
    // 创建神经元模型实例
    NeuronModel<10> model; // 使用10x10x10的网格
    
    std::cout << "神经元模型初始化完成" << std::endl;
    
    // 测试文本特征提取
    std::string test_text = "这是一个用于测试语义匹配的文本";
    auto feature = model.extractTextFeature(test_text);
    
    if (feature.dimension > 0) {
        std::cout << "文本特征提取成功，维度: " << feature.dimension << std::endl;
    } else {
        std::cout << "文本特征提取失败" << std::endl;
        return;
    }
    
    // 注册一些测试文本
    std::vector<std::string> programming_texts = {
        "C++是一种高效的编程语言",
        "Python适合数据科学和机器学习",
        "Java在企业级应用中很流行",
        "JavaScript用于网页开发",
        "Go语言适合并发编程"
    };
    
    std::vector<std::string> ai_texts = {
        "深度学习是人工智能的重要分支",
        "神经网络可以用于图像识别",
        "机器学习算法需要大量数据",
        "自然语言处理是AI的热门领域",
        "计算机视觉技术发展迅速"
    };
    
    // 注册编程相关文本
    auto prog_ids = model.batchRegisterTexts(programming_texts, "编程");
    std::cout << "注册编程文本数量: " << prog_ids.size() << std::endl;
    
    // 注册AI相关文本
    auto ai_ids = model.batchRegisterTexts(ai_texts, "AI");
    std::cout << "注册AI文本数量: " << ai_ids.size() << std::endl;
    
    // 测试语义搜索
    std::string query1 = "编程语言的特点";
    auto results1 = model.semanticSearch(query1, 3, 0.3);
    
    std::cout << "\n查询 '" << query1 << "' 的结果:" << std::endl;
    for (const auto& match : results1) {
        std::cout << "  - " << match.text << " (相似度: " << match.similarity 
                  << ", 类别: " << match.category << ")" << std::endl;
    }
    
    // 测试按类别搜索
    std::string query2 = "机器学习算法";
    auto results2 = model.semanticSearchByCategory(query2, "AI", 2, 0.3);
    
    std::cout << "\n查询 '" << query2 << "' (仅在AI类别) 的结果:" << std::endl;
    for (const auto& match : results2) {
        std::cout << "  - " << match.text << " (相似度: " << match.similarity 
                  << ", 类别: " << match.category << ")" << std::endl;
    }
    
    // 测试相似度计算
    std::string text1 = "人工智能";
    std::string text2 = "机器学习";
    std::string text3 = "网页开发";
    
    double sim12 = model.calculateTextSimilarity(text1, text2);
    double sim13 = model.calculateTextSimilarity(text1, text3);
    
    std::cout << "\n文本相似度计算:" << std::endl;
    std::cout << "  '" << text1 << "' 与 '" << text2 << "' 相似度: " << sim12 << std::endl;
    std::cout << "  '" << text1 << "' 与 '" << text3 << "' 相似度: " << sim13 << std::endl;
    
    // 获取统计信息
    auto stats = model.getSemanticMatchingStats();
    std::cout << "\n语义匹配统计信息:" << std::endl;
    std::cout << "  总文本数: " << stats.total_texts << std::endl;
    std::cout << "  唯一类别数: " << stats.unique_categories << std::endl;
    for (const auto& cat_count : stats.category_counts) {
        std::cout << "  类别 '" << cat_count.first << "': " << cat_count.second << " 个文本" << std::endl;
    }
    
    // 测试集成语义匹配器
    std::cout << "\n=== 测试集成语义匹配器 ===" << std::endl;
    
    std::string new_text = "深度学习神经网络技术";
    uint64_t slot_id = model.registerTextForMatching(new_text, "AI");
    std::cout << "注册新文本，slot_id: " << slot_id << std::endl;
    
    // 搜索并存储结果
    auto search_results = model.semanticSearch("神经网络深度学习", 5, 0.2);
    std::cout << "集成搜索结果数量: " << search_results.size() << std::endl;
    
    // 获取最热的匹配
    auto hottest = model.getHottestSemanticMatches(3);
    std::cout << "\n最热的语义匹配结果:" << std::endl;
    for (const auto& match : hottest) {
        std::cout << "  - " << match.text << " (相似度: " << match.similarity 
                  << ", 类别: " << match.category << ")" << std::endl;
    }
}

int main() {
    std::cout << "开始测试NeuronModel语义匹配功能..." << std::endl;
    
    try {
        testNeuronModelSemanticMatching();
        std::cout << "\n=== NeuronModel语义匹配测试完成 ===" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "测试过程中出现异常: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}