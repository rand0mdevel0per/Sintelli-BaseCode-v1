// Logic语义匹配和注入示例
// 演示如何在NeuronModel中注册Logic并通过语义匹配自动调用

#include <iostream>
#include <functional>
#include "NeuronModel.cu"

// 示例Logic回调函数
void logicCallback1(const std::string& logic_id, NeuronInput& input) {
    std::cout << "[Logic1] 执行了图像处理逻辑: " << logic_id << std::endl;
    // 这里可以注入图像处理相关的神经元活动
    // 设置NeuronInput参数
    input.activity = 1.0;
    input.weight = 1.0;
    // ... 其他初始化
}

void logicCallback2(const std::string& logic_id, NeuronInput& input) {
    std::cout << "[Logic2] 执行了文本分析逻辑: " << logic_id << std::endl;
    // 这里可以注入文本分析相关的神经元活动
    input.activity = 1.0;
    input.weight = 1.0;
    // ... 其他初始化
}

void logicCallback3(const std::string& logic_id, NeuronInput& input) {
    std::cout << "[Logic3] 执行了推理逻辑: " << logic_id << std::endl;
    // 这里可以注入推理相关的神经元活动
    input.activity = 1.0;
    input.weight = 1.0;
    // ... 其他初始化
}

void logicCallback4(const std::string& logic_id, NeuronInput& input) {
    std::cout << "[Logic4] 执行了记忆检索逻辑: " << logic_id << std::endl;
    // 这里可以注入记忆检索相关的神经元活动
    input.activity = 1.0;
    input.weight = 1.0;
    // ... 其他初始化
}

void exampleLogicRegistration() {
    std::cout << "=== Logic注册示例 ===" << std::endl;
    
    NeuronModel<8> model;
    
    // 注册不同类型的Logic
    bool success1 = model.registerLogic(
        "image_processor", 
        "处理和分析图像的神经网络逻辑",
        "图像处理",
        0.6, // 激活阈值
        logicCallback1
    );
    
    bool success2 = model.registerLogic(
        "text_analyzer",
        "分析和理解文本内容的逻辑",
        "文本分析", 
        0.5,
        logicCallback2
    );
    
    bool success3 = model.registerLogic(
        "reasoning_engine",
        "执行逻辑推理和问题解决的模块",
        "推理",
        0.7,
        logicCallback3
    );
    
    bool success4 = model.registerLogic(
        "memory_retrieval",
        "从长期记忆中检索相关信息的逻辑",
        "记忆",
        0.4,
        logicCallback4
    );
    
    std::cout << "Logic注册结果:" << std::endl;
    std::cout << "  图像处理Logic: " << (success1 ? "成功" : "失败") << std::endl;
    std::cout << "  文本分析Logic: " << (success2 ? "成功" : "失败") << std::endl;
    std::cout << "  推理Logic: " << (success3 ? "成功" : "失败") << std::endl;
    std::cout << "  记忆检索Logic: " << (success4 ? "成功" : "失败") << std::endl;
    
    // 获取统计信息
    auto stats = model.getLogicStats();
    std::cout << "\nLogic统计信息:" << std::endl;
    std::cout << "  总Logic数量: " << stats.total_logics << std::endl;
    std::cout << "  类别数量: " << stats.unique_categories << std::endl;
    for (const auto& cat_count : stats.category_counts) {
        std::cout << "  类别 '" << cat_count.first << "': " << cat_count.second << " 个Logic" << std::endl;
    }
}

void exampleLogicMatching() {
    std::cout << "\n=== Logic匹配示例 ===" << std::endl;
    
    NeuronModel<8> model;
    
    // 注册测试Logic
    model.registerLogic("img_proc", "图像处理和计算机视觉算法", "视觉", 0.5, logicCallback1);
    model.registerLogic("nlp_engine", "自然语言处理和文本理解", "语言", 0.5, logicCallback2);
    model.registerLogic("reasoning", "逻辑推理和问题解决", "推理", 0.6, logicCallback3);
    model.registerLogic("memory", "记忆存储和检索系统", "记忆", 0.4, logicCallback4);
    
    // 测试查询
    std::vector<std::string> test_queries = {
        "我需要处理一张图片",
        "分析这段文本的内容",
        "解决这个逻辑问题",
        "回忆之前学过的知识",
        "识别图像中的物体"
    };
    
    for (const auto& query : test_queries) {
        std::cout << "\n查询: '" << query << "'" << std::endl;
        
        auto matching_logics = model.findMatchingLogics(query, 3, 0.3);
        
        if (matching_logics.empty()) {
            std::cout << "  没有找到匹配的Logic" << std::endl;
        } else {
            std::cout << "  匹配的Logic:" << std::endl;
            for (const auto& [logic, similarity] : matching_logics) {
                std::cout << "    - " << logic.logic_id << " (相似度: " << similarity 
                          << ", 类别: " << logic.category 
                          << ", 阈值: " << logic.activation_threshold << ")" << std::endl;
            }
        }
    }
}

void exampleLogicInjection() {
    std::cout << "\n=== Logic注入示例 ===" << std::endl;
    
    NeuronModel<8> model;
    
    // 注册带回调的Logic
    model.registerLogic("vision", "处理视觉信息和图像识别", "视觉", 0.4, [](const std::string& logic_id, NeuronInput& input) {
        std::cout << "[注入] 激活视觉处理神经元: " << logic_id << std::endl;
        input.activity = 1.0;
        input.weight = 1.0;
    });
    
    model.registerLogic("language", "理解和生成自然语言", "语言", 0.4, [](const std::string& logic_id, NeuronInput& input) {
        std::cout << "[注入] 激活语言处理神经元: " << logic_id << std::endl;
        input.activity = 1.0;
        input.weight = 1.0;
    });
    
    model.registerLogic("reason", "执行逻辑推理和决策", "推理", 0.5, [](const std::string& logic_id, NeuronInput& input) {
        std::cout << "[注入] 激活推理神经元: " << logic_id << std::endl;
        input.activity = 1.0;
        input.weight = 1.0;
    });
    
    // 模拟用户输入并自动注入Logic
    std::vector<std::string> user_inputs = {
        "看看这张图片里有什么",
        "帮我分析这段话的意思",
        "解决这个数学问题",
        "回忆昨天学的内容"
    };
    
    for (const auto& input : user_inputs) {
        std::cout << "\n用户输入: '" << input << "'" << std::endl;
        
        auto activated = model.injectMatchingLogics(input, 2, 0.3);
        
        if (!activated.empty()) {
            std::cout << "  激活的Logic: ";
            for (const auto& logic_id : activated) {
                std::cout << logic_id << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "  没有激活任何Logic" << std::endl;
        }
    }
}

void exampleRealTimeLogicInjection() {
    std::cout << "\n=== 实时Logic注入示例 ===" << std::endl;
    
    NeuronModel<10> model;
    
    // 注册实时处理Logic
    model.registerLogic("real_time_vision", "实时图像处理和物体检测", "实时", 0.3, [](const std::string& logic_id, NeuronInput& input) {
        std::cout << "[实时] 注入视觉处理模式: " << logic_id << std::endl;
        // 这里可以设置神经元的实时处理模式
        input.activity = 1.0;
        input.weight = 1.0;
    });
    
    model.registerLogic("attention_boost", "增强注意力和专注力", "认知", 0.4, [](const std::string& logic_id, NeuronInput& input) {
        std::cout << "[实时] 注入注意力增强模式: " << logic_id << std::endl;
        // 这里可以调整神经元的注意力参数
        input.activity = 1.0;
        input.weight = 1.0;
    });
    
    model.registerLogic("memory_consolidation", "巩固和强化记忆", "记忆", 0.5, [](const std::string& logic_id, NeuronInput& input) {
        std::cout << "[实时] 注入记忆巩固模式: " << logic_id << std::endl;
        // 这里可以触发记忆强化过程
        input.activity = 1.0;
        input.weight = 1.0;
    });
    
    // 模拟实时输入流
    std::vector<std::string> real_time_inputs = {
        "注意这个区域",
        "记住这个信息",
        "处理这个视觉输入",
        "强化学习这个模式",
        "专注在这个任务上"
    };
    
    for (const auto& input : real_time_inputs) {
        std::cout << "\n实时输入: '" << input << "'" << std::endl;
        
        // 使用processInputWithLogicInjection自动处理
        bool injected = model.processInputWithLogicInjection(input, 2, 0.3);
        
        if (injected) {
            std::cout << "  Logic注入成功" << std::endl;
        } else {
            std::cout << "  没有合适的Logic可注入" << std::endl;
        }
    }
}

void exampleCategorySpecificMatching() {
    std::cout << "\n=== 类别特定匹配示例 ===" << std::endl;
    
    NeuronModel<8> model;
    
    // 注册不同类别的Logic
    model.registerLogic("math_solver", "解决数学问题和计算", "数学", 0.5, [](const std::string& logic_id, NeuronInput& input) {
        std::cout << "[数学] 激活数学求解器: " << logic_id << std::endl;
        input.activity = 1.0;
        input.weight = 1.0;
    });
    
    model.registerLogic("code_generator", "生成和优化代码", "编程", 0.5, [](const std::string& logic_id, NeuronInput& input) {
        std::cout << "[编程] 激活代码生成器: " << logic_id << std::endl;
        input.activity = 1.0;
        input.weight = 1.0;
    });
    
    model.registerLogic("data_analyzer", "分析数据和统计信息", "数据分析", 0.4, [](const std::string& logic_id, NeuronInput& input) {
        std::cout << "[数据分析] 激活数据分析器: " << logic_id << std::endl;
        input.activity = 1.0;
        input.weight = 1.0;
    });
    
    // 在不同类别中搜索
    std::string query = "计算这个问题";
    
    std::cout << "查询: '" << query << "'" << std::endl;
    
    // 全局搜索
    auto global_results = model.findMatchingLogics(query, 5, 0.2);
    std::cout << "全局搜索结果:" << std::endl;
    for (const auto& [logic, similarity] : global_results) {
        std::cout << "  - " << logic.logic_id << " (相似度: " << similarity 
                  << ", 类别: " << logic.category << ")" << std::endl;
    }
    
    // 特定类别搜索
    auto math_results = model.findMatchingLogics(query, 3, 0.2, "数学");
    std::cout << "\n数学类别搜索结果:" << std::endl;
    for (const auto& [logic, similarity] : math_results) {
        std::cout << "  - " << logic.logic_id << " (相似度: " << similarity 
                  << ", 类别: " << logic.category << ")" << std::endl;
    }
}

int main() {
    std::cout << "Logic语义匹配和注入功能演示" << std::endl;
    std::cout << "============================" << std::endl;
    
    try {
        exampleLogicRegistration();
        exampleLogicMatching();
        exampleLogicInjection();
        exampleRealTimeLogicInjection();
        exampleCategorySpecificMatching();
        
        std::cout << "\n=== 所有示例执行完成 ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "示例执行过程中出现异常: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}