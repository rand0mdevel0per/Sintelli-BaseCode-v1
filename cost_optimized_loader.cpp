// 成本优化的数据集加载器
#include "rag_knowledge_loader.h"
#include "lightweight_dataset_loader.cpp"
#include <vector>
#include <map>

using namespace std;

class CostOptimizedLoader {
private:
    LightweightDatasetLoader lightweight_loader;
    RAGKnowledgeBaseLoader rag_loader;
    
    // 成本配置
    struct CostConfig {
        bool use_cloud_storage = false;
        size_t max_local_storage_mb = 1024; // 1GB限制
        bool streaming_mode = true;
        vector<string> priority_categories = {"计算机科学", "机器学习", "数学"};
    } cost_config;
    
public:
    CostOptimizedLoader() : lightweight_loader("./optimized_cache", 512) {}
    
    // 设置成本优化配置
    void setCostConfig(const CostConfig& config) {
        cost_config = config;
        cout << "💰 成本优化配置已设置:" << endl;
        cout << "  - 云存储: " << (config.use_cloud_storage ? "启用" : "禁用") << endl;
        cout << "  - 本地存储限制: " << config.max_local_storage_mb << " MB" << endl;
        cout << "  - 流式模式: " << (config.streaming_mode ? "启用" : "禁用") << endl;
        cout << "  - 优先类别: ";
        for (const auto& cat : config.priority_categories) {
            cout << cat << " ";
        }
        cout << endl;
    }
    
    // 智能数据加载策略
    bool smartLoadDataset(const string& dataset_name, 
                         const string& category_filter = "") {
        cout << "🤖 智能加载数据集: " << dataset_name << endl;
        
        // 1. 检查本地缓存
        if (checkLocalCache(dataset_name)) {
            cout << "  ✅ 使用本地缓存" << endl;
            return loadFromCache(dataset_name);
        }
        
        // 2. 根据数据集大小选择策略
        auto dataset_info = getDatasetInfo(dataset_name);
        
        if (dataset_info.estimated_size_mb > cost_config.max_local_storage_mb) {
            cout << "  📦 数据集过大(" << dataset_info.estimated_size_mb 
                 << "MB)，使用流式加载" << endl;
            return loadStreaming(dataset_name, category_filter);
        } else {
            cout << "  💾 数据集适中，下载完整样本" << endl;
            return lightweight_loader.downloadHuggingFaceSample(
                dataset_name, "", 1000); // 限制1000个样本
        }
    }
    
    // 多数据集融合加载
    void loadMultipleDatasets(const vector<string>& datasets, 
                             const map<string, string>& category_filters = {}) {
        cout << "🔄 加载多个数据集..." << endl;
        
        size_t total_loaded = 0;
        
        for (const auto& dataset : datasets) {
            string filter = "";
            if (category_filters.count(dataset)) {
                filter = category_filters.at(dataset);
            }
            
            if (smartLoadDataset(dataset, filter)) {
                total_loaded++;
                
                // 显示进度
                auto stats = rag_loader.getKnowledgeStats();
                cout << "  当前知识库: " << stats.total_entries << " 个条目" << endl;
                
                // 内存检查
                if (stats.total_entries > 5000) {
                    cout << "  ⚠️  条目较多，考虑清理缓存" << endl;
                }
            }
        }
        
        cout << "✅ 成功加载 " << total_loaded << "/" << datasets.size() << " 个数据集" << endl;
    }
    
    // 成本分析报告
    void generateCostReport() {
        cout << "\n📊 成本优化报告" << endl;
        cout << "===============" << endl;
        
        auto stats = rag_loader.getKnowledgeStats();
        
        // 计算存储成本（假设 $0.023/GB/月）
        double storage_cost = (cost_config.max_local_storage_mb / 1024.0) * 0.023;
        
        // 数据传输成本（假设 $0.09/GB）
        double transfer_cost = stats.total_entries * 0.0001; // 简化估算
        
        cout << "💾 存储成本: $" << storage_cost << "/月 (" 
             << cost_config.max_local_storage_mb << " MB)" << endl;
        cout << "📡 传输成本: $" << transfer_cost << " (估算)" << endl;
        cout << "📚 知识库大小: " << stats.total_entries << " 个条目" << endl;
        cout << "🏷️  覆盖类别: " << stats.unique_categories << " 个" << endl;
        
        // 成本节省建议
        cout << "\n💡 成本节省建议:" << endl;
        if (stats.total_entries < 1000) {
            cout << "  ✅ 当前配置成本较低，可以继续使用" << endl;
        } else {
            cout << "  ⚠️  考虑启用云存储和流式处理" << endl;
        }
    }
    
    // 获取RAG加载器
    RAGKnowledgeBaseLoader& getRAGLoader() {
        return rag_loader;
    }
    
private:
    struct DatasetInfo {
        string name;
        size_t estimated_size_mb;
        vector<string> available_categories;
    };
    
    // 获取数据集信息
    DatasetInfo getDatasetInfo(const string& dataset_name) {
        // 简化的数据集信息
        static map<string, DatasetInfo> dataset_info = {
            {"HuggingFaceFW/fineweb-edu-100b-shuffle", 
                {"FineWeb-Edu", 500, {"教育", "学术", "技术"}}},
            {"ethanolivertroy/nist-cybersecurity-training", 
                {"NIST安全", 1000, {"安全", "标准", "技术"}}},
            {"nick007x/github-code-2025", 
                {"GitHub代码", 800, {"编程", "代码", "计算机科学"}}}
        };
        
        if (dataset_info.count(dataset_name)) {
            return dataset_info[dataset_name];
        }
        
        // 默认值
        return {dataset_name, 100, {"通用"}};
    }
    
    bool checkLocalCache(const string& dataset_name) {
        // 检查本地缓存文件是否存在
        string cache_file = "./optimized_cache/" + dataset_name + ".cache";
        ifstream file(cache_file);
        return file.good();
    }
    
    bool loadFromCache(const string& dataset_name) {
        string cache_file = "./optimized_cache/" + dataset_name + ".cache";
        return rag_loader.loadFromFile(cache_file, "缓存数据");
    }
    
    bool loadStreaming(const string& dataset_name, const string& category_filter) {
        // 模拟流式加载
        cout << "  📡 模拟流式加载 " << dataset_name << endl;
        
        // 这里可以集成实际的流式加载逻辑
        // 使用datasets库的streaming=True模式
        
        // 临时创建一些模拟数据
        vector<KnowledgeEntry> streamed_data = {
            {"流式数据1", "这是从 " + dataset_name + " 流式加载的数据", "流式", dataset_name},
            {"流式数据2", "类别过滤: " + category_filter, "流式", dataset_name},
            {"流式数据3", "成本优化模式下的数据", "流式", dataset_name}
        };
        
        for (const auto& entry : streamed_data) {
            rag_loader.addKnowledgeEntry(entry);
        }
        
        return true;
    }
};

// 使用示例
void exampleCostOptimizedLoading() {
    cout << "💰 成本优化数据加载演示" << endl;
    cout << "======================" << endl;
    
    CostOptimizedLoader cost_loader;
    
    // 设置成本优化配置
    CostOptimizedLoader::CostConfig config;
    config.use_cloud_storage = false;  // 本地运行
    config.max_local_storage_mb = 512; // 512MB限制
    config.streaming_mode = true;      // 启用流式
    config.priority_categories = {"计算机科学", "机器学习", "数学"};
    
    cost_loader.setCostConfig(config);
    
    // 加载多个数据集
    vector<string> datasets = {
        "HuggingFaceFW/fineweb-edu-100b-shuffle",
        "ethanolivertroy/nist-cybersecurity-training",
        "nick007x/github-code-2025"
    };
    
    map<string, string> category_filters = {
        {"HuggingFaceFW/fineweb-edu-100b-shuffle", "技术"},
        {"nick007x/github-code-2025", "计算机科学"}
    };
    
    cost_loader.loadMultipleDatasets(datasets, category_filters);
    
    // 生成成本报告
    cost_loader.generateCostReport();
    
    // 生成Logic树
    auto& rag_loader = cost_loader.getRAGLoader();
    auto logics = rag_loader.generateLogicTreeFromCategory("计算机科学", 10, 0.6);
    
    cout << "\n🌳 生成 " << logics.size() << " 个Logic" << endl;
    
    cout << "\n🎯 成本优化策略总结:" << endl;
    cout << "1. 使用流式加载避免大文件下载" << endl;
    cout << "2. 设置存储限制控制成本" << endl;
    cout << "3. 按类别过滤减少数据量" << endl;
    cout << "4. 优先加载高质量小数据集" << endl;
    cout << "5. 定期清理缓存文件" << endl;
}

// 主函数
int main() {
    cout << "🚀 成本优化数据加载系统" << endl;
    cout << "======================" << endl;
    
    try {
        exampleCostOptimizedLoading();
        
        cout << "\n💡 实际部署建议:" << endl;
        cout << "1. 对于个人使用: 使用流式加载 + 小样本" << endl;
        cout << "2. 对于团队使用: 考虑AWS S3 + 缓存策略" << endl;
        cout << "3. 对于生产环境: 云存储 + CDN + 缓存" << endl;
        cout << "4. 预算控制: 设置月度数据使用限额" << endl;
        
    } catch (const exception& e) {
        cerr << "❌ 系统错误: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}