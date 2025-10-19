# RAGKnowledgeBaseLoader与Hugging Face数据集集成说明

## 功能概述

本项目扩展了RAGKnowledgeBaseLoader类，增加了以下功能：

1. **Hugging Face数据集流式解析** - 支持从Hugging Face数据集流式获取数据
2. **ExternalStorage集成** - 自动将获取的数据插入到ExternalStorage中
3. **存储大小检查和清理** - 检查存储大小并在必要时清理L3缓存
4. **Logic匹配不足时的自动数据获取** - 在Logic匹配不足时自动从Hugging Face数据集获取数据
5. **语义查询引擎集成** - 集成E5模型进行语义相似度计算

## 主要新增功能

### 1. Hugging Face数据集流式解析
- `streamHuggingFaceDataset` - 流式解析Hugging Face数据集
- `queryAndLoadFromHFDataset` - 查询并加载Hugging Face数据集中的相关条目

### 2. ExternalStorage集成
- `setExternalStorage` - 设置ExternalStorage实例
- `insertToExternalStorage` - 将数据插入ExternalStorage
- `checkAndCleanupStorage` - 检查存储大小并清理
- `cleanupL3Cache` - 清理L3缓存中的冷数据

### 3. Logic匹配不足时的自动数据获取
- `autoFetchDataWhenLogicInsufficient` - 在Logic匹配不足时自动获取数据

### 4. 语义查询引擎
- 独立的`SemanticQueryEngine`类，集成E5模型
- 支持文本嵌入生成和语义相似度计算

## 新增文件

1. `semantic_query_engine.h` - 语义查询引擎头文件
2. `semantic_query_engine.cpp` - 语义查询引擎实现文件
3. `huggingface_streaming.py` - Python脚本用于Hugging Face数据集处理
4. `demo_rag_enhanced.cpp` - 增强版RAG演示程序
5. `huggingface_rag_integration.cpp` - Hugging Face集成演示程序

## 使用方法

### 基本使用
```cpp
// 创建ExternalStorage实例
auto storage = std::make_shared<ExternalStorage<KnowledgeEntry>>(1000, 100.0, 10.0);

// 创建RAG知识库加载器
RAGKnowledgeBaseLoader loader("", "", 100, 0.3, 2000);

// 设置ExternalStorage
loader.setExternalStorage(storage);
loader.setMaxStorageSize(500);  // 设置最大存储大小为500条目

// 流式解析Hugging Face数据集
loader.streamHuggingFaceDataset("HuggingFaceFW/fineweb", "sample-10BT", "train", 10, "test_category");

// 查询和加载
loader.queryAndLoadFromHFDataset("machine learning", "HuggingFaceFW/fineweb", "sample-10BT", 5, "ml_research");

// Logic匹配不足时自动获取数据
loader.autoFetchDataWhenLogicInsufficient("artificial intelligence", 10, "HuggingFaceFW/fineweb", "sample-10BT");
```

### 语义查询使用
```cpp
// 创建语义查询引擎
SemanticQueryEngine semantic_engine("/models/e5/e5_large.onnx");

// 获取文本嵌入
FeatureVector<float> feature;
if (semantic_engine.getTextEmbedding("人工智能技术", feature)) {
    std::cout << "成功获取文本嵌入，维度: " << feature.dimension << std::endl;
}

// 计算语义相似度
double similarity = semantic_engine.getSemanticSimilarity("机器学习", "深度学习");
std::cout << "语义相似度: " << similarity << std::endl;
```

## 已修改的文件

1. `rag_knowledge_loader.h` - 添加了新的方法声明和语义引擎成员
2. `rag_knowledge_loader.cpp` - 实现了新的功能
3. `isw.hpp` - 修复了错误嵌套的SemanticQueryEngine类，移除了语义查询相关代码
4. `CMakeLists.txt` - 添加了新源文件和示例程序

## 实现细节

### Hugging Face集成
- 通过Python脚本`huggingface_streaming.py`实现真实的Hugging Face数据集处理
- C++代码通过系统调用执行Python脚本并解析返回的JSON数据
- 支持流式处理，防止内存爆炸，分批获取和处理数据
- 在Python环境不可用时回退到模拟实现

### 流式处理机制
- **分批处理**：将大量数据分成小批次处理，每批次默认处理10个条目
- **临时文件管理**：每个批次处理完成后立即删除临时JSON文件，释放磁盘空间
- **偏移量支持**：支持从指定位置开始获取数据，避免重复处理
- **内存控制**：通过分批处理有效控制内存使用，防止存储爆炸
- **智能同步**：只将最新获取的相关条目同步到ExternalStorage，避免存储爆炸

### 语义查询引擎
- 独立的`SemanticQueryEngine`类，避免了之前错误嵌套在ExternalStorage中的问题
- 集成smry.cpp中的E5模型实现
- 支持批量文本嵌入生成和语义搜索

### ExternalStorage改进
- 修复了类结构问题，移除了错误嵌套的语义查询引擎
- 保持了原有的存储层级管理（L2内存 + L3磁盘）
- 支持热度管理和自动清理
- 添加了公共删除方法，支持从存储中物理删除条目

## 注意事项

1. 需要安装Python和datasets库：`pip install datasets`
2. 确保模型文件路径正确
3. ExternalStorage的清理功能需要定期调用以保持存储大小在限制范围内
4. 在生产环境中，建议使用真实的Hugging Face API而不是流式处理

## 示例程序

1. `demo_rag_enhanced.cpp` - 演示所有新功能的综合示例
2. `huggingface_rag_integration.cpp` - 专门演示Hugging Face集成的示例
3. `example_external_storage_api.cpp` - 演示ExternalStorage API的使用

编译后可以通过以下命令运行示例：
```bash
./demo_rag_enhanced
./huggingface_rag_integration
./example_external_storage_api
```