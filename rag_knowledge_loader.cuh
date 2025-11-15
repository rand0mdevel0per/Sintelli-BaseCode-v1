// RAG Knowledge Base Loader Implementation
#pragma once

#include "openai_client.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <utility>
#include "semantic_matcher.cuh"

struct KnowledgeEntry {
    std::string title; // 标题
    std::string content; // 内容
    std::string category; // 类别
    std::string source; // 数据源
    double relevance_score; // 相关性分数
    std::vector<std::string> tags; // 标签

    explicit KnowledgeEntry(const std::string t = "", const std::string &c = "",
                   const std::string &cat = "", const std::string &src = "")
        : title(std::move(t)), content(c), category(cat), source(src), relevance_score(0.0) {
    }

    // 添加hash函数以支持ExternalStorage
    [[nodiscard]] std::string hash() const {
        // 简单的hash实现，可以使用title和content的组合
        // return std::to_string(std::hash<std::string>{}(title + content));
        return sha256_hash<KnowledgeEntry>(*this);
    }

    // 添加getText函数以支持ExternalStorage的语义查询
    [[nodiscard]] std::string getText() const {
        return content;
    }
};

// RAG知识库加载器类
class RAGKnowledgeBaseLoader {
private:
    std::unique_ptr<OpenAIClient::HttpClient> openai_client;
    std::unique_ptr<LogicSemanticMatcher> logic_matcher;
    std::unique_ptr<SemanticQueryEngine> semantic_engine; // 语义查询引擎
    std::vector<KnowledgeEntry> knowledge_base;
    std::map<std::string, std::vector<KnowledgeEntry> > category_to_entries;

    // ExternalStorage集成
    std::shared_ptr<ExternalStorage<KnowledgeEntry> > external_storage;

    // 配置参数
    int max_entries_per_category;
    double min_relevance_threshold;
    int max_content_length;
    size_t max_storage_size; // 最大存储大小限制

public:
    RAGKnowledgeBaseLoader(const std::string &openai_api_key = "",
                           const std::string &base_url = "https://api.openai.com/v1",
                           int max_entries = 100,
                           double min_relevance = 0.3,
                           int max_length = 2000);

    // === Data Source Loading Methods ===
    // Methods for loading knowledge from various data sources including URLs, files, and APIs

    // 从URL加载知识库（支持Wikipedia、arXiv等）
    bool loadFromURL(const std::string &url, const std::string &category = "general");

    // 从本地文件加载知识库
    bool loadFromFile(const std::string &file_path, const std::string &category = "general");

    // 从API端点加载知识库
    bool loadFromAPI(const std::string &api_endpoint,
                     const std::string &query,
                     const std::string &category = "general",
                     const std::map<std::string, std::string> &params = {});

    // arXiv API专用方法
    bool fetchArxivPapers(const std::string &query,
                          const std::string &category = "学术论文",
                          int max_results = 10);

    // 从JSON数据加载知识库
    bool loadFromJSON(const std::string &json_data, const std::string &category = "general");

    // 从CSV数据加载知识库
    bool loadFromCSV(const std::string &csv_data, const std::string &category = "general");

    // === Hugging Face Dataset Streaming Methods ===
    // Methods for streaming and processing Hugging Face datasets
    // Supports large dataset processing without memory overflow

    /**
     * @brief Stream and parse Hugging Face dataset.
     *
     * Processes large Hugging Face datasets in streaming mode to avoid memory overflow.
     * Supports various dataset subsets and splits with configurable entry limits.
     *
     * @param dataset_name Name of the Hugging Face dataset
     * @param subset Dataset subset (default: "default")
     * @param split Data split (default: "train")
     * @param max_entries Maximum number of entries to process
     * @param category Category for the processed entries
     * @return true if successful, false otherwise
     */
    bool streamHuggingFaceDataset(const std::string &dataset_name,
                                  const std::string &subset = "default",
                                  const std::string &split = "train",
                                  int max_entries = 100,
                                  const std::string &category = "huggingface");

    /**
     * @brief Query and load relevant entries from Hugging Face dataset.
     *
     * Searches Hugging Face dataset for entries matching the query text.
     * Useful for targeted knowledge retrieval and RAG enhancement.
     *
     * @param query Search query text
     * @param dataset_name Name of the Hugging Face dataset
     * @param subset Dataset subset
     * @param max_results Maximum number of results to return
     * @param category Category for the results
     * @return true if successful, false otherwise
     */
    bool queryAndLoadFromHFDataset(const std::string &query,
                                   const std::string &dataset_name = "HuggingFaceFW/fineweb",
                                   const std::string &subset = "sample-10BT",
                                   int max_results = 50,
                                   const std::string &category = "huggingface_query");

    /**
     * @brief Automatically fetch data when Logic matches are insufficient.
     *
     * Dynamically retrieves additional data from external sources when
     * existing Logic matches fall below the minimum required threshold.
     *
     * @param query Search query to find relevant data
     * @param min_required_matches Minimum required Logic matches
     * @param dataset_name Source dataset name
     * @param subset Dataset subset
     * @return true if data was successfully fetched and integrated
     */
    bool autoFetchDataWhenLogicInsufficient(const std::string &query,
                                            int min_required_matches = 5,
                                            const std::string &dataset_name = "HuggingFaceFW/fineweb",
                                            const std::string &subset = "sample-10BT");

    // === Logic Generation Methods ===
    // Methods for generating Logic trees from knowledge entries
    // Integrates with AI models for enhanced Logic creation

    /**
     * @brief Automatically generate Logic tree from knowledge text.
     *
     * Processes knowledge text to create structured Logic descriptors
     * with semantic embeddings for matching and retrieval.
     *
     * @param knowledge_text Input knowledge text
     * @param category Category for the generated Logic
     * @param activation_threshold Activation threshold for the Logic
     * @return Vector of generated Logic descriptors
     */
    std::vector<LogicDescriptor> generateLogicTreeFromKnowledge(
        const std::string &knowledge_text,
        const std::string &category = "general",
        double activation_threshold = 0.5);

    /**
     * @brief Batch generate Logic trees from a category.
     *
     * Creates multiple Logic descriptors from all knowledge entries
     * belonging to the specified category.
     *
     * @param category Category of knowledge entries to process
     * @param max_logics Maximum number of Logic descriptors to generate
     * @param activation_threshold Activation threshold for generated Logic
     * @return Vector of generated Logic descriptors
     */
    std::vector<LogicDescriptor> generateLogicTreeFromCategory(
        const std::string &category,
        int max_logics = 50,
        double activation_threshold = 0.5);

    // 使用AI辅助生成Logic描述
    std::string generateLogicDescriptionWithAI(const std::string &knowledge_content);

    // 注册Logic到语义匹配器
    bool registerLogicTree(LogicSemanticMatcher &matcher,
                           const std::vector<LogicDescriptor> &logics);

    // === Knowledge Base Management ===
    // Methods for managing and querying the knowledge base
    // Includes search, statistics, and export functionality

    // 添加知识条目
    bool addKnowledgeEntry(const KnowledgeEntry &entry);

    // 搜索知识库
    std::vector<KnowledgeEntry> searchKnowledge(const std::string &query,
                                                int max_results = 10,
                                                const std::string &category = "");

    // 获取知识库统计信息
    struct KnowledgeStats {
        size_t total_entries;
        std::map<std::string, size_t> category_counts;
        size_t unique_categories;
        double avg_relevance_score;
    };

    KnowledgeStats getKnowledgeStats() const;

    // 清空知识库
    void clearKnowledgeBase();

    // 导出知识库到文件
    bool exportToFile(const std::string &file_path, const std::string &format = "json");

    // 获取所有知识条目
    const std::vector<KnowledgeEntry> &getAllEntries() const { return knowledge_base; }

    // 按类别获取知识条目
    std::vector<KnowledgeEntry> getEntriesByCategory(const std::string &category) const;

    // === Configuration Methods ===
    // Methods for configuring the RAG knowledge base loader
    // Allows customization of loading parameters and behavior

    // 设置OpenAI客户端
    void setOpenAIClient(std::unique_ptr<OpenAIClient::HttpClient> client);

    // 设置Logic匹配器
    void setLogicMatcher(std::unique_ptr<LogicSemanticMatcher> matcher);

    // 配置参数
    void setMaxEntriesPerCategory(int max_entries);

    void setMinRelevanceThreshold(double threshold);

    void setMaxContentLength(int max_length);

    void setMaxStorageSize(size_t max_size);

    // ExternalStorage相关方法
    void setExternalStorage(std::shared_ptr<ExternalStorage<KnowledgeEntry> > storage);

    std::shared_ptr<ExternalStorage<KnowledgeEntry> > getExternalStorage() const;

    bool insertToExternalStorage(const KnowledgeEntry &entry);

    bool insertToExternalStorage(const std::vector<KnowledgeEntry> &entries);

    bool insertToExternalStorageWithSemanticFeatures(const KnowledgeEntry &entry);

    bool checkAndCleanupStorage();

    void cleanupL3Cache(int num_entries_to_remove = 10);

    // Logic系统集成方法
    bool registerKnowledgeAsLogic(LogicInjector *logic_injector,
                                  ExternalStorage<Logic> *logic_tree,
                                  const std::string &category = "rag_knowledge");

    bool autoFetchAndRegisterLogic(LogicInjector *logic_injector,
                                   ExternalStorage<Logic> *logic_tree,
                                   const std::string &query,
                                   int min_logics = 10,
                                   const std::string &dataset_name = "HuggingFaceFW/fineweb",
                                   const std::string &subset = "sample-10BT",
                                   const std::string &category = "rag_knowledge");

    // 语义搜索方法
    std::vector<KnowledgeEntry> semanticSearchKnowledge(const std::string &query,
                                                        int top_k = 10,
                                                        double similarity_threshold = 0.3);

private:
    // 内部辅助方法
    std::string extractKeyPoints(const std::string &content);

    std::string summarizeContent(const std::string &content, int max_length = 200);

    double calculateRelevance(const std::string &content, const std::string &category);

    std::vector<std::string> extractTags(const std::string &content);

    // 数据源特定解析器
    bool parseWikipediaResponse(const std::string &response, KnowledgeEntry &entry);

    bool parseArxivResponse(const std::string &response, KnowledgeEntry &entry);

    bool parseJSONData(const std::string &json_data, std::vector<KnowledgeEntry> &entries);

    bool parseCSVData(const std::string &csv_data, std::vector<KnowledgeEntry> &entries);

    // OpenAI API调用
    std::string callOpenAICompletion(const std::string &prompt,
                                     const std::string &model = "gpt-3.5-turbo",
                                     double temperature = 0.7);
};

char *wideCharToMultiByte(wchar_t *pWCStrKey) {
    // First call to confirm the length of the converted single-byte string, used to allocate space
    int pSize = WideCharToMultiByte(CP_OEMCP, 0, pWCStrKey, wcslen(pWCStrKey), NULL, 0, NULL, NULL);
    char *pCStrKey = new char[pSize + 1];
    // Second call to convert double-byte string to single-byte string
    WideCharToMultiByte(CP_OEMCP, 0, pWCStrKey, wcslen(pWCStrKey), pCStrKey, pSize, NULL, NULL);
    pCStrKey[pSize] = '\0';
    return pCStrKey;

    // If you want to convert to string, just assign directly
    //string pKey = pCStrKey;
}

wchar_t *multiByteToWideChar(const std::string &pKey) {
    const char *pCStrKey = pKey.c_str();
    // First call returns the length of the converted string, used to confirm how much memory space to allocate for wchar_t*
    int pSize = MultiByteToWideChar(CP_OEMCP, 0, pCStrKey, strlen(pCStrKey) + 1, NULL, 0);
    wchar_t *pWCStrKey = new wchar_t[pSize];
    // Second call to convert single-byte string to double-byte string
    MultiByteToWideChar(CP_OEMCP, 0, pCStrKey, strlen(pCStrKey) + 1, pWCStrKey, pSize);
    return pWCStrKey;
}

using namespace std;

// 回调函数用于CURL写入数据
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, string *userp) {
    size_t total_size = size * nmemb;
    userp->append((char *) contents, total_size);
    return total_size;
}

RAGKnowledgeBaseLoader::RAGKnowledgeBaseLoader(const string &openai_api_key,
                                               const string &base_url,
                                               int max_entries,
                                               double min_relevance,
                                               int max_length)
    : max_entries_per_category(max_entries),
      min_relevance_threshold(min_relevance),
      max_content_length(max_length),
      max_storage_size(10000) {
    // 默认最大存储大小为10000条目

    if (!openai_api_key.empty()) {
        openai_client = make_unique<OpenAIClient::HttpClient>(openai_api_key, base_url);
    }

    // Initialize Logic matcher
    logic_matcher = make_unique<LogicSemanticMatcher>("/models/e5/e5_large.onnx");

    // Initialize semantic query engine
    semantic_engine = make_unique<SemanticQueryEngine>("/models/e5/e5_large.onnx");
}

bool RAGKnowledgeBaseLoader::loadFromURL(const string &url, const string &category) {
    CURL *curl;
    CURLcode res;
    string response;

    curl = curl_easy_init();
    if (!curl) {
        cerr << "Unable to init curl" << endl;
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "RAGKnowledgeLoader/1.0");
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L); // 30秒超时

    res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        cerr << "cUrl fetch failed: " << curl_easy_strerror(res) << endl;
        return false;
    }

    // 根据URL类型解析响应
    KnowledgeEntry entry;
    bool success = false;

    if (url.find("wikipedia") != string::npos) {
        success = parseWikipediaResponse(response, entry);
    } else if (url.find("arxiv") != string::npos) {
        success = parseArxivResponse(response, entry);
    } else {
        // 通用解析
        entry.title = "Content loaded from url:";
        entry.content = response.substr(0, max_content_length);
        entry.category = category;
        entry.source = url;
        success = true;
    }

    if (success) {
        entry.relevance_score = calculateRelevance(entry.content, category);
        entry.tags = extractTags(entry.content);
        return addKnowledgeEntry(entry);
    }

    return false;
}

bool RAGKnowledgeBaseLoader::streamHuggingFaceDataset(const string &dataset_name,
                                                      const string &subset,
                                                      const string &split,
                                                      int max_entries,
                                                      const string &category) {
    cout << "Get dataset through python script..." << endl;

    // 分段处理，防止内存爆炸
    const int batch_size = 10; // 每批次处理10个条目
    int total_processed = 0;
    bool success = true;

    while (total_processed < max_entries) {
        int current_batch_size = min(batch_size, max_entries - total_processed);
        string output_file = "temp_hf_stream_" + to_string(total_processed) + ".json";

        // 构建Python脚本命令
        string cmd = "python \"" + string(getenv("PWD") ? getenv("PWD") : ".") + "/huggingface_streaming.py\" stream";
        cmd += " --dataset \"" + dataset_name + "\"";
        if (!subset.empty()) {
            cmd += " --subset \"" + subset + "\"";
        }
        cmd += " --split \"" + split + "\"";
        cmd += " --max-entries " + to_string(current_batch_size);
        cmd += " --category \"" + category + "\"";
        cmd += " --output \"" + output_file + "\"";
        cmd += " --offset " + to_string(total_processed); // 添加偏移量参数

        cout << "   Execute command: " << cmd << endl;

        // 执行Python脚本
        int result = system(cmd.c_str());

        if (result != 0) {
            cerr << "Script execution failed (RetVal: " << result << ")" << endl;
            success = false;
            break;
        }

        // 读取Python脚本生成的JSON文件
        ifstream file(output_file);
        if (!file.is_open()) {
            cerr << "Unable to open the temp file generated by the python script: " << output_file << endl;
            success = false;
            break;
        }

        stringstream buffer;
        buffer << file.rdbuf();
        string json_content = buffer.str();
        file.close();

        // 解析JSON内容
        vector<KnowledgeEntry> entries;
        if (!parseJSONData(json_content, entries)) {
            cerr << "Unable to parse the json file generated by script: " << output_file << endl;
            success = false;
            // 删除临时文件
            remove(output_file.c_str());
            break;
        }

        // 添加到知识库
        int batch_added = 0;
        for (auto &entry: entries) {
            if (total_processed + batch_added >= max_entries) {
                break; // 达到最大条目数限制
            }

            entry.relevance_score = calculateRelevance(entry.content, category);
            entry.tags = extractTags(entry.content);

            if (!addKnowledgeEntry(entry)) {
                cerr << "Unable to add category from HuggingFace dataset: " << total_processed + batch_added + 1 <<
                        endl;
                success = false;
            } else {
                batch_added++;
            }
        }

        total_processed += batch_added;
        cout << "   Batch completed " << total_processed << " / " << max_entries << " categories." << endl;

        // 删除临时文件
        remove(output_file.c_str());

        // 如果这一批次没有添加任何条目，说明数据已经用完
        if (batch_added == 0) {
            cout << "   Dataset ended" << endl;
            break;
        }
    }

    if (success && total_processed > 0) {
        cout << "Succeeded to" << dataset_name << "load " << total_processed << " categories in stream" << endl;
    } else if (total_processed == 0) {
        cerr << "Unable to load any category from" << dataset_name << endl;
        return false;
    }

    return success;
}

bool RAGKnowledgeBaseLoader::loadFromFile(const string &file_path, const string &category) {
    ifstream file(file_path);
    if (!file.is_open()) {
        cerr << "Unable to open file: " << file_path << endl;
        return false;
    }

    stringstream buffer;
    buffer << file.rdbuf();
    string content = buffer.str();

    // 根据文件扩展名选择解析器
    if (file_path.find(".json") != string::npos) {
        return loadFromJSON(content, category);
    } else if (file_path.find(".csv") != string::npos) {
        return loadFromCSV(content, category);
    } else {
        // 普通文本文件
        KnowledgeEntry entry;
        entry.title = file_path.substr(file_path.find_last_of("/\\") + 1);
        entry.content = content.substr(0, max_content_length);
        entry.category = category;
        entry.source = file_path;
        entry.relevance_score = calculateRelevance(entry.content, category);
        entry.tags = extractTags(entry.content);

        return addKnowledgeEntry(entry);
    }
}

bool RAGKnowledgeBaseLoader::queryAndLoadFromHFDataset(const string &query,
                                                       const string &dataset_name,
                                                       const string &subset,
                                                       int max_results,
                                                       const string &category) {
    cout << "Query HuggingFace dataset through python script..." << endl;

    // 分段处理查询结果，防止内存爆炸
    const int batch_size = 5; // 每批次处理5个条目
    int total_processed = 0;
    bool success = true;

    while (total_processed < max_results) {
        int current_batch_size = min(batch_size, max_results - total_processed);
        string output_file = "temp_hf_query_" + to_string(total_processed) + ".json";

        // 构建Python脚本命令
        string cmd = "python \"" + string(getenv("PWD") ? getenv("PWD") : ".") + "/huggingface_streaming.py\" query";
        cmd += " --dataset \"" + dataset_name + "\"";
        if (!subset.empty()) {
            cmd += " --subset \"" + subset + "\"";
        }
        cmd += " --query \"" + query + "\"";
        cmd += " --max-entries " + to_string(current_batch_size);
        cmd += " --category \"" + category + "\"";
        cmd += " --output \"" + output_file + "\"";

        cout << "   Execute: " << cmd << endl;

        // 执行Python脚本
        int result = system(cmd.c_str());

        if (result != 0) {
            cerr << "Python script execution failed (RetVal: " << result << ")" << endl;
            success = false;
            break;
        }

        // 读取Python脚本生成的JSON文件
        ifstream file(output_file);
        if (!file.is_open()) {
            cerr << "Unable to open temp file generated by python script: " << output_file << endl;
            success = false;
            break;
        }

        stringstream buffer;
        buffer << file.rdbuf();
        string json_content = buffer.str();
        file.close();

        // 解析JSON内容
        vector<KnowledgeEntry> entries;
        if (!parseJSONData(json_content, entries)) {
            cerr << "Unable to parse json from python script: " << output_file << endl;
            success = false;
            // 删除临时文件
            remove(output_file.c_str());
            break;
        }

        // 添加到知识库
        int batch_added = 0;
        for (auto &entry: entries) {
            if (total_processed + batch_added >= max_results) {
                break; // 达到最大条目数限制
            }

            entry.relevance_score = calculateRelevance(entry.content, category);
            entry.tags = extractTags(entry.content);

            if (!addKnowledgeEntry(entry)) {
                cerr << "Unable to add categories: " << total_processed + batch_added + 1 << endl;
                success = false;
            } else {
                batch_added++;
            }
        }

        total_processed += batch_added;
        cout << "   Batch completed.Proceeded  " << total_processed << " / " << max_results << " categories" << endl;

        // 删除临时文件
        remove(output_file.c_str());

        // 如果这一批次没有添加任何条目，说明查询结果已经用完
        if (batch_added == 0) {
            cout << "   Dataset ended." << endl;
            break;
        }
    }

    if (success && total_processed > 0) {
        cout << "Succeeded to query and load " << total_processed << "categories from " << dataset_name << endl;
    } else if (total_processed == 0) {
        cerr << "Unable to get any results from " << dataset_name << endl;
        return false;
    }

    return success;
}

bool RAGKnowledgeBaseLoader::loadFromJSON(const string &json_data, const string &category) {
    vector<KnowledgeEntry> entries;
    if (!parseJSONData(json_data, entries)) {
        return false;
    }

    bool success = true;
    for (auto &entry: entries) {
        entry.category = category;
        entry.relevance_score = calculateRelevance(entry.content, category);
        entry.tags = extractTags(entry.content);

        if (!addKnowledgeEntry(entry)) {
            success = false;
        }
    }

    return success;
}

vector<LogicDescriptor> RAGKnowledgeBaseLoader::generateLogicTreeFromKnowledge(
    const string &knowledge_text,
    const string &category,
    double activation_threshold) {
    vector<LogicDescriptor> logics;

    // 提取关键点
    string key_points = extractKeyPoints(knowledge_text);
    string summary = summarizeContent(knowledge_text, 100);

    // 使用AI生成Logic描述（如果有OpenAI客户端）
    string logic_description;
    if (openai_client) {
        logic_description = generateLogicDescriptionWithAI(knowledge_text);
    } else {
        logic_description = summary;
    }

    // 生成Logic ID
    string logic_id = "logic_" + to_string(hash<string>{}(knowledge_text));

    // 创建Logic描述符
    LogicDescriptor logic;
    logic.logic_id = logic_id;
    logic.description = logic_description;
    logic.category = category;
    logic.activation_threshold = activation_threshold;

    // 创建默认的回调函数
    logic.generate_input_callback = LogicDescriptor::createDefaultGenerator(
        knowledge_text, 1.0, 1.0, 0, 0, 0);

    logics.push_back(logic);

    return logics;
}

vector<LogicDescriptor> RAGKnowledgeBaseLoader::generateLogicTreeFromCategory(
    const string &category,
    int max_logics,
    double activation_threshold) {
    vector<LogicDescriptor> all_logics;

    auto it = category_to_entries.find(category);
    if (it == category_to_entries.end()) {
        return all_logics;
    }

    const auto &entries = it->second;
    int count = 0;

    for (const auto &entry: entries) {
        if (count >= max_logics) break;

        auto logics = generateLogicTreeFromKnowledge(entry.content, category, activation_threshold);
        all_logics.insert(all_logics.end(), logics.begin(), logics.end());
        count++;
    }

    return all_logics;
}

string RAGKnowledgeBaseLoader::generateLogicDescriptionWithAI(const string &knowledge_content) {
    if (!openai_client) {
        return "Category: " + summarizeContent(knowledge_content, 50);
    }

    string prompt = "Please generate a simple description for the knowledge below(not above 30 tokens):\n\n" +
                    knowledge_content.substr(0, 500) +
                    "\n\nLogic Description:";

    try {
        OpenAIClient::ChatCompletionRequest request;
        request.model = "minimax/minimax-m2:free";
        request.messages.push_back(OpenAIClient::ChatMessage("user", prompt));
        request.temperature = 0.7;
        request.max_tokens = 50;

        auto response = openai_client->createChatCompletion(request);

        if (!response.choices.empty()) {
            return response.choices[0].message.content;
        }
    } catch (const exception &e) {
        cerr << "Unable to get response from api: " << e.what() << endl;
    }

    return "知识条目: " + summarizeContent(knowledge_content, 30);
}

bool RAGKnowledgeBaseLoader::registerLogicTree(LogicSemanticMatcher &matcher,
                                               const vector<LogicDescriptor> &logics) {
    bool success = true;

    for (const auto &logic: logics) {
        if (!matcher.registerLogic(logic)) {
            cerr << "Failed in registering logics: " << logic.logic_id << endl;
            success = false;
        } else {
            cout << "Succeeded in registering logics: " << logic.logic_id << endl;
        }
    }

    return success;
}

bool RAGKnowledgeBaseLoader::addKnowledgeEntry(const KnowledgeEntry &entry) {
    // 检查是否超过最大条目数
    if (category_to_entries[entry.category].size() >= max_entries_per_category) {
        // 移除相关性最低的条目
        auto &entries = category_to_entries[entry.category];
        auto min_it = min_element(entries.begin(), entries.end(),
                                  [](const KnowledgeEntry &a, const KnowledgeEntry &b) {
                                      return a.relevance_score < b.relevance_score;
                                  });

        if (min_it != entries.end() && entry.relevance_score > min_it->relevance_score) {
            entries.erase(min_it);
        } else {
            return false; // 新条目相关性不够高
        }
    }

    // 检查相关性阈值
    if (entry.relevance_score < min_relevance_threshold) {
        return false;
    }

    // 添加到知识库
    knowledge_base.push_back(entry);
    category_to_entries[entry.category].push_back(entry);

    return true;
}

vector<KnowledgeEntry> RAGKnowledgeBaseLoader::searchKnowledge(const string &query,
                                                               int max_results,
                                                               const string &category) {
    vector<KnowledgeEntry> results;

    // 如果指定了类别，只在该类别中搜索
    if (!category.empty()) {
        auto it = category_to_entries.find(category);
        if (it != category_to_entries.end()) {
            const auto &entries = it->second;

            // 简单的文本匹配搜索
            for (const auto &entry: entries) {
                // 检查标题、内容或标签是否包含查询关键词
                if (entry.title.find(query) != string::npos ||
                    entry.content.find(query) != string::npos) {
                    results.push_back(entry);

                    // 限制结果数量，避免返回过多数据
                    if (results.size() >= static_cast<size_t>(max_results)) {
                        break;
                    }
                }
            }
        }
    } else {
        // 在所有条目中搜索
        for (const auto &entry: knowledge_base) {
            // 检查标题、内容或标签是否包含查询关键词
            if (entry.title.find(query) != string::npos ||
                entry.content.find(query) != string::npos) {
                results.push_back(entry);

                // 限制结果数量，避免返回过多数据
                if (results.size() >= static_cast<size_t>(max_results)) {
                    break;
                }
            }
        }
    }

    return results;
}

// 辅助方法实现
string RAGKnowledgeBaseLoader::extractKeyPoints(const string &content) {
    // 简单的关键点提取（可以改进为更复杂的NLP方法）
    vector<string> sentences;
    stringstream ss(content);
    string sentence;

    while (getline(ss, sentence, '.')) {
        if (sentence.length() > 20) {
            // 过滤短句
            sentences.push_back(sentence);
        }
    }

    // 取前3个句子作为关键点
    string key_points;
    for (size_t i = 0; i < min(sentences.size(), size_t(3)); ++i) {
        key_points += sentences[i] + ". ";
    }

    return key_points;
}

string RAGKnowledgeBaseLoader::summarizeContent(const string &content, int max_length) {
    if (content.length() <= max_length) {
        return content;
    }

    // 简单的截断摘要
    return content.substr(0, max_length) + "...";
}

double RAGKnowledgeBaseLoader::calculateRelevance(const string &content, const string &category) {
    // 简单的相关性计算（可以改进为基于语义的算法）
    double relevance = 0.5; // 基础分数

    // 基于内容长度
    relevance += min(content.length() / 1000.0, 0.3);

    // 基于关键词匹配（如果类别有特定关键词）
    map<string, vector<string> > category_keywords = {
        {"compute science", {"algorithms", "coding", "data structure", "artifical intelligence"}},
        {"maths", {"theorem", "formula", "prove", "calculate"}},
        {"physical", {"mechanics", "quantum", "relativity", "energy"}}
    };

    if (category_keywords.count(category)) {
        const auto &keywords = category_keywords[category];
        for (const auto &keyword: keywords) {
            if (content.find(keyword) != string::npos) {
                relevance += 0.1;
            }
        }
    }

    return min(relevance, 1.0);
}

vector<string> RAGKnowledgeBaseLoader::extractTags(const string &content) {
    vector<string> tags;

    // 简单的标签提取（可以改进为NLP方法）
    vector<string> common_tags = {"technology", "science", "educational", "research", "innovation"};

    for (const auto &tag: common_tags) {
        if (content.find(tag) != string::npos) {
            tags.push_back(tag);
        }
    }

    return tags;
}

// arXiv API解析实现
bool RAGKnowledgeBaseLoader::parseArxivResponse(const string &response, KnowledgeEntry &entry) {
    // arXiv API返回的是Atom格式，这里简化处理
    // 实际应用中应该使用XML解析器

    try {
        // 简单的XML解析（实际应该用专门的XML库）
        size_t title_start = response.find("<title>");
        size_t title_end = response.find("</title>", title_start);
        size_t summary_start = response.find("<summary>", title_end);
        size_t summary_end = response.find("</summary>", summary_start);

        if (title_start != string::npos && title_end != string::npos) {
            entry.title = response.substr(title_start + 7, title_end - title_start - 7);
            // 移除可能的多余空白
            entry.title.erase(0, entry.title.find_first_not_of(" \n\r\t"));
            entry.title.erase(entry.title.find_last_not_of(" \n\r\t") + 1);
        } else {
            entry.title = "arXiv paper";
        }

        if (summary_start != string::npos && summary_end != string::npos) {
            entry.content = response.substr(summary_start + 9, summary_end - summary_start - 9);
            // 清理摘要内容
            entry.content.erase(0, entry.content.find_first_not_of(" \n\r\t"));
            entry.content.erase(entry.content.find_last_not_of(" \n\r\t") + 1);
        } else {
            entry.content = "arXiv paper abstract";
        }

        entry.category = "academic papers";
        entry.source = "arXiv API";

        return true;
    } catch (const exception &e) {
        cerr << "Unable to parse the response from arxiv api: " << e.what() << endl;
        return false;
    }
}

bool RAGKnowledgeBaseLoader::fetchArxivPapers(const string &query,
                                              const string &category,
                                              int max_results) {
    // 构建arXiv API URL
    string base_url = "http://export.arxiv.org/api/query?";
    string search_query = "search_query=all:" + query + "&max_results=" + to_string(max_results);
    string url = base_url + search_query;

    return loadFromURL(url, category);
}

bool RAGKnowledgeBaseLoader::autoFetchDataWhenLogicInsufficient(const string &query,
                                                                int min_required_matches,
                                                                const string &dataset_name,
                                                                const string &subset) {
    cout << "Insufficient query results.attempting to get data directly from huggingface dataset..." << endl;

    // 直接从HuggingFace数据集查询并加载相关条目（使用流式处理）
    bool success = queryAndLoadFromHFDataset(query, dataset_name, subset, min_required_matches);

    if (success) {
        cout << "Query success" << endl;

        // 插入到外部存储（如果有的话）
        if (external_storage) {
            cout << "Syncing data to ExternalStorage..." << endl;
            auto entries = getAllEntries();
            int count = 0;

            // 只同步新添加的条目，限制数量避免存储爆炸
            int max_to_sync = min(20, min_required_matches); // 最多同步20个或所需数量
            for (int i = entries.size() - 1; i >= 0 && count < max_to_sync; --i) {
                if (!insertToExternalStorage(entries[i])) {
                    cerr << "Failed to sync: " << entries[i].title << endl;
                }
                count++;
            }
        }

        // 检查存储大小并清理L3缓存（如果需要）
        if (external_storage) {
            checkAndCleanupStorage();
        }

        return true;
    }
    cerr << "Failed to fetch data" << endl;
    return false;
}

// Get knowledge base statistics
RAGKnowledgeBaseLoader::KnowledgeStats RAGKnowledgeBaseLoader::getKnowledgeStats() const {
    KnowledgeStats stats;
    stats.total_entries = knowledge_base.size();
    stats.unique_categories = category_to_entries.size();

    double total_relevance = 0.0;
    for (const auto &entry: knowledge_base) {
        stats.category_counts[entry.category]++;
        total_relevance += entry.relevance_score;
    }

    stats.avg_relevance_score = stats.total_entries > 0 ? total_relevance / stats.total_entries : 0.0;

    return stats;
}

// 清空知识库
void RAGKnowledgeBaseLoader::clearKnowledgeBase() {
    knowledge_base.clear();
    category_to_entries.clear();
}

// 导出知识库到文件
bool RAGKnowledgeBaseLoader::exportToFile(const std::string &file_path, const std::string &format) {
    ofstream file(file_path);
    if (!file.is_open()) {
        cerr << "❌ Unable to create export file: " << file_path << endl;
        return false;
    }

    if (format == "json") {
        // 导出为JSON格式
        file << "[\n";
        for (size_t i = 0; i < knowledge_base.size(); ++i) {
            const auto &entry = knowledge_base[i];
            file << "  {\n";
            file << "    \"title\": \"" << entry.title << "\",\n";
            file << "    \"content\": \"" << entry.content << "\",\n";
            file << "    \"category\": \"" << entry.category << "\",\n";
            file << "    \"source\": \"" << entry.source << "\",\n";
            file << "    \"relevance_score\": " << entry.relevance_score << ",\n";
            file << "    \"tags\": [";
            for (size_t j = 0; j < entry.tags.size(); ++j) {
                file << "\"" << entry.tags[j] << "\"";
                if (j < entry.tags.size() - 1) file << ", ";
            }
            file << "]\n";
            file << "  }";
            if (i < knowledge_base.size() - 1) file << ",";
            file << "\n";
        }
        file << "]\n";
    } else {
        // 导出为CSV格式
        file << "title,content,category,source,relevance_score,tags\n";
        for (const auto &entry: knowledge_base) {
            file << "\"" << entry.title << "\",";
            file << "\"" << entry.content << "\",";
            file << "\"" << entry.category << "\",";
            file << "\"" << entry.source << "\",";
            file << entry.relevance_score << ",";
            file << "\"";
            for (size_t i = 0; i < entry.tags.size(); ++i) {
                file << entry.tags[i];
                if (i < entry.tags.size() - 1) file << ";";
            }
            file << "\"\n";
        }
    }

    file.close();
    cout << "✅ Knowledge base exported to: " << file_path << " (" << format << " format)" << endl;
    return true;
}

// 设置OpenAI客户端
void RAGKnowledgeBaseLoader::setOpenAIClient(std::unique_ptr<OpenAIClient::HttpClient> client) {
    openai_client = std::move(client);
}

// 按类别获取知识条目
void RAGKnowledgeBaseLoader::setExternalStorage(std::shared_ptr<ExternalStorage<KnowledgeEntry> > storage) {
    external_storage = storage;
}

std::shared_ptr<ExternalStorage<KnowledgeEntry> > RAGKnowledgeBaseLoader::getExternalStorage() const {
    return external_storage;
}

bool RAGKnowledgeBaseLoader::insertToExternalStorage(const KnowledgeEntry &entry) {
    if (!external_storage) {
        cerr << "❌ External storage not initialized" << endl;
        return false;
    }

    // 存储到外部存储
    uint64_t slot_id = external_storage->store<KnowledgeEntry>(entry);
    if (slot_id == 0) {
        cerr << "❌ Failed to store to external storage" << endl;
        return false;
    }

    cout << "✅ Successfully stored to external storage (slot_id: " << slot_id << ")" << endl;
    return true;
}

std::vector<KnowledgeEntry> RAGKnowledgeBaseLoader::getEntriesByCategory(const std::string &category) const {
    auto it = category_to_entries.find(category);
    if (it != category_to_entries.end()) {
        return it->second;
    }
    return {};
}

// 设置Logic匹配器
void RAGKnowledgeBaseLoader::setLogicMatcher(std::unique_ptr<LogicSemanticMatcher> matcher) {
    logic_matcher = std::move(matcher);
}

// 配置参数
void RAGKnowledgeBaseLoader::setMaxEntriesPerCategory(int max_entries) {
    max_entries_per_category = max_entries;
}

void RAGKnowledgeBaseLoader::setMinRelevanceThreshold(double threshold) {
    min_relevance_threshold = threshold;
}

void RAGKnowledgeBaseLoader::setMaxContentLength(int max_length) {
    max_content_length = max_length;
}

void RAGKnowledgeBaseLoader::setMaxStorageSize(size_t max_size) {
    max_storage_size = max_size;
}

bool RAGKnowledgeBaseLoader::insertToExternalStorage(const std::vector<KnowledgeEntry> &entries) {
    if (!external_storage) {
        cerr << "❌ External storage not initialized" << endl;
        return false;
    }

    bool success = true;
    for (const auto &entry: entries) {
        if (!insertToExternalStorage(entry)) {
            success = false;
        }
    }

    return success;
}

bool RAGKnowledgeBaseLoader::insertToExternalStorageWithSemanticFeatures(const KnowledgeEntry &entry) {
    if (!external_storage) {
        cerr << "❌ External storage not initialized" << endl;
        return false;
    }

    // 存储到外部存储（带语义特征）
    uint64_t slot_id = external_storage->storeWithSemanticFeature<KnowledgeEntry>(
        entry,
        1.0, // initial_heat
        "/models/e5/e5_large.onnx", // model_path
        "/models/e5/vocab.json", // vocab_path
        "/models/e5/merges.txt", // merges_path
        "/models/e5/special_tokens.json" // special_tokens_path
    );

    if (slot_id == 0) {
        cerr << "❌ Failed to store to external storage" << endl;
        return false;
    }

    cout << "✅ Successfully stored to external storage (slot_id: " << slot_id << ") with semantic features" << endl;
    return true;
}

// Logic系统集成方法
bool RAGKnowledgeBaseLoader::registerKnowledgeAsLogic(LogicInjector *logic_injector,
                                                      ExternalStorage<Logic> *logic_tree,
                                                      const std::string &category) {
    if (!logic_injector || !logic_tree) {
        cerr << "❌ Logic injector or Logic storage not initialized" << endl;
        return false;
    }

    bool success = true;
    int registered_count = 0;

    for (const auto &entry: knowledge_base) {
        // Create Logic descriptor (for semantic matching)
        LogicDescriptor logic_desc;
        logic_desc.logic_id = "rag_" + to_string(hash<string>{}(entry.title + entry.content));
        logic_desc.description = entry.title + ": " + entry.content; // 使用标题和内容作为描述
        logic_desc.category = category;
        logic_desc.activation_threshold = 0.4; // 设置激活阈值

        // Create default NeuronInput generator
        logic_desc.generate_input_callback = LogicDescriptor::createDefaultGenerator(
            entry.content, // Logic内容
            1.0, // activity
            1.0, // weight
            0, 0, 0 // 坐标
        );

        // Register to Logic system (descriptor for semantic matching)
        if (!logic_injector->registerLogicWithStorage(logic_desc)) {
            cerr << "❌ Failed to register Logic descriptor: " << entry.title << endl;
            success = false;
        }

        // Create actual Logic object (for content storage)
        Logic actual_logic;
        actual_logic.Rcycles = 0;
        actual_logic.importance = 1.0;

        // Convert content to wide characters and store
        string full_content = entry.title + ": " + entry.content;
        size_t content_len = min(full_content.length(), size_t(1023)); // 保留一个字符给null终止符
        const char *content_ptr = full_content.c_str();

        // Convert to wide characters
        wchar_t *wide_content = multiByteToWideChar(full_content);
        size_t wide_len = wcslen(wide_content);
        size_t copy_len = min(wide_len, size_t(1023));

        // Copy to Logic content array
        wcsncpy(actual_logic.content, wide_content, copy_len);
        actual_logic.content[copy_len] = L'\0'; // Ensure null termination

        delete[] wide_content;

        // 存储到logic_tree（实际内容存储）
        uint64_t slot_id = logic_tree->store(actual_logic, 1.0);
        if (slot_id == 0) {
            cerr << "❌ Failed to store Logic content: " << entry.title << endl;
            success = false;
        } else {
            registered_count++;
        }
    }

    cout << "✅ Successfully registered " << registered_count << " Logics to Logic system" << endl;
    cout << "   - Logic descriptors stored to logic_storage for semantic matching" << endl;
    cout << "   - Logic content stored to logic_tree for actual execution" << endl;
    return success;
}

bool RAGKnowledgeBaseLoader::autoFetchAndRegisterLogic(LogicInjector *logic_injector,
                                                       ExternalStorage<Logic> *logic_tree,
                                                       const std::string &query,
                                                       int min_logics,
                                                       const std::string &dataset_name,
                                                       const std::string &subset,
                                                       const std::string &category) {
    cout << "🔍 Automatically fetch data and register as Logic..." << endl;

    // 从HuggingFace获取数据
    bool fetch_success = queryAndLoadFromHFDataset(query, dataset_name, subset, min_logics, category);

    if (!fetch_success) {
        cerr << "❌ Failed to fetch data" << endl;
        return false;
    }

    // 将获取的知识注册为Logic
    bool register_success = registerKnowledgeAsLogic(logic_injector, logic_tree, category);

    if (register_success) {
        cout << "✅ Successfully fetched and registered " << knowledge_base.size() << " Logics" << endl;
    }

    return register_success;
}

bool RAGKnowledgeBaseLoader::checkAndCleanupStorage() {
    if (!external_storage) {
        return true; // 没有存储，无需清理
    }

    try {
        // 获取存储统计信息
        auto stats = external_storage->getStatistics();

        // 检查总大小是否超过限制
        if (stats.total_size > max_storage_size) {
            // 计算需要移除的条目数量
            size_t excess_entries = stats.total_size - max_storage_size;
            // 清理L3缓存中的冷数据
            cleanupL3Cache(static_cast<int>(excess_entries));
            return true;
        }

        return true;
    } catch (const std::exception &e) {
        cerr << "❌ Failed to check storage size: " << e.what() << std::endl;
        return false;
    }
}

void RAGKnowledgeBaseLoader::cleanupL3Cache(int num_entries_to_remove) {
    if (!external_storage) {
        return;
    }

    try {
        // 获取最冷的数据
        auto coldest_entries = external_storage->getColdestK(num_entries_to_remove);

        // 移除这些数据
        int removed_count = 0;
        for (uint64_t slot_id: coldest_entries) {
            // 从ExternalStorage中实际删除条目
            if (external_storage->remove(slot_id)) {
                cout << "   Delete L3 cache entry, slot_id: " << slot_id << endl;
                removed_count++;
            } else {
                cerr << "   Failed to delete L3 cache entry, slot_id: " << slot_id << endl;
            }
        }

        cout << "✅ Cleaned up " << removed_count << " L3 cache entries" << endl;
    } catch (const std::exception &e) {
        cerr << "❌ Failed to clean up L3 cache: " << e.what() << endl;
    }
}

// 从API端点加载知识库
bool RAGKnowledgeBaseLoader::loadFromAPI(const std::string &api_endpoint,
                                         const std::string &query,
                                         const std::string &category,
                                         const std::map<std::string, std::string> &params) {
    CURL *curl;
    CURLcode res;
    string response;

    curl = curl_easy_init();
    if (!curl) {
        cerr << "❌ Failed to initialize CURL" << endl;
        return false;
    }

    string full_url = api_endpoint + "?query=" + query;
    for (const auto &[key, value]: params) {
        full_url += "&" + key + "=" + value;
    }

    curl_easy_setopt(curl, CURLOPT_URL, full_url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "RAGKnowledgeLoader/1.0");
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);

    res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        cerr << "❌ API request failed: " << curl_easy_strerror(res) << endl;
        return false;
    }

    // 解析API响应
    KnowledgeEntry entry;
    entry.title = "API query: " + query;
    entry.content = response.substr(0, max_content_length);
    entry.category = category;
    entry.source = api_endpoint;
    entry.relevance_score = calculateRelevance(entry.content, category);
    entry.tags = extractTags(entry.content);

    return addKnowledgeEntry(entry);
}

// Wikipedia响应解析
bool RAGKnowledgeBaseLoader::parseWikipediaResponse(const string &response, KnowledgeEntry &entry) {
    // 简化版Wikipedia解析（实际应该用HTML解析器）
    try {
        // 提取标题
        size_t title_start = response.find("<title>") + 7;
        size_t title_end = response.find("</title>", title_start);
        if (title_start != string::npos && title_end != string::npos) {
            entry.title = response.substr(title_start, title_end - title_start);
            // 移除" - Wikipedia"后缀
            size_t wiki_pos = entry.title.find(" - Wikipedia");
            if (wiki_pos != string::npos) {
                entry.title = entry.title.substr(0, wiki_pos);
            }
        } else {
            entry.title = "Wikipedia category";
        }

        // 提取内容（简化版，只取第一段）
        size_t content_start = response.find("<p>", title_end);
        size_t content_end = response.find("</p>", content_start);
        if (content_start != string::npos && content_end != string::npos) {
            entry.content = response.substr(content_start + 3, content_end - content_start - 3);
            // 移除HTML标签（简化处理）
            regex html_tags("<[^>]*>");
            entry.content = regex_replace(entry.content, html_tags, "");
        } else {
            entry.content = "Wikipedia content";
        }

        entry.category = "wiki knowledge";
        entry.source = "Wikipedia";

        return true;
    } catch (const exception &e) {
        cerr << "⚠️ Wikipedia response parsing failed: " << e.what() << endl;
        return false;
    }
}

// JSON数据解析
bool RAGKnowledgeBaseLoader::parseJSONData(const string &json_data, vector<KnowledgeEntry> &entries) {
    try {
        // 使用nlohmann::json库解析JSON
        auto json = nlohmann::json::parse(json_data);

        // 检查JSON是否为数组格式
        if (json.is_array()) {
            for (const auto &item: json) {
                KnowledgeEntry entry;

                // 从JSON对象中提取数据
                if (item.contains("title")) {
                    entry.title = item["title"].get<string>();
                }

                if (item.contains("content")) {
                    entry.content = item["content"].get<string>();
                }

                if (item.contains("category")) {
                    entry.category = item["category"].get<string>();
                }

                if (item.contains("source")) {
                    entry.source = item["source"].get<string>();
                }

                if (item.contains("relevance_score")) {
                    entry.relevance_score = item["relevance_score"].get<double>();
                }

                if (item.contains("tags") && item["tags"].is_array()) {
                    for (const auto &tag: item["tags"]) {
                        if (tag.is_string()) {
                            entry.tags.push_back(tag.get<string>());
                        }
                    }
                }

                entries.push_back(entry);
            }
        } else if (json.is_object()) {
            // 单个对象的情况
            KnowledgeEntry entry;

            if (json.contains("title")) {
                entry.title = json["title"].get<string>();
            }

            if (json.contains("content")) {
                entry.content = json["content"].get<string>();
            }

            if (json.contains("category")) {
                entry.category = json["category"].get<string>();
            }

            if (json.contains("source")) {
                entry.source = json["source"].get<string>();
            }

            if (json.contains("relevance_score")) {
                entry.relevance_score = json["relevance_score"].get<double>();
            }

            if (json.contains("tags") && json["tags"].is_array()) {
                for (const auto &tag: json["tags"]) {
                    if (tag.is_string()) {
                        entry.tags.push_back(tag.get<string>());
                    }
                }
            }

            entries.push_back(entry);
        }

        return !entries.empty();
        //   {"title": "标题", "content": "内容", "category": "类别"},
        //   ...
        // ]
        size_t start_pos = json_data.find('[');
        size_t end_pos = json_data.rfind(']');

        if (start_pos == string::npos || end_pos == string::npos) {
            cerr << "❌ Invalid JSON format" << endl;
            return false;
        }

        string json_content = json_data.substr(start_pos + 1, end_pos - start_pos - 1);

        // 简单分割对象
        vector<string> objects;
        size_t obj_start = 0;
        size_t obj_end = 0;

        while ((obj_start = json_content.find('{', obj_end)) != string::npos) {
            obj_end = json_content.find('}', obj_start) + 1;
            if (obj_end == string::npos) break;

            string obj_str = json_content.substr(obj_start, obj_end - obj_start);
            objects.push_back(obj_str);
        }

        for (const auto &obj: objects) {
            KnowledgeEntry entry;

            // 提取标题
            size_t title_start = obj.find("\"title\":\"");
            if (title_start != string::npos) {
                title_start += 8; // "title":" 的长度
                size_t title_end = obj.find("\"", title_start);
                if (title_end != string::npos) {
                    entry.title = obj.substr(title_start, title_end - title_start);
                }
            }

            // 提取内容
            size_t content_start = obj.find("\"content\":\"");
            if (content_start != string::npos) {
                content_start += 10; // "content":" 的长度
                size_t content_end = obj.find("\"", content_start);
                if (content_end != string::npos) {
                    entry.content = obj.substr(content_start, content_end - content_start);
                }
            }

            // 提取类别
            size_t category_start = obj.find("\"category\":\"");
            if (category_start != string::npos) {
                category_start += 11; // "category":" 的长度
                size_t category_end = obj.find("\"", category_start);
                if (category_end != string::npos) {
                    entry.category = obj.substr(category_start, category_end - category_start);
                }
            }

            if (!entry.title.empty() && !entry.content.empty()) {
                entry.source = "JSON import";
                entries.push_back(entry);
            }
        }

        return !entries.empty();
    } catch (const exception &e) {
        cerr << "⚠️ JSON data parsing failed: " << e.what() << endl;
        return false;
    }
}

// CSV数据解析
bool RAGKnowledgeBaseLoader::parseCSVData(const string &csv_data, vector<KnowledgeEntry> &entries) {
    try {
        stringstream ss(csv_data);
        string line;
        bool first_line = true;

        while (getline(ss, line)) {
            if (first_line) {
                first_line = false;
                continue; // 跳过标题行
            }

            vector<string> fields;
            stringstream line_ss(line);
            string field;

            while (getline(line_ss, field, ',')) {
                // 移除引号
                if (!field.empty() && field.front() == '"' && field.back() == '"') {
                    field = field.substr(1, field.length() - 2);
                }
                fields.push_back(field);
            }

            if (fields.size() >= 3) {
                KnowledgeEntry entry;
                entry.title = fields[0];
                entry.content = fields[1];
                entry.category = fields[2];
                entry.source = "CSV import";

                entries.push_back(entry);
            }
        }

        return !entries.empty();
    } catch (const exception &e) {
        cerr << "⚠️ CSV data parsing failed: " << e.what() << endl;
        return false;
    }
}

// 从CSV数据加载知识库
bool RAGKnowledgeBaseLoader::loadFromCSV(const string &csv_data, const string &category) {
    vector<KnowledgeEntry> entries;
    if (!parseCSVData(csv_data, entries)) {
        return false;
    }

    bool success = true;
    for (auto &entry: entries) {
        if (!category.empty()) {
            entry.category = category;
        }
        entry.relevance_score = calculateRelevance(entry.content, entry.category);
        entry.tags = extractTags(entry.content);

        if (!addKnowledgeEntry(entry)) {
            success = false;
        }
    }

    return success;
}

// OpenAI API调用
string RAGKnowledgeBaseLoader::callOpenAICompletion(const string &prompt,
                                                    const string &model,
                                                    double temperature) {
    if (!openai_client) {
        cerr << "❌ OpenAI client not initialized" << endl;
        return "";
    }

    try {
        OpenAIClient::ChatCompletionRequest request;
        request.model = model;
        request.messages.push_back(OpenAIClient::ChatMessage("user", prompt));
        request.temperature = temperature;
        request.max_tokens = 200;

        auto response = openai_client->createChatCompletion(request);

        if (!response.choices.empty()) {
            return response.choices[0].message.content;
        }
    } catch (const exception &e) {
        cerr << "⚠️ OpenAI API call failed: " << e.what() << endl;
    }

    return "";
}
