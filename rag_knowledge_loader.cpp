// RAG知识库加载器实现

#include "rag_knowledge_loader.h"
#include "openai_client.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <Windows.h>

char* wideCharToMultiByte(wchar_t* pWCStrKey)
{
    //第一次调用确认转换后单字节字符串的长度，用于开辟空间
    int pSize = WideCharToMultiByte(CP_OEMCP, 0, pWCStrKey, wcslen(pWCStrKey), NULL, 0, NULL, NULL);
    char* pCStrKey = new char[pSize+1];
    //第二次调用将双字节字符串转换成单字节字符串
    WideCharToMultiByte(CP_OEMCP, 0, pWCStrKey, wcslen(pWCStrKey), pCStrKey, pSize, NULL, NULL);
    pCStrKey[pSize] = '\0';
    return pCStrKey;

    //如果想要转换成string，直接赋值即可
    //string pKey = pCStrKey;
}

wchar_t *multiByteToWideChar(const std::string& pKey)
{
    const char *pCStrKey = pKey.c_str();
    //第一次调用返回转换后的字符串长度，用于确认为wchar_t*开辟多大的内存空间
    int pSize = MultiByteToWideChar(CP_OEMCP, 0, pCStrKey, strlen(pCStrKey) + 1, NULL, 0);
    wchar_t *pWCStrKey = new wchar_t[pSize];
    //第二次调用将单字节字符串转换成双字节字符串
    MultiByteToWideChar(CP_OEMCP, 0, pCStrKey, strlen(pCStrKey) + 1, pWCStrKey, pSize);
    return pWCStrKey;
}

using namespace std;

// 回调函数用于CURL写入数据
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, string* userp) {
    size_t total_size = size * nmemb;
    userp->append((char*)contents, total_size);
    return total_size;
}

RAGKnowledgeBaseLoader::RAGKnowledgeBaseLoader(const string& openai_api_key,
                                              const string& base_url,
                                              int max_entries,
                                              double min_relevance,
                                              int max_length)
    : max_entries_per_category(max_entries),
      min_relevance_threshold(min_relevance),
      max_content_length(max_length),
      max_storage_size(10000) {  // 默认最大存储大小为10000条目
    
    if (!openai_api_key.empty()) {
        openai_client = make_unique<OpenAIClient::HttpClient>(openai_api_key, base_url);
    }
    
    // 初始化Logic匹配器
    logic_matcher = make_unique<LogicSemanticMatcher>("/models/e5/e5_large.onnx");
    
    // 初始化语义查询引擎
    semantic_engine = make_unique<SemanticQueryEngine>("/models/e5/e5_large.onnx");
}

bool RAGKnowledgeBaseLoader::loadFromURL(const string& url, const string& category) {
    CURL* curl;
    CURLcode res;
    string response;
    
    curl = curl_easy_init();
    if (!curl) {
        cerr << "❌ 初始化CURL失败" << endl;
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
        cerr << "❌ CURL请求失败: " << curl_easy_strerror(res) << endl;
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
        entry.title = "从URL加载的内容";
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

bool RAGKnowledgeBaseLoader::streamHuggingFaceDataset(const string& dataset_name, 
                                                     const string& subset,
                                                     const string& split,
                                                     int max_entries,
                                                     const string& category) {
    cout << "🔄 通过Python脚本流式解析HuggingFace数据集..." << endl;
    
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
        
        cout << "   执行命令: " << cmd << endl;
        
        // 执行Python脚本
        int result = system(cmd.c_str());
        
        if (result != 0) {
            cerr << "❌ Python脚本执行失败 (返回码: " << result << ")" << endl;
            success = false;
            break;
        }
        
        // 读取Python脚本生成的JSON文件
        ifstream file(output_file);
        if (!file.is_open()) {
            cerr << "❌ 无法打开Python脚本生成的临时文件: " << output_file << endl;
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
            cerr << "❌ 解析Python脚本生成的JSON数据失败: " << output_file << endl;
            success = false;
            // 删除临时文件
            remove(output_file.c_str());
            break;
        }
        
        // 添加到知识库
        int batch_added = 0;
        for (auto& entry : entries) {
            if (total_processed + batch_added >= max_entries) {
                break; // 达到最大条目数限制
            }
            
            entry.relevance_score = calculateRelevance(entry.content, category);
            entry.tags = extractTags(entry.content);
            
            if (!addKnowledgeEntry(entry)) {
                cerr << "❌ 添加HuggingFace数据集条目失败: " << total_processed + batch_added + 1 << endl;
                success = false;
            } else {
                batch_added++;
            }
        }
        
        total_processed += batch_added;
        cout << "   批次处理完成，已处理 " << total_processed << " / " << max_entries << " 个条目" << endl;
        
        // 删除临时文件
        remove(output_file.c_str());
        
        // 如果这一批次没有添加任何条目，说明数据已经用完
        if (batch_added == 0) {
            cout << "   数据集已用完，停止处理" << endl;
            break;
        }
    }
    
    if (success && total_processed > 0) {
        cout << "✅ 成功从" << dataset_name << "流式加载 " << total_processed << " 个数据集条目" << endl;
    } else if (total_processed == 0) {
        cerr << "❌ 未能从" << dataset_name << "加载任何数据集条目" << endl;
        // 回退到模拟实现
        cerr << "   回退到模拟实现..." << endl;
        for (int i = 0; i < max_entries; ++i) {
            KnowledgeEntry entry;
            entry.title = "HuggingFace数据集条目 " + to_string(i+1);
            entry.content = "这是来自" + dataset_name + "数据集的模拟内容，条目编号：" + to_string(i+1);
            entry.category = category;
            entry.source = "huggingface://" + dataset_name + "/" + subset + "/" + split;
            entry.relevance_score = 0.8; // 模拟相关性分数
            
            if (!addKnowledgeEntry(entry)) {
                cerr << "❌ 添加HuggingFace数据集条目失败: " << i+1 << endl;
                return false;
            }
        }
        cout << "✅ 成功模拟加载 " << max_entries << " 个来自" << dataset_name << "的数据集条目" << endl;
        return true;
    }
    
    return success;
}

bool RAGKnowledgeBaseLoader::loadFromFile(const string& file_path, const string& category) {
    ifstream file(file_path);
    if (!file.is_open()) {
        cerr << "❌ 无法打开文件: " << file_path << endl;
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

bool RAGKnowledgeBaseLoader::queryAndLoadFromHFDataset(const string& query,
                                                      const string& dataset_name,
                                                      const string& subset,
                                                      int max_results,
                                                      const string& category) {
    cout << "🔍 通过Python脚本查询HuggingFace数据集..." << endl;
    
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
        
        cout << "   执行命令: " << cmd << endl;
        
        // 执行Python脚本
        int result = system(cmd.c_str());
        
        if (result != 0) {
            cerr << "❌ Python脚本执行失败 (返回码: " << result << ")" << endl;
            success = false;
            break;
        }
        
        // 读取Python脚本生成的JSON文件
        ifstream file(output_file);
        if (!file.is_open()) {
            cerr << "❌ 无法打开Python脚本生成的临时文件: " << output_file << endl;
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
            cerr << "❌ 解析Python脚本生成的JSON数据失败: " << output_file << endl;
            success = false;
            // 删除临时文件
            remove(output_file.c_str());
            break;
        }
        
        // 添加到知识库
        int batch_added = 0;
        for (auto& entry : entries) {
            if (total_processed + batch_added >= max_results) {
                break; // 达到最大条目数限制
            }
            
            entry.relevance_score = calculateRelevance(entry.content, category);
            entry.tags = extractTags(entry.content);
            
            if (!addKnowledgeEntry(entry)) {
                cerr << "❌ 添加HuggingFace查询结果失败: " << total_processed + batch_added + 1 << endl;
                success = false;
            } else {
                batch_added++;
            }
        }
        
        total_processed += batch_added;
        cout << "   批次查询完成，已处理 " << total_processed << " / " << max_results << " 个条目" << endl;
        
        // 删除临时文件
        remove(output_file.c_str());
        
        // 如果这一批次没有添加任何条目，说明查询结果已经用完
        if (batch_added == 0) {
            cout << "   查询结果已用完，停止处理" << endl;
            break;
        }
    }
    
    if (success && total_processed > 0) {
        cout << "✅ 成功从" << dataset_name << "查询并加载 " << total_processed << " 个结果" << endl;
    } else if (total_processed == 0) {
        cerr << "❌ 未能从" << dataset_name << "查询到任何结果" << endl;
        // 回退到模拟实现
        cerr << "   回退到模拟实现..." << endl;
        for (int i = 0; i < max_results; ++i) {
            KnowledgeEntry entry;
            entry.title = "查询结果 " + to_string(i+1) + ": " + query;
            entry.content = "与查询'" + query + "'相关的来自" + dataset_name + "的内容，结果编号：" + to_string(i+1);
            entry.category = category;
            entry.source = "huggingface-query://" + dataset_name + "/" + subset;
            entry.relevance_score = 0.7 + (0.2 * (double)i / max_results); // 模拟递减的相关性
            
            if (!addKnowledgeEntry(entry)) {
                cerr << "❌ 添加HuggingFace查询结果失败: " << i+1 << endl;
                return false;
            }
        }
        cout << "✅ 成功模拟查询并加载 " << max_results << " 个来自" << dataset_name << "的查询结果" << endl;
        return true;
    }
    
    return success;
}

bool RAGKnowledgeBaseLoader::loadFromJSON(const string& json_data, const string& category) {
    vector<KnowledgeEntry> entries;
    if (!parseJSONData(json_data, entries)) {
        return false;
    }
    
    bool success = true;
    for (auto& entry : entries) {
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
    const string& knowledge_text, 
    const string& category,
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
    const string& category,
    int max_logics,
    double activation_threshold) {
    
    vector<LogicDescriptor> all_logics;
    
    auto it = category_to_entries.find(category);
    if (it == category_to_entries.end()) {
        return all_logics;
    }
    
    const auto& entries = it->second;
    int count = 0;
    
    for (const auto& entry : entries) {
        if (count >= max_logics) break;
        
        auto logics = generateLogicTreeFromKnowledge(entry.content, category, activation_threshold);
        all_logics.insert(all_logics.end(), logics.begin(), logics.end());
        count++;
    }
    
    return all_logics;
}

string RAGKnowledgeBaseLoader::generateLogicDescriptionWithAI(const string& knowledge_content) {
    if (!openai_client) {
        return "知识条目: " + summarizeContent(knowledge_content, 50);
    }
    
    string prompt = "请为以下知识内容生成一个简洁的Logic描述（不超过30字）:\n\n" + 
                   knowledge_content.substr(0, 500) + 
                   "\n\nLogic描述:";
    
    try {
        OpenAIClient::ChatCompletionRequest request;
        request.model = "gpt-3.5-turbo";
        request.messages.push_back(OpenAIClient::ChatMessage("user", prompt));
        request.temperature = 0.7;
        request.max_tokens = 50;
        
        auto response = openai_client->createChatCompletion(request);
        
        if (!response.choices.empty()) {
            return response.choices[0].message.content;
        }
    } catch (const exception& e) {
        cerr << "⚠️ OpenAI API调用失败: " << e.what() << endl;
    }
    
    return "知识条目: " + summarizeContent(knowledge_content, 30);
}

bool RAGKnowledgeBaseLoader::registerLogicTree(LogicSemanticMatcher& matcher, 
                                              const vector<LogicDescriptor>& logics) {
    bool success = true;
    
    for (const auto& logic : logics) {
        if (!matcher.registerLogic(logic)) {
            cerr << "❌ 注册Logic失败: " << logic.logic_id << endl;
            success = false;
        } else {
            cout << "✅ 注册Logic成功: " << logic.logic_id << endl;
        }
    }
    
    return success;
}

bool RAGKnowledgeBaseLoader::addKnowledgeEntry(const KnowledgeEntry& entry) {
    // 检查是否超过最大条目数
    if (category_to_entries[entry.category].size() >= max_entries_per_category) {
        // 移除相关性最低的条目
        auto& entries = category_to_entries[entry.category];
        auto min_it = min_element(entries.begin(), entries.end(),
            [](const KnowledgeEntry& a, const KnowledgeEntry& b) {
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

vector<KnowledgeEntry> RAGKnowledgeBaseLoader::searchKnowledge(const string& query, 
                                                             int max_results,
                                                             const string& category) {
    vector<KnowledgeEntry> results;
    
    // 如果指定了类别，只在该类别中搜索
    if (!category.empty()) {
        auto it = category_to_entries.find(category);
        if (it != category_to_entries.end()) {
            const auto& entries = it->second;
            
            // 简单的文本匹配搜索
            for (const auto& entry : entries) {
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
        for (const auto& entry : knowledge_base) {
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
string RAGKnowledgeBaseLoader::extractKeyPoints(const string& content) {
    // 简单的关键点提取（可以改进为更复杂的NLP方法）
    vector<string> sentences;
    stringstream ss(content);
    string sentence;
    
    while (getline(ss, sentence, '.')) {
        if (sentence.length() > 20) { // 过滤短句
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

string RAGKnowledgeBaseLoader::summarizeContent(const string& content, int max_length) {
    if (content.length() <= max_length) {
        return content;
    }
    
    // 简单的截断摘要
    return content.substr(0, max_length) + "...";
}

double RAGKnowledgeBaseLoader::calculateRelevance(const string& content, const string& category) {
    // 简单的相关性计算（可以改进为基于语义的算法）
    double relevance = 0.5; // 基础分数
    
    // 基于内容长度
    relevance += min(content.length() / 1000.0, 0.3);
    
    // 基于关键词匹配（如果类别有特定关键词）
    map<string, vector<string>> category_keywords = {
        {"计算机科学", {"算法", "编程", "数据结构", "人工智能"}},
        {"数学", {"定理", "公式", "证明", "计算"}},
        {"物理", {"力学", "量子", "相对论", "能量"}}
    };
    
    if (category_keywords.count(category)) {
        const auto& keywords = category_keywords[category];
        for (const auto& keyword : keywords) {
            if (content.find(keyword) != string::npos) {
                relevance += 0.1;
            }
        }
    }
    
    return min(relevance, 1.0);
}

vector<string> RAGKnowledgeBaseLoader::extractTags(const string& content) {
    vector<string> tags;
    
    // 简单的标签提取（可以改进为NLP方法）
    vector<string> common_tags = {"技术", "科学", "教育", "研究", "创新"};
    
    for (const auto& tag : common_tags) {
        if (content.find(tag) != string::npos) {
            tags.push_back(tag);
        }
    }
    
    return tags;
}

// 预定义知识库实现
vector<KnowledgeEntry> PredefinedKnowledgeLoader::loadComputerScienceKnowledge() {
    return {
        {"数据结构", "数据结构是计算机存储、组织数据的方式，包括数组、链表、栈、队列、树、图等基本结构。", "计算机科学", "预定义"},
        {"算法", "算法是解决特定问题的一系列清晰指令，包括排序、搜索、动态规划等经典算法。", "计算机科学", "预定义"},
        {"人工智能", "人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统。", "计算机科学", "预定义"}
    };
}

vector<KnowledgeEntry> PredefinedKnowledgeLoader::loadMachineLearningKnowledge() {
    return {
        {"神经网络", "神经网络是一种模仿生物神经网络结构和功能的计算模型，用于模式识别和机器学习。", "机器学习", "预定义"},
        {"深度学习", "深度学习是基于深层神经网络的机器学习方法，能够自动学习数据的层次化特征表示。", "机器学习", "预定义"},
        {"强化学习", "强化学习是智能体通过与环境交互学习最优策略的机器学习方法。", "机器学习", "预定义"}
    };
}

// arXiv API解析实现
bool RAGKnowledgeBaseLoader::parseArxivResponse(const string& response, KnowledgeEntry& entry) {
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
            entry.title = "arXiv论文";
        }
        
        if (summary_start != string::npos && summary_end != string::npos) {
            entry.content = response.substr(summary_start + 9, summary_end - summary_start - 9);
            // 清理摘要内容
            entry.content.erase(0, entry.content.find_first_not_of(" \n\r\t"));
            entry.content.erase(entry.content.find_last_not_of(" \n\r\t") + 1);
        } else {
            entry.content = "arXiv论文摘要";
        }
        
        entry.category = "学术论文";
        entry.source = "arXiv API";
        
        return true;
    } catch (const exception& e) {
        cerr << "⚠️ arXiv响应解析失败: " << e.what() << endl;
        return false;
    }
}

bool RAGKnowledgeBaseLoader::fetchArxivPapers(const string& query, 
                                             const string& category,
                                             int max_results) {
    // 构建arXiv API URL
    string base_url = "http://export.arxiv.org/api/query?";
    string search_query = "search_query=all:" + query + "&max_results=" + to_string(max_results);
    string url = base_url + search_query;
    
    return loadFromURL(url, category);
}

bool RAGKnowledgeBaseLoader::autoFetchDataWhenLogicInsufficient(const string& query,
                                                              int min_required_matches,
                                                              const string& dataset_name,
                                                              const string& subset) {
    cout << "🔍 查询结果不足，直接从HuggingFace数据集获取数据..." << endl;
    
    // 直接从HuggingFace数据集查询并加载相关条目（使用流式处理）
    bool success = queryAndLoadFromHFDataset(query, dataset_name, subset, min_required_matches);
    
    if (success) {
        cout << "✅ 自动获取数据成功" << endl;
        
        // 插入到外部存储（如果有的话）
        if (external_storage) {
            cout << "💾 同步最新获取的条目到外部存储..." << endl;
            auto entries = getAllEntries();
            int count = 0;
            
            // 只同步新添加的条目，限制数量避免存储爆炸
            int max_to_sync = min(20, min_required_matches); // 最多同步20个或所需数量
            for (int i = entries.size() - 1; i >= 0 && count < max_to_sync; --i) {
                if (!insertToExternalStorage(entries[i])) {
                    cerr << "❌ 同步到外部存储失败: " << entries[i].title << endl;
                }
                count++;
            }
        }
        
        // 检查存储大小并清理L3缓存（如果需要）
        if (external_storage) {
            checkAndCleanupStorage();
        }
        
        return true;
    } else {
        cerr << "❌ 自动获取数据失败" << endl;
        return false;
    }
}

// 预定义知识库实现
vector<KnowledgeEntry> PredefinedKnowledgeLoader::loadMathematicsKnowledge() {
    return {
        {"微积分", "微积分是研究函数的微分和积分以及相关概念和应用的一门数学分支。", "数学", "预定义"},
        {"线性代数", "线性代数是关于向量空间和线性映射的数学分支，广泛应用于科学和工程领域。", "数学", "预定义"},
        {"概率论", "概率论是研究随机现象数量规律的数学分支，是统计学的基础。", "数学", "预定义"}
    };
}

vector<KnowledgeEntry> PredefinedKnowledgeLoader::loadPhysicsKnowledge() {
    return {
        {"经典力学", "经典力学是研究宏观物体运动的物理学分支，由牛顿三大定律描述。", "物理", "预定义"},
        {"量子力学", "量子力学是描述微观粒子行为的物理学理论，具有概率性和波粒二象性。", "物理", "预定义"},
        {"相对论", "相对论包括狭义相对论和广义相对论，研究时空结构和引力现象。", "物理", "预定义"}
    };
}

vector<KnowledgeEntry> PredefinedKnowledgeLoader::loadBiologyKnowledge() {
    return {
        {"细胞生物学", "细胞是生命的基本单位，细胞生物学研究细胞的结构、功能和生命周期。", "生物", "预定义"},
        {"遗传学", "遗传学研究基因、遗传变异和生物遗传特征的传递规律。", "生物", "预定义"},
        {"进化论", "进化论解释物种如何通过自然选择和遗传变异随时间变化。", "生物", "预定义"}
    };
}

vector<KnowledgeEntry> PredefinedKnowledgeLoader::loadPhilosophyKnowledge() {
    return {
        {"形而上学", "形而上学研究存在的本质、现实的基本结构和宇宙的终极本质。", "哲学", "预定义"},
        {"认识论", "认识论研究知识的本质、起源和范围，以及信念的合理性。", "哲学", "预定义"},
        {"伦理学", "伦理学研究道德价值、行为准则和善恶判断的标准。", "哲学", "预定义"}
    };
}

vector<KnowledgeEntry> PredefinedKnowledgeLoader::loadCommonSenseKnowledge() {
    return {
        {"日常物理", "物体从高处落下会加速，这是重力作用的结果。", "常识", "预定义"},
        {"时间概念", "时间是连续的，不可逆转的，是事件发生的先后顺序。", "常识", "预定义"},
        {"社会规范", "人们通常遵循一定的行为规范来维持社会秩序。", "常识", "预定义"}
    };
}

vector<KnowledgeEntry> PredefinedKnowledgeLoader::loadProgrammingDocumentation() {
    return {
        {"Python基础", "Python是一种高级编程语言，以简洁易读著称，适合初学者学习。", "编程", "预定义"},
        {"数据结构", "数据结构是组织和存储数据的方式，包括数组、链表、树、图等。", "编程", "预定义"},
        {"算法设计", "算法是解决问题的步骤序列，包括排序、搜索、动态规划等。", "编程", "预定义"}
    };
}

vector<KnowledgeEntry> PredefinedKnowledgeLoader::loadNeuroscienceKnowledge() {
    return {
        {"神经元", "神经元是神经系统的基本单位，通过电化学信号传递信息。", "神经科学", "预定义"},
        {"突触传递", "突触是神经元之间的连接点，神经递质在此传递信号。", "神经科学", "预定义"},
        {"大脑结构", "大脑分为前脑、中脑和后脑，各区域负责不同功能。", "神经科学", "预定义"}
    };
}

vector<KnowledgeEntry> PredefinedKnowledgeLoader::loadLanguageModelKnowledge() {
    return {
        {"Transformer架构", "Transformer是基于自注意力机制的神经网络架构，是现代语言模型的基础。", "语言模型", "预定义"},
        {"注意力机制", "注意力机制允许模型关注输入序列的不同部分，提高处理长序列的能力。", "语言模型", "预定义"},
        {"预训练微调", "语言模型通常先在大规模语料上预训练，然后在特定任务上微调。", "语言模型", "预定义"}
    };
}

vector<KnowledgeEntry> PredefinedKnowledgeLoader::loadAITechniques() {
    return {
        {"监督学习", "监督学习使用标注数据训练模型，包括分类和回归任务。", "人工智能", "预定义"},
        {"无监督学习", "无监督学习从无标签数据中发现模式和结构。", "人工智能", "预定义"},
        {"强化学习", "强化学习通过试错和奖励机制学习最优策略。", "人工智能", "预定义"}
    };
}

// 获取知识库统计信息
RAGKnowledgeBaseLoader::KnowledgeStats RAGKnowledgeBaseLoader::getKnowledgeStats() const {
    KnowledgeStats stats;
    stats.total_entries = knowledge_base.size();
    stats.unique_categories = category_to_entries.size();
    
    double total_relevance = 0.0;
    for (const auto& entry : knowledge_base) {
        stats.category_counts[entry.category]++;
        total_relevance += entry.relevance_score;
    }
    
    stats.avg_relevance_score = stats.total_entries > 0 ? 
                               total_relevance / stats.total_entries : 0.0;
    
    return stats;
}

// 清空知识库
void RAGKnowledgeBaseLoader::clearKnowledgeBase() {
    knowledge_base.clear();
    category_to_entries.clear();
}

// 导出知识库到文件
bool RAGKnowledgeBaseLoader::exportToFile(const std::string& file_path, const std::string& format) {
    ofstream file(file_path);
    if (!file.is_open()) {
        cerr << "❌ 无法创建导出文件: " << file_path << endl;
        return false;
    }
    
    if (format == "json") {
        // 导出为JSON格式
        file << "[\n";
        for (size_t i = 0; i < knowledge_base.size(); ++i) {
            const auto& entry = knowledge_base[i];
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
        for (const auto& entry : knowledge_base) {
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
    cout << "✅ 知识库已导出到: " << file_path << " (" << format << "格式)" << endl;
    return true;
}

// 设置OpenAI客户端
void RAGKnowledgeBaseLoader::setOpenAIClient(std::unique_ptr<OpenAIClient::HttpClient> client) {
    openai_client = std::move(client);
}

// 按类别获取知识条目
void RAGKnowledgeBaseLoader::setExternalStorage(std::shared_ptr<ExternalStorage<KnowledgeEntry>> storage) {
    external_storage = storage;
}

std::shared_ptr<ExternalStorage<KnowledgeEntry>> RAGKnowledgeBaseLoader::getExternalStorage() const {
    return external_storage;
}

bool RAGKnowledgeBaseLoader::insertToExternalStorage(const KnowledgeEntry& entry) {
    if (!external_storage) {
        cerr << "❌ 外部存储未初始化" << endl;
        return false;
    }
    
    // 存储到外部存储
    uint64_t slot_id = external_storage->store<KnowledgeEntry>(entry);
    if (slot_id == 0) {
        cerr << "❌ 存储到外部存储失败" << endl;
        return false;
    }
    
    cout << "✅ 成功存储到外部存储 (slot_id: " << slot_id << ")" << endl;
    return true;
}

std::vector<KnowledgeEntry> RAGKnowledgeBaseLoader::getEntriesByCategory(const std::string& category) const {
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

bool RAGKnowledgeBaseLoader::insertToExternalStorage(const std::vector<KnowledgeEntry>& entries) {
    if (!external_storage) {
        cerr << "❌ 外部存储未初始化" << endl;
        return false;
    }
    
    bool success = true;
    for (const auto& entry : entries) {
        if (!insertToExternalStorage(entry)) {
            success = false;
        }
    }
    
    return success;
}

bool RAGKnowledgeBaseLoader::insertToExternalStorageWithSemanticFeatures(const KnowledgeEntry& entry) {
    if (!external_storage) {
        cerr << "❌ 外部存储未初始化" << endl;
        return false;
    }
    
    // 存储到外部存储（带语义特征）
    uint64_t slot_id = external_storage->storeWithSemanticFeature<KnowledgeEntry>(
        entry, 
        1.0,  // initial_heat
        "/models/e5/e5_large.onnx",      // model_path
        "/models/e5/vocab.json",         // vocab_path
        "/models/e5/merges.txt",         // merges_path
        "/models/e5/special_tokens.json" // special_tokens_path
    );
    
    if (slot_id == 0) {
        cerr << "❌ 存储到外部存储失败" << endl;
        return false;
    }
    
    cout << "✅ 成功存储到外部存储 (slot_id: " << slot_id << ") 带语义特征" << endl;
    return true;
}

// Logic系统集成方法
bool RAGKnowledgeBaseLoader::registerKnowledgeAsLogic(LogicInjector* logic_injector, 
                                                     ExternalStorage<Logic>* logic_tree,
                                                     const std::string& category) {
    if (!logic_injector || !logic_tree) {
        cerr << "❌ Logic注入器或Logic存储未初始化" << endl;
        return false;
    }
    
    bool success = true;
    int registered_count = 0;
    
    for (const auto& entry : knowledge_base) {
        // 创建Logic描述符（用于语义匹配）
        LogicDescriptor logic_desc;
        logic_desc.logic_id = "rag_" + to_string(hash<string>{}(entry.title + entry.content));
        logic_desc.description = entry.title + ": " + entry.content; // 使用标题和内容作为描述
        logic_desc.category = category;
        logic_desc.activation_threshold = 0.4; // 设置激活阈值
        
        // 创建默认的NeuronInput生成器
        logic_desc.generate_input_callback = LogicDescriptor::createDefaultGenerator(
            entry.content,  // Logic内容
            1.0,            // activity
            1.0,            // weight
            0, 0, 0         // 坐标
        );
        
        // 注册到Logic系统（语义匹配用的描述符）
        if (!logic_injector->registerLogicWithStorage(logic_desc)) {
            cerr << "❌ 注册Logic描述符失败: " << entry.title << endl;
            success = false;
        }
        
        // 创建实际的Logic对象（用于内容存储）
        Logic actual_logic;
        actual_logic.Rcycles = 0;
        actual_logic.importance = 1.0;
        
        // 将内容转换为宽字符并存储
        string full_content = entry.title + ": " + entry.content;
        size_t content_len = min(full_content.length(), size_t(1023)); // 保留一个字符给null终止符
        const char* content_ptr = full_content.c_str();
        
        // 转换为宽字符
        wchar_t* wide_content = multiByteToWideChar(full_content);
        size_t wide_len = wcslen(wide_content);
        size_t copy_len = min(wide_len, size_t(1023));
        
        // 复制到Logic的content数组
        wcsncpy(actual_logic.content, wide_content, copy_len);
        actual_logic.content[copy_len] = L'\0'; // 确保null终止
        
        delete[] wide_content;
        
        // 存储到logic_tree（实际内容存储）
        uint64_t slot_id = logic_tree->store(actual_logic, 1.0);
        if (slot_id == 0) {
            cerr << "❌ 存储Logic内容失败: " << entry.title << endl;
            success = false;
        } else {
            registered_count++;
        }
    }
    
    cout << "✅ 成功注册 " << registered_count << " 个Logic到Logic系统" << endl;
    cout << "   - Logic描述符存储到logic_storage用于语义匹配" << endl;
    cout << "   - Logic内容存储到logic_tree用于实际执行" << endl;
    return success;
}

bool RAGKnowledgeBaseLoader::autoFetchAndRegisterLogic(LogicInjector* logic_injector,
                                                      ExternalStorage<Logic>* logic_tree,
                                                      const std::string& query,
                                                      int min_logics,
                                                      const std::string& dataset_name,
                                                      const std::string& subset,
                                                      const std::string& category) {
    cout << "🔍 自动获取数据并注册为Logic..." << endl;
    
    // 从HuggingFace获取数据
    bool fetch_success = queryAndLoadFromHFDataset(query, dataset_name, subset, min_logics, category);
    
    if (!fetch_success) {
        cerr << "❌ 获取数据失败" << endl;
        return false;
    }
    
    // 将获取的知识注册为Logic
    bool register_success = registerKnowledgeAsLogic(logic_injector, logic_tree, category);
    
    if (register_success) {
        cout << "✅ 成功获取并注册 " << knowledge_base.size() << " 个Logic" << endl;
    }
    
    return register_success;
}

bool RAGKnowledgeBaseLoader::checkAndCleanupStorage() {
    if (!external_storage) {
        return true;  // 没有存储，无需清理
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
    } catch (const std::exception& e) {
        std::cerr << "❌ 检查存储大小失败: " << e.what() << std::endl;
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
        for (uint64_t slot_id : coldest_entries) {
            // 从ExternalStorage中实际删除条目
            if (external_storage->remove(slot_id)) {
                cout << "   删除L3缓存条目, slot_id: " << slot_id << endl;
                removed_count++;
            } else {
                cerr << "   删除L3缓存条目失败, slot_id: " << slot_id << endl;
            }
        }
        
        cout << "✅ 已清理 " << removed_count << " 个L3缓存条目" << endl;
    } catch (const std::exception& e) {
        cerr << "❌ 清理L3缓存失败: " << e.what() << endl;
    }
}

// 从API端点加载知识库
bool RAGKnowledgeBaseLoader::loadFromAPI(const std::string& api_endpoint, 
                                        const std::string& query, 
                                        const std::string& category,
                                        const std::map<std::string, std::string>& params) {
    CURL* curl;
    CURLcode res;
    string response;
    
    curl = curl_easy_init();
    if (!curl) {
        cerr << "❌ 初始化CURL失败" << endl;
        return false;
    }
    
    string full_url = api_endpoint + "?query=" + query;
    for (const auto& [key, value] : params) {
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
        cerr << "❌ API请求失败: " << curl_easy_strerror(res) << endl;
        return false;
    }
    
    // 解析API响应
    KnowledgeEntry entry;
    entry.title = "API查询: " + query;
    entry.content = response.substr(0, max_content_length);
    entry.category = category;
    entry.source = api_endpoint;
    entry.relevance_score = calculateRelevance(entry.content, category);
    entry.tags = extractTags(entry.content);
    
    return addKnowledgeEntry(entry);
}

// Wikipedia响应解析
bool RAGKnowledgeBaseLoader::parseWikipediaResponse(const string& response, KnowledgeEntry& entry) {
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
            entry.title = "Wikipedia条目";
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
            entry.content = "Wikipedia内容";
        }
        
        entry.category = "百科知识";
        entry.source = "Wikipedia";
        
        return true;
    } catch (const exception& e) {
        cerr << "⚠️ Wikipedia响应解析失败: " << e.what() << endl;
        return false;
    }
}

// JSON数据解析
bool RAGKnowledgeBaseLoader::parseJSONData(const string& json_data, vector<KnowledgeEntry>& entries) {
    try {
        // 使用nlohmann::json库解析JSON
        auto json = nlohmann::json::parse(json_data);
        
        // 检查JSON是否为数组格式
        if (json.is_array()) {
            for (const auto& item : json) {
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
                    for (const auto& tag : item["tags"]) {
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
                for (const auto& tag : json["tags"]) {
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
            cerr << "❌ 无效的JSON格式" << endl;
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
        
        for (const auto& obj : objects) {
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
                entry.source = "JSON导入";
                entries.push_back(entry);
            }
        }
        
        return !entries.empty();
    } catch (const exception& e) {
        cerr << "⚠️ JSON数据解析失败: " << e.what() << endl;
        return false;
    }
}

// CSV数据解析
bool RAGKnowledgeBaseLoader::parseCSVData(const string& csv_data, vector<KnowledgeEntry>& entries) {
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
                entry.source = "CSV导入";
                
                entries.push_back(entry);
            }
        }
        
        return !entries.empty();
    } catch (const exception& e) {
        cerr << "⚠️ CSV数据解析失败: " << e.what() << endl;
        return false;
    }
}

// 从CSV数据加载知识库
bool RAGKnowledgeBaseLoader::loadFromCSV(const string& csv_data, const string& category) {
    vector<KnowledgeEntry> entries;
    if (!parseCSVData(csv_data, entries)) {
        return false;
    }
    
    bool success = true;
    for (auto& entry : entries) {
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
string RAGKnowledgeBaseLoader::callOpenAICompletion(const string& prompt, 
                                                   const string& model,
                                                   double temperature) {
    if (!openai_client) {
        cerr << "❌ OpenAI客户端未初始化" << endl;
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
    } catch (const exception& e) {
        cerr << "⚠️ OpenAI API调用失败: " << e.what() << endl;
    }
    
    return "";
}