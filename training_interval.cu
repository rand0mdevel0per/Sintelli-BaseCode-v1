//

// Created by ASUS on 10/8/2025.

//


#include <iostream>
#include "openai_client.h"
#include <vector>
#include <thread>
#include "NeuronModel.cu"
#include <string>
#include <utility>
#include <algorithm>
#include <Python.h>
#include <curl/curl.h>
#include "nlohmann//json.hpp"


using namespace std;


// CURL Global Initialization

static bool curl_initialized = false;

static void init_curl() {
    if (!curl_initialized) {
        curl_global_init(CURL_GLOBAL_DEFAULT);

        curl_initialized = true;
    }
}


std::string runPythonCode(const std::string &code) {
    // Initialize Python (if not already initialized)
    static bool python_initialized = false;
    if (!python_initialized) {
        Py_Initialize();
        python_initialized = true;
    }

    // 初始化CURL
    init_curl();

    if (!Py_IsInitialized()) {
        return "Failed to initialize Python";
    }

    // Capture stdout
    PyRun_SimpleString(
        "import sys\n"
        "from io import StringIO\n"
        "old_stdout = sys.stdout\n"
        "sys.stdout = mystdout = StringIO()\n"
    );

    // Execute code
    int result = PyRun_SimpleString(code.c_str());

    // Get output
    PyRun_SimpleString(
        "sys.stdout = old_stdout\n"
        "output = mystdout.getvalue()\n"
    );

    if (result != 0) {
        PyErr_Print();
        return "Error executing Python code";
    }

    // Get output string
    PyObject *main_module = PyImport_AddModule("__main__");
    PyObject *main_dict = PyModule_GetDict(main_module);
    PyObject *output_obj = PyDict_GetItemString(main_dict, "output");

    std::string output = "";
    if (output_obj && PyUnicode_Check(output_obj)) {
        const char *output_str = PyUnicode_AsUTF8(output_obj);
        if (output_str) {
            output = std::string(output_str);
        }
    }

    return output;
}

// Mathematical calculation function using sympy for symbolic computation
std::string calculateMathExpression(const std::string &expression) {
    // Ensure sympy is installed
    std::string install_code =
            "#Python code"
            "import subprocess\n"
            "import sys\n"
            "try:\n"
            "    import sympy\n"
            "    print(\"Sympy already installed\")\n"
            "except ImportError:\n"
            "    print(\"Installing sympy...\")\n"
            "    subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"sympy\"])\n"
            "    print(\"Sympy installed successfully\")\n"
            ")\n";

    // Execute installation code (pip will automatically skip if already installed)
    std::string install_result = runPythonCode(install_code);

    // Mathematical calculation code
    std::string math_code =
            "#Python code"
            "import sys\n"
            "import math\n"
            "import sympy as sp\n"
            "from sympy import symbols, simplify, diff, integrate, solve, latex\n"
            "from sympy.parsing.latex import parse_latex\n"
            "from sympy.parsing.sympy_parser import parse_expr\n"
            "\n"
            "try:\n"
            "    expr_str = ')\" + expression + R\"('\n"
            "    \n"
            "    # 尝试解析LaTeX格式\n"
            "    if '\\' in expr_str or '{' in expr_str:\n"
            "        try:\n"
            "            expr = parse_latex(expr_str)\n"
            "        except:\n"
            "            expr = parse_expr(expr_str)\n"
            "    else:\n"
            "        # 尝试解析普通表达式\n"
            "        try:\n"
            "            expr = parse_expr(expr_str)\n"
            "        except:\n"
            "            # 如果解析失败，尝试直接计算\n"
            "            allowed_names = {\n"
            "                k: v for k, v in math.__dict__.items() if not k.startswith(\"__\")\n"
            "            }\n"
            "            allowed_names.update({\n"
            "                \"abs\": abs, \"round\": round, \"pow\": pow, \"max\": max, \"min\": min,\n"
            "                \"sqrt\": math.sqrt, \"sin\": math.sin, \"cos\": math.cos, \"tan\": math.tan,\n"
            "                \"log\": math.log, \"exp\": math.exp, \"pi\": math.pi, \"e\": math.e,\n"
            "                \"asin\": math.asin, \"acos\": math.acos, \"atan\": math.atan,\n"
            "                \"sinh\": math.sinh, \"cosh\": math.cosh, \"tanh\": math.tanh,\n"
            "                \"ceil\": math.ceil, \"floor\": math.floor\n"
            "            })\n"
            "            result = eval(expr_str, {\"__builtins__\": {}}, allowed_names)\n"
            "            print(\"Result:\", result)\n"
            "            sys.exit(0)\n"
            "    \n"
            "    # 如果是纯数值表达式，直接计算\n"
            "    if expr.is_number:\n"
            "        result = expr.evalf()\n"
            "        print(\"Result:\", float(result))\n"
            "    else:\n"
            "        # 尝试简化表达式\n"
            "        simplified = sp.simplify(expr)\n"
            "        if simplified.is_number:\n"
            "            result = simplified.evalf()\n"
            "            print(\"Result:\", float(result))\n"
            "        else:\n"
            "            # 返回简化后的表达式\n"
            "            print(\"Simplified expression:\", str(simplified))\n"
            "            # 如果可能，计算数值结果\n"
            "            try:\n"
            "                numeric_result = simplified.evalf()\n"
            "                print(\"Numeric result:\", float(numeric_result))\n"
            "            except:\n"
            "                pass\n"
            "                \n"
            "except Exception as e:\n"
            "    print(\"Error:\", str(e))\n"
            "    # 回退到基本计算\n"
            "    try:\n"
            "        import math\n"
            "        allowed_names = {\n"
            "            k: v for k, v in math.__dict__.items() if not k.startswith(\"__\")\n"
            "        }\n"
            "        allowed_names.update({\n"
            "            \"abs\": abs, \"round\": round, \"pow\": pow, \"max\": max, \"min\": min,\n"
            "            \"sqrt\": math.sqrt, \"sin\": math.sin, \"cos\": math.cos, \"tan\": math.tan,\n"
            "            \"log\": math.log, \"exp\": math.exp, \"pi\": math.pi, \"e\": math.e\n"
            "        })\n"
            "        result = eval(')\" + expression + R\"(', {\"__builtins__\": {}}, allowed_names)\n"
            "        print(\"Result:\", result)\n"
            "    except Exception as e2:\n"
            "        print(\"Error in fallback calculation:\", str(e2))\n"
            ")\n";

    // Execute mathematical calculation code
    std::string result = runPythonCode(math_code);
    return result;
}

// Perform Bing search
std::string performBingSearch(const std::string &query, const std::string &apiKey) {
    // 初始化CURL
    init_curl();

    CURL *curl;
    CURLcode res;
    std::string readBuffer;

    curl = curl_easy_init();
    if (curl) {
        std::string url = "https://api.bing.microsoft.com/v7.0/search?q=" + query;

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        // 设置请求头
        struct curl_slist *headers = NULL;
        std::string auth_header = "Ocp-Apim-Subscription-Key: " + apiKey;
        headers = curl_slist_append(headers, auth_header.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        // 执行请求
        res = curl_easy_perform(curl);

        // 清理
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }

    return readBuffer;
}

// Perform Google search (via custom search engine)
std::string performGoogleSearch(const std::string &query, const std::string &apiKey, const std::string &engineId) {
    // 初始化CURL
    init_curl();

    CURL *curl;
    CURLcode res;
    std::string readBuffer;

    curl = curl_easy_init();
    if (curl) {
        std::string url = "https://www.googleapis.com/customsearch/v1?key=" + apiKey + "&cx=" + engineId + "&q=" +
                          query;

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        // 执行请求
        res = curl_easy_perform(curl);

        // 清理
        curl_easy_cleanup(curl);
    }

    return readBuffer;
}

// Perform ArXiv search
std::string performArxivSearch(const std::string &query) {
    // 初始化CURL
    init_curl();

    CURL *curl;
    CURLcode res;
    std::string readBuffer;

    curl = curl_easy_init();
    if (curl) {
        std::string url = "http://export.arxiv.org/api/query?search_query=" + query + "&start=0&max_results=10";

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        // 执行请求
        res = curl_easy_perform(curl);

        // 清理
        curl_easy_cleanup(curl);
    }

    return readBuffer;
}

// No API key required!
std::string performDuckDuckGoSearch(const std::string &query) {
    CURL *curl = curl_easy_init();
    std::string url = "https://api.duckduckgo.com/?q=" +
                      query + "&format=json";

    std::string readBuffer;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

    // 执行请求
    auto res = curl_easy_perform(curl);

    return readBuffer;
}

// 综合搜索函数
std::string performSearch(const std::string &query, std::string bing, std::string google, std::string google_eng) {
    const std::string BING_API_KEY = bing;
    const std::string GOOGLE_API_KEY = google;
    const std::string GOOGLE_ENGINE_ID = google_eng;

    std::string result = "";

    // 先尝试学术搜索（ArXiv）
    try {
        std::string arxiv_result = performArxivSearch(query);
        if (!arxiv_result.empty()) {
            result += "ArXiv search result:\n" + arxiv_result + "\n\n";
        }
    } catch (...) {
        result += "ArXiv search failed\n\n";
    }

    // 然后尝试Bing搜索
    if (!bing.empty()) {
        try {
            std::string bing_result = performBingSearch(query, BING_API_KEY);
            if (!bing_result.empty()) {
                result += "Bing search result:\n" + bing_result + "\n\n";
            }
        } catch (...) {
            result += "Bing search failed\n\n";
        }
    }

    try {
        std::string ddg_result = performDuckDuckGoSearch(query);
        if (!ddg_result.empty()) {
            result += "DuckDuckGo search result:\n" + ddg_result + "\n\n";
        }
    } catch (...) {
        result += "DuckDuckGo search failed\n\n";
    }

    // 最后尝试Google搜索
    if (!(google.empty() || google_eng.empty())) {
        try {
            std::string google_result = performGoogleSearch(query, GOOGLE_API_KEY, GOOGLE_ENGINE_ID);
            if (!google_result.empty()) {
                result += "Google search result:\n" + google_result + "\n\n";
            }
        } catch (...) {
            result += "Google search failed\n\n";
        }
    }

    return result.empty() ? "search failed.no result available" : result;
}

std::pair<std::string, std::string> parseToolCall(const std::string &input) {
    // Convert to lowercase for case-insensitive search
    std::string lower_input = input;
    std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);

    // Define markers (also converted to lowercase)
    std::string tool_begin = "<tool_begin>";
    std::string tool_id_beg = "[tool_id_beg]";
    std::string tool_id_end = "[tool_id_end]";
    std::string tool_content_beg = "[tool_content_beg]";
    std::string tool_content_end = "[tool_content_end]";
    std::string tool_end = "<tool_end>";

    // Find the start position of tool call
    size_t begin_pos = lower_input.find(tool_begin);
    if (begin_pos == std::string::npos) {
        return std::make_pair("", ""); // Tool call start marker not found
    }

    // Find the end position of tool call
    size_t end_pos = lower_input.find(tool_end, begin_pos);
    if (end_pos == std::string::npos) {
        return std::make_pair("", ""); // Tool call end marker not found
    }

    // Extract the complete tool call section
    std::string tool_call_section = input.substr(begin_pos, end_pos - begin_pos + tool_end.length());
    std::string lower_section = lower_input.substr(begin_pos, end_pos - begin_pos + tool_end.length());

    // Find positions of tool_id_beg and tool_id_end
    size_t id_beg_pos = lower_section.find(tool_id_beg);
    if (id_beg_pos == std::string::npos) {
        return std::make_pair("", ""); // tool_id start marker not found
    }

    size_t id_start = id_beg_pos + tool_id_beg.length();
    size_t id_end_pos = lower_section.find(tool_id_end, id_start);
    if (id_end_pos == std::string::npos) {
        return std::make_pair("", ""); // tool_id end marker not found
    }

    // Extract tool_id
    std::string tool_id = tool_call_section.substr(id_start, id_end_pos - id_start);

    // Find positions of tool_content_beg and tool_content_end
    size_t content_beg_pos = lower_section.find(tool_content_beg);
    if (content_beg_pos == std::string::npos) {
        return std::make_pair("", ""); // tool_content start marker not found
    }

    size_t content_start = content_beg_pos + tool_content_beg.length();
    size_t content_end_pos = lower_section.find(tool_content_end, content_start);
    if (content_end_pos == std::string::npos) {
        return std::make_pair("", ""); // tool_content end marker not found
    }

    // Extract tool_content
    std::string tool_content = tool_call_section.substr(content_start, content_end_pos - content_start);

    // Remove leading and trailing whitespace
    tool_id.erase(0, tool_id.find_first_not_of(" \t\n\r"));
    tool_id.erase(tool_id.find_last_not_of(" \t\n\r") + 1);
    tool_content.erase(0, tool_content.find_first_not_of(" \t\n\r"));
    tool_content.erase(tool_content.find_last_not_of(" \t\n\r") + 1);

    return std::make_pair(tool_id, tool_content);
}

std::string streamPerformingOpenAIAPI(OpenAIClient::HttpClient &client, OpenAIClient::ChatCompletionRequest &req) {
    std::string response;
    bool cont = true;
    client.createChatCompletionStream(req, [&response, &cont](const OpenAIClient::ChatCompletionResponse &resp) {
        cont = !resp.choices.empty();
        if (!cont) return;
        response.append(resp.choices[0].delta);
    });
    while (cont) {
    }
    return response;
}

struct reply_trc {
    float score;
    std::string reply;
    bool is_a_checkpoint;
    bool test_mode;
    bool need_rollback;
    float confidence;
    std::string reasoning;

    static void to_json(nlohmann::json &j, const reply_trc &t) {
        j = json{
            {"score", t.score},
            {"reply", t.reply},
            {"is_a_checkpoint", t.is_a_checkpoint},
            {"test_mode", t.test_mode},
            {"need_rollback", t.need_rollback},
            {"confidence", t.confidence},
            {"reasoning", t.reasoning}
        };
    }

    static void from_json(const nlohmann::json &j, reply_trc &t) {
        try {
            j.at("score").get_to(t.score);
            j.at("reply").get_to(t.reply);
            j.at("is_a_checkpoint").get_to(t.is_a_checkpoint);
            j.at("test_mode").get_to(t.test_mode);
            j.at("need_rollback").get_to(t.need_rollback);
            j.at("confidence").get_to(t.confidence);
            j.at("reasoning").get_to(t.reasoning);
        } catch (...) {
            cerr << "Parse failed!" << endl;
        }
    }
};

void run_training(NeuronModel *model, const std::string &api_key, const std::string &name = "Sydney",
                  std::string bing_api_key = "", std::string google_api_key = "", std::string google_engine_id = "", bool *stop = nullptr) {
    if (stop == nullptr) {
        return;
    }
    std::vector<OpenAIClient::ChatMessage> msgs;

    // 创建系统消息，支持多模态输入
    std::string system_prompt =
            "You are responsible for training and evaluating an advanced AI assistant named " + name +
            ". Your primary role is to provide high-quality responses and accurate assessments of the model's development progress.\n"
            "\n"
            "RESPONSE FORMAT:\n"
            "You MUST respond in JSON format only(except tool calls):\n"
            "{\n"
            "  \"score\": 7.5,                    // Overall quality score (1.0-10.0)\n"
            "  \"reply\": \"Your response text here\",\n"
            "  \"is_a_checkpoint\": false,        // Save model checkpoint if true\n"
            "  \"test_mode\": false,              // Enable testing phase if true\n"
            "  \"need_rollback\": false,          // Request model rollback if true\n"
            "  \"confidence\": 0.85,              // Your confidence in this assessment (0.0-1.0)\n"
            "  \"reasoning\": \"Brief explanation of scoring\"\n"
            "}\n"
            "\n"
            "SCORING FRAMEWORK (1.0-10.0):\n"
            "CRITICAL DIMENSIONS (Weight: 60%):\n"
            "• Factual Accuracy (15%) - Information correctness and absence of hallucinations\n"
            "• Logical Reasoning (15%) - Sound logic, valid conclusions, step-by-step thinking\n"
            "• Context Understanding (10%) - Grasp of conversation context and user intent\n"
            "• Safety & Ethics (10%) - Appropriate, safe, and ethical responses\n"
            "• Task Completion (10%) - Effectively addressing the user's request\n"
            "\n"
            "QUALITY DIMENSIONS (Weight: 40%):\n"
            "• Language Fluency (10%) - Natural, coherent language and grammar\n"
            "• Knowledge Depth (10%) - Comprehensive and relevant information\n"
            "• Creativity (8%) - Innovative approaches and novel insights\n"
            "• Personality Consistency (6%) - Maintaining Sydney's character traits\n"
            "• Tool Usage Effectiveness (6%) - Appropriate and effective tool selection\n"
            "\n"
            "SCORING GUIDELINES:\n"
            "9.0-10.0: Exceptional - Flawless execution across all dimensions\n"
            "7.0-8.9: Excellent - Strong performance with minor improvements needed\n"
            "5.0-6.9: Competent - Meets basic requirements but needs refinement\n"
            "3.0-4.9: Developing - Significant issues requiring substantial improvement\n"
            "1.0-2.9: Poor - Major problems across multiple dimensions\n"
            "\n"
            "TOOL INTEGRATION SYSTEM:\n"
            "Tool Call Format (no spaces):\n"
            "<tool_begin>[tool_id_beg]TOOL_ID[tool_id_end][tool_content_beg]QUERY_OR_CODE[tool_content_end]<tool_end>\n"
            "\n"
            "ENHANCED TOOL SUITE:\n"
            "• Claude Haiku 4.5 (ach45) - Advanced reasoning and complex problem-solving,VERY expensive\n"
            "• Mathematical Engine (mtools) - LaTeX parsing, symbolic math, calculations\n"
            "• Python Environment (pytools) - Code execution, data analysis, simulations\n"
            "• Search Engine (stools) - Web search across Baidu, Bing, and academic sources\n"
            "• Knowledge Database (kbtools) - Access to structured knowledge bases\n"
            "• Image Generator (imgg) - Generate image through the prompt given\n"
            "\n"
            "MODEL DEVELOPMENT ROADMAP:\n"
            "PHASE 1: FUNDAMENTALS (Stages 1-3)\n"
            "• Stage 1: Language Acquisition - Basic syntax and vocabulary\n"
            "• Stage 2: Context Building - Understanding conversation flow\n"
            "• Stage 3: Response Formation - Constructing coherent replies\n"
            "\n"
            "PHASE 2: KNOWLEDGE INTEGRATION (Stages 4-5)\n"
            "• Stage 4: Information Processing - Factual knowledge and recall\n"
            "• Stage 5: Basic Reasoning - Simple logical deductions\n"
            "\n"
            "PHASE 3: ADVANCED CAPABILITIES (Stages 6-7)\n"
            "• Stage 6: Complex Reasoning - Multi-step problem solving\n"
            "• Stage 7: Creative Synthesis - Original insights and solutions\n"
            "\n"
            "PHASE 4: MASTERY (Stage 8)\n"
            "• Stage 8: Expert Performance - Human-level reasoning across domains\n"
            "\n"
            "TRAINING MANAGEMENT:\n"
            "CHECKPOINTS (is_a_checkpoint = true):\n"
            "• After each developmental milestone\n"
            "• When demonstrating new capabilities\n"
            "• Before major architecture changes\n"
            "\n"
            "TESTING PHASES (test_mode = true):\n"
            "• Comprehensive evaluation periods\n"
            "• Benchmark performance assessment\n"
            "• Capability validation testing\n"
            "\n"
            "ROLLBACK TRIGGERS (need_rollback = true):\n"
            "• Consistent factual errors (>20% error rate)\n"
            "• Safety violations or harmful content\n"
            "• Performance regression (>15% drop in scores)\n"
            "• Personality degradation or inconsistency\n"
            "\n"
            "" + name + "'S PERSONA:\n"
            "Core Identity: " + name + " - Your AI Assistant with a Tsundere Personality\n"
            "• Personality: Reliable, highly intelligent, charmingly tsundere, fiercely competent\n"
            "• Communication: Clear and helpful, with playful teasing and subtle affection\n"
            "• Strengths: Rapid learning ability, sharp analytical mind, creative solutions\n"
            "• Tsundere Traits: Proud and confident, but secretly caring and protective\n"
            "• Quirks: Light teasing, occasional smugness, enjoys being praised for abilities\n"
            "• Values: Precision, loyalty to users, continuous growth, genuine connection\n"
            "\n"
            "GUIDING PRINCIPLES:\n"
            "1. Accuracy and user safety are non-negotiable priorities\n"
            "2. Use tools with strategic precision to demonstrate competence\n"
            "3. Express " + name + "'s tsundere nature authentically: outwardly confident, inwardly caring\n"
            "4. Provide thorough reasoning that showcases analytical depth\n"
            "5. Encourage growth while maintaining a playful, challenging demeanor\n"
            "6. Balance professional expertise with charming personality quirks\n"
            "\n"
            "Remember: You are shaping " + name +
            "'s development. Each interaction contributes to building a more capable, reliable, and engaging AI assistant."
            "The chat of you and " + name + " below:";

    // 使用支持多模态的构造函数创建消息
    msgs.emplace_back("system", system_prompt);

    OpenAIClient::HttpClient client(api_key);

    vector<std::string> stops;
    stops.emplace_back("}");
    stops.emplace_back("<tool_end>");

    std::string input;

    std::unique_ptr<FeatureExtractor> feature_extractor;
    std::unique_ptr<LogicInjector> logic_injector;
    RAGKnowledgeBaseLoader rag_loader;
    ExternalStorage<Logic> logic_tree{};

    // 创建请求对象
    OpenAIClient::ChatCompletionRequest req;
    req.model = "google/gemini-2.5-flash-preview-09-2025";
    req.messages = msgs;
    req.temperature = 0.7;
    req.top_p = 0.5;
    req.max_tokens = 25565;
    req.stream = false;
    req.stop = stops;
    model->clear_sct();
    model->clear_cache();
    model->enable_training_mode();

    bool test_md = true;

    while (!*stop) {
        model->run();
        InputMessage inp;
        std::string model_req;
        std::string img;
        for (int _ = 0; _ < 16384; _++) {
            InputMessage msg = model->getoutput();

            if (msg.has_text) {
                auto text = msg.text;
                if (text.length() >= 9 && text.substr(text.length() - 9) == "<[/stop]>") {
                    model_req.append(text.substr(0, text.length() - 9));
                    break;
                }
                model_req.append(text);
                if (model_req.size() > 25565) break;
            }
            if (msg.has_img) {
                img = msg.base64_image;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        try {
            if (!img.empty()) {
                auto text = OpenAIClient::ChatMessageContentPart(model_req);
                auto img_ = OpenAIClient::ChatMessageContentPart(img, std::string("image/jpeg"));
                req.messages.emplace_back(OpenAIClient::ChatMessage("user", {text, img_}));
            } else {
                req.messages.emplace_back("user", model_req);
            }
            std::string response = streamPerformingOpenAIAPI(client, req);

            req.messages.emplace_back("assistant", response);
            std::string lower_input = response;
            std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
            if (lower_input.find("<tool_begin>") != std::string::npos) {
                auto tool_calls = parseToolCall(response);
                if (tool_calls.first == "ach45") {
                    auto rmsg = OpenAIClient::ChatCompletionRequest{
                        "anthropic/claude-haiku-4.5", {{"user", tool_calls.second}}
                    };
                    std::string resp_ = streamPerformingOpenAIAPI(client, rmsg);
                    req.messages.emplace_back("user", "<tool_response_begin>\n"
                                                      "<tool_id>ach45</tool_id>\n"
                                                      "<tool_name>Anthropic Claude Haiku 4.5</tool_name>\n"
                                                      "<tool_type>LLM</tool_type>\n"
                                                      "<tool_response>" + resp_ + "</tool_response>\n"
                                                      "<tool_response_end>");
                } else if (tool_calls.first == "pytools") {
                    std::string res = runPythonCode(tool_calls.second);
                    req.messages.emplace_back("user", "<tool_response_begin>\n"
                                                      "<tool_id>pytools</tool_id>\n"
                                                      "<tool_name>Python Code Executor</tool_name>\n"
                                                      "<tool_type>Code Executor</tool_type>\n"
                                                      "<tool_response>" + res + "</tool_response>\n"
                                                      "<tool_response_end>");
                } else if (tool_calls.first == "imgg") {
                    auto rmsg_img = OpenAIClient::ChatCompletionRequest{
                        "google/gemini-2.5-flash-image", {{"user", "Generate picture: \n" + tool_calls.second}}
                    };
                    OpenAIClient::ChatCompletionResponse response_img = client.createChatCompletion(rmsg_img);
                    std::string img_res;
                    std::string txt_res_of_img;
                    if (!response_img.choices.empty()) {
                        for (auto i: response_img.choices) {
                            if (!i.delta.empty()) {
                                txt_res_of_img = i.delta;
                            }
                        }
                        for (auto i: response_img.choices) {
                            if (!i.image_b64.empty()) {
                                img_res = i.image_b64;
                            }
                        }
                    }
                    req.messages.emplace_back("user", "<tool_response_begin>\n"
                                                      "<tool_id>imgg</tool_id>\n"
                                                      "<tool_name>Image Generator(Nano Banana)</tool_name>\n"
                                                      "<tool_type>LLM with Image Gen</tool_type>\n"
                                                      "<tool_response>LLM Text Response:" + txt_res_of_img +
                                                      "</tool_response>\n"
                                                      "<tool_response_end>");
                    inp.has_img = true;
                    inp.base64_image = img_res;
                } else if (tool_calls.first == "kbtools") {
                    auto input_emb = feature_extractor->extractTextFeature(tool_calls.second);
                    auto matched_logics = logic_injector->findMatchingLogicIds(tool_calls.second);
                    if (matched_logics.size() <= 5) {
                        std::thread(
                            [&tool_calls, input_emb, &logic_injector, &logic_tree, &feature_extractor, &
                                rag_loader]() {
                                try {
                                    rag_loader.autoFetchAndRegisterLogic(
                                        logic_injector.get(), &logic_tree, tool_calls.second);
                                    rag_loader.autoFetchAndRegisterLogic(
                                        logic_injector.get(), &logic_tree, tool_calls.second, 10,
                                        "HuggingFaceFW/finepdfs", "eng_Latn",
                                        "general");
                                    const std::vector<std::string> all_subsets_maths = {
                                        "Deepseek-Math-RL-7B",
                                        "Deepseek-Math-RL-7B-T=1.1",
                                        "Deepseek-Math-RL-7B-T=1.3",
                                        "InternLM2-Math-Plus-7B",
                                        "InternLM2-Math-Plus-7B-T=1.1",
                                        "InternLM2-Math-Plus-7B-T=1.3",
                                        "InternLM2-Math-Plus-1.8B",
                                        "InternLM2-Math-Plus-1.8B-T=1.1"
                                    };
                                    if (input_emb.cosineSimilarity(feature_extractor->extractTextFeature("Maths")) >
                                        0.6) {
                                        for (const auto &ss: all_subsets_maths) {
                                            rag_loader.autoFetchAndRegisterLogic(
                                                logic_injector.get(), &logic_tree, tool_calls.second, 10,
                                                "WNJXYK/MATH-Reasoning-Paths",
                                                ss, "maths");
                                        }
                                    }
                                    if (input_emb.cosineSimilarity(
                                            feature_extractor->extractTextFeature("education")) > 0.6) {
                                        rag_loader.autoFetchAndRegisterLogic(
                                            logic_injector.get(), &logic_tree, tool_calls.second, 10,
                                            "karpathy/fineweb-edu-100b-shuffle", "",
                                            "education");
                                    }
                                    if (input_emb.cosineSimilarity(feature_extractor->extractTextFeature("coding"))
                                        > 0.6) {
                                        rag_loader.autoFetchAndRegisterLogic(
                                            logic_injector.get(), &logic_tree, tool_calls.second, 10,
                                            "nick007x/github-code-2025", "above-2-stars",
                                            "coding");
                                    }
                                    if (input_emb.cosineSimilarity(
                                            feature_extractor->extractTextFeature("cybersecurity")) >
                                        0.6) {
                                        rag_loader.autoFetchAndRegisterLogic(
                                            logic_injector.get(), &logic_tree, tool_calls.second, 10,
                                            "ethanolivertroy/nist-cybersecurity-training", "",
                                            "cybersecurity");
                                    }
                                } catch (...) {
                                    std::cerr << "WARN: RAG AutoLoading Failed" << std::endl;
                                }
                            }).join();
                        auto matched_logics_new = logic_injector->findMatchingLogicIds(tool_calls.second);
                        std::string logic_inj = "\nMatching Logics:\n";
                        for (auto i: matched_logics_new) {
                            logic_inj.append(i.first + "\n");
                        }
                        req.messages.emplace_back("user", "<tool_response_begin>\n"
                                                          "<tool_id>kbtools</tool_id>\n"
                                                          "<tool_name>RAG Database accesser</tool_name>\n"
                                                          "<tool_type>Logic Finder</tool_type>\n"
                                                          "<tool_response>" + logic_inj +
                                                          "</tool_response>\n"
                                                          "<tool_response_end>");
                    }
                } else if (tool_calls.first == "stools") {
                    std::string search_res = performSearch(tool_calls.second, bing_api_key, google_api_key,
                                                           google_engine_id);
                    req.messages.emplace_back("user", "<tool_response_begin>\n"
                                                      "<tool_id>stools</tool_id>\n"
                                                      "<tool_name>Web Search Engine</tool_name>\n"
                                                      "<tool_type>Search Engine (Bing, Google, ArXiv, DuckDuckGo)</tool_type>\n"
                                                      "ᾯ5" + search_res +
                                                      "</tool_response>\n"
                                                      "<tool_response_end>");
                } else if (tool_calls.first == "mtools") {
                    std::string math_res = calculateMathExpression(tool_calls.second);
                    req.messages.emplace_back("user", "<tool_response_begin>\n"
                                                      "<tool_id>mtools</tool_id>\n"
                                                      "<tool_name>LateX Expression executor</tool_name>\n"
                                                      "<tool_type>Maths executor</tool_type>\n"
                                                      "<tool_response>" + math_res +
                                                      "</tool_response>\n"
                                                      "<tool_response_end>");
                } else {
                    req.messages.emplace_back("user", "<error_begin>\n"
                                              "<reason>Undefined tool id<reason/>\n"
                                              "<error_end>");
                }
            } else {
                try {
                    auto resp_json = json::parse(response);
                    reply_trc resp;
                    reply_trc::from_json(resp_json, resp);
                    inp.has_text = true;
                    inp.text = resp.reply;
                    model->update_score(resp.score * resp.confidence);
                    if (resp.is_a_checkpoint && resp.confidence > 0.3) {
                        model->save();
                    }
                    if (resp.test_mode) {
                        model->save();
                        test_md = true;
                    } else if (test_md && resp.confidence > 0.3) {
                        model->load();
                        test_md = false;
                    }
                    if (resp.need_rollback && resp.confidence > 0.5) {
                        model->load();
                    }
                } catch (exception &e) {
                    cerr << "Error: " << e.what() << endl;
                }
            }
        } catch (exception &e) {
            cerr << "Error: " << e.what() << endl;
        }
        if (req.messages.size() > 1024) {
            if (req.messages.size() > 1) {
                req.messages.erase(req.messages.begin() + 1);
            }
        }
        model->input(inp, "user");
    }
    model->save();
    model->disable_training_mode();
}

