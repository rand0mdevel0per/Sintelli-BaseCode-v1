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

using namespace std;

bool run = true;

void detect_stop() {
    std::string inp;
    while (run) {
        std::cin >> inp;
        if (inp == "stop") run = false;
    }
}

std::pair<std::string, std::string> parseToolCall(const std::string &input) {
    // 转换为小写进行不区分大小写的搜索
    std::string lower_input = input;
    std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);

    // 定义标记（也转换为小写）
    std::string tool_begin = "<tool_begin>";
    std::string tool_id_beg = "[tool_id_beg]";
    std::string tool_id_end = "[tool_id_end]";
    std::string tool_content_beg = "[tool_content_beg]";
    std::string tool_content_end = "[tool_content_end]";
    std::string tool_end = "<tool_end>";

    // 查找工具调用的开始位置
    size_t begin_pos = lower_input.find(tool_begin);
    if (begin_pos == std::string::npos) {
        return std::make_pair("", ""); // 未找到工具调用开始标记
    }

    // 查找工具调用的结束位置
    size_t end_pos = lower_input.find(tool_end, begin_pos);
    if (end_pos == std::string::npos) {
        return std::make_pair("", ""); // 未找到工具调用结束标记
    }

    // 提取完整的工具调用部分
    std::string tool_call_section = input.substr(begin_pos, end_pos - begin_pos + tool_end.length());
    std::string lower_section = lower_input.substr(begin_pos, end_pos - begin_pos + tool_end.length());

    // 查找tool_id_beg和tool_id_end的位置
    size_t id_beg_pos = lower_section.find(tool_id_beg);
    if (id_beg_pos == std::string::npos) {
        return std::make_pair("", ""); // 未找到tool_id开始标记
    }

    size_t id_start = id_beg_pos + tool_id_beg.length();
    size_t id_end_pos = lower_section.find(tool_id_end, id_start);
    if (id_end_pos == std::string::npos) {
        return std::make_pair("", ""); // 未找到tool_id结束标记
    }

    // 提取tool_id
    std::string tool_id = tool_call_section.substr(id_start, id_end_pos - id_start);

    // 查找tool_content_beg和tool_content_end的位置
    size_t content_beg_pos = lower_section.find(tool_content_beg);
    if (content_beg_pos == std::string::npos) {
        return std::make_pair("", ""); // 未找到tool_content开始标记
    }

    size_t content_start = content_beg_pos + tool_content_beg.length();
    size_t content_end_pos = lower_section.find(tool_content_end, content_start);
    if (content_end_pos == std::string::npos) {
        return std::make_pair("", ""); // 未找到tool_content结束标记
    }

    // 提取tool_content
    std::string tool_content = tool_call_section.substr(content_start, content_end_pos - content_start);

    // 去除首尾空格
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

void run_training(NeuronModel *model, const std::string &api_key, const std::string &name = "Sydney") {
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
            "• Claude Sonnet 4.5 (acs45) - Advanced reasoning and complex problem-solving,VERY expensive\n"
            "• Mathematical Engine (mtools) - LaTeX parsing, symbolic math, calculations\n"
            "• Python Environment (pytools) - Code execution, data analysis, simulations\n"
            "• Search Engine (stools) - Web search across Baidu, Bing, and academic sources\n"
            "• Knowledge Database (kbtools) - Access to structured knowledge bases\n"
            "• Code Analysis (catools) - Code review, debugging, optimization suggestions\n"
            "• Creative Assistant (crtools) - Content generation, writing assistance\n"
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

    run = true;

    OpenAIClient::HttpClient client(api_key);

    vector<std::string> stops;
    stops.emplace_back("}");
    stops.emplace_back("<tool_end>");

    std::string input;

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

    while (run) {
        model->run();
        InputMessage inp;
        inp.has_text = true;
        inp.text = input;
        model->input(inp, "user");
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
                switch (tool_calls.first) {
                    case "acs45":
                        auto rmsg = OpenAIClient::ChatCompletionRequest{
                            "anthropic/claude-sonnet-4.5", {{"user", tool_calls.second}}
                        };
                        std::string resp_ = streamPerformingOpenAIAPI(client, rmsg);
                        req.messages.emplace_back("user", "<tool_response_begin>\n"
                                                          "<tool_id>acs45<tool_id/>\n"
                                                          "<tool_name>Anthropic Claude Sonnet 4.5<tool_name/>\n"
                                                          "<tool_type>LLM<tool_type/>\n"
                                                          "<tool_response>" + resp_ + "<tool_response/>\n"
                                                          "<tool_response_end>");
                        break;
                    case "pytools":
                        system(("python " + tool_calls.second).c_str());
                }
            }
            if (req.messages.size() > 1024) {
                if (req.messages.size() > 1) {
                    req.messages.erase(req.messages.begin() + 1);
                }
            }
        } catch (const std::exception &e) {
            std::cerr << "Error calling API: " << e.what() << std::endl;
        }
    }
}

// 示例函数：展示如何使用图像输入功能
void example_with_image_input(OpenAIClient::HttpClient &client, const std::string &base64_image_data) {
    // 创建请求
    OpenAIClient::ChatCompletionRequest req;
    req.model = "gpt-4-vision-preview";
    req.max_tokens = 1000;

    // 创建图像内容部分
    OpenAIClient::ChatMessageContentPart image_part(base64_image_data, "image/jpeg");
    OpenAIClient::ChatMessageContentPart text_part("请描述这张图片的内容");

    // 组合内容部分
    std::vector<OpenAIClient::ChatMessageContentPart> parts = {text_part, image_part};
    OpenAIClient::ChatMessage message("user", parts);

    // 添加到请求中
    req.messages.push_back(message);

    // 发送请求
    try {
        OpenAIClient::ChatCompletionResponse response = client.createChatCompletion(req);
        if (!response.choices.empty()) {
            std::cout << "Image Analysis: " << response.choices[0].message.content << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << "Error calling API with image: " << e.what() << std::endl;
    }
}
