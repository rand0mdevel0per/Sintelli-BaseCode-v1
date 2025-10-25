//
// Optimized BPE Tokenizer Implementation
// Created with improved performance and error handling
//

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <memory>
#include "nlohmann/json.hpp"

using json = nlohmann::json;

class BPETokenizer {
private:
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<std::string, std::string> merge_rules; // Fast lookup for merge rules
    std::vector<std::pair<std::string, std::string>> merges;
    std::unordered_set<char> punctuation_chars;

    int pad_token_id = 0;
    int unk_token_id = 1;
    int cls_token_id = 101;
    int sep_token_id = 102;
    int max_length = 512;

    // Pre-compiled punctuation character set
    void initializePunctuation() {
        punctuation_chars = {
            ' ', '\t', '\n', '\r', ',', '.', '!', '?', ';', ':',
            '(', ')', '[', ']', '{', '}', '"', '\'', '-', '_',
            '/', '\\', '@', '#', '$', '%', '^', '&', '*', '+',
            '=', '|', '<', '>', '~', '`'
        };
    }

    // Optimized UTF-8 character extraction
    std::vector<std::string> getUtf8Chars(const std::string& text) {
        std::vector<std::string> chars;
        chars.reserve(text.size()); // 预分配内存

        for (size_t i = 0; i < text.length(); ) {
            unsigned char c = static_cast<unsigned char>(text[i]);
            int char_len = 1;

            // UTF-8字节序列长度判定
            if ((c & 0x80) == 0) {
                char_len = 1; // ASCII字符
            } else if ((c & 0xE0) == 0xC0) {
                char_len = 2; // 2字节UTF-8
            } else if ((c & 0xF0) == 0xE0) {
                char_len = 3; // 3字节UTF-8
            } else if ((c & 0xF8) == 0xF0) {
                char_len = 4; // 4字节UTF-8
            } else {
                // 无效UTF-8序列，跳过并记录警告
                std::cerr << "Warning: Invalid UTF-8 sequence at position " << i << std::endl;
                i++;
                continue;
            }

            // 检查是否有足够的字节
            if (i + char_len > text.length()) {
                std::cerr << "Warning: Incomplete UTF-8 sequence at end of text" << std::endl;
                break;
            }

            chars.push_back(text.substr(i, char_len));
            i += char_len;
        }

        return chars;
    }

    // 优化的BPE合并算法
    std::vector<std::string> applyBPE(const std::vector<std::string>& chars) {
        if (chars.empty()) return {};
        if (chars.size() == 1) return chars; // 单字符直接返回

        std::vector<std::string> tokens(chars.begin(), chars.end());
        bool changed = true;

        // 使用迭代合并直到无法继续合并
        while (changed && tokens.size() > 1) {
            changed = false;

            // 寻找最佳合并位置
            size_t best_pos = tokens.size();
            std::string best_merge;

            for (size_t i = 0; i < tokens.size() - 1; ++i) {
                std::string potential_merge = tokens[i] + tokens[i + 1];

                // 检查是否在merge_rules中存在（快速查找）
                if (merge_rules.find(potential_merge) != merge_rules.end() ||
                    vocab.find(potential_merge) != vocab.end()) {
                    best_pos = i;
                    best_merge = potential_merge;
                    break; // 找到第一个可合并的位置
                }
            }

            if (best_pos < tokens.size()) {
                // 执行合并
                tokens[best_pos] = best_merge;
                tokens.erase(tokens.begin() + best_pos + 1);
                changed = true;
            }
        }

        return tokens;
    }

    // 优化的预分词
    std::vector<std::string> preTokenize(const std::string& text) {
        std::vector<std::string> words;
        std::string current;
        current.reserve(64); // 预分配内存

        for (char c : text) {
            if (punctuation_chars.find(c) != punctuation_chars.end()) {
                if (!current.empty()) {
                    words.push_back(std::move(current));
                    current.clear();
                }
            } else {
                current += c;
            }
        }

        if (!current.empty()) {
            words.push_back(std::move(current));
        }

        return words;
    }

public:
    BPETokenizer(const char* vocab_path,
                 const char* merges_path,
                 const char* special_tokens_path,
                 int max_len = 512) : max_length(max_len) {

        initializePunctuation();

        // 加载词汇表
        std::ifstream vocab_file(vocab_path);
        if (!vocab_file.is_open()) {
            throw std::runtime_error("Failed to open vocabulary file: " + std::string(vocab_path));
        }

        json vocab_json;
        try {
            vocab_file >> vocab_json;
        } catch (const json::parse_error& e) {
            throw std::runtime_error("Failed to parse vocabulary JSON: " + std::string(e.what()));
        }

        for (auto& el : vocab_json.items()) {
            vocab[el.key()] = el.value();
        }

        // 加载merges
        std::ifstream merges_file(merges_path);
        if (!merges_file.is_open()) {
            throw std::runtime_error("Failed to open merges file: " + std::string(merges_path));
        }

        std::string line;
        while (std::getline(merges_file, line)) {
            if (line.empty() || line[0] == '#') continue;

            size_t space_pos = line.find(' ');
            if (space_pos != std::string::npos) {
                std::string first = line.substr(0, space_pos);
                std::string second = line.substr(space_pos + 1);
                merges.push_back({first, second});
                merge_rules[first + second] = first + second;
            }
        }

        // 加载特殊tokens
        std::ifstream special_file(special_tokens_path);
        if (!special_file.is_open()) {
            throw std::runtime_error("Failed to open special tokens file: " + std::string(special_tokens_path));
        }

        json special_json;
        try {
            special_file >> special_json;
        } catch (const json::parse_error& e) {
            throw std::runtime_error("Failed to parse special tokens JSON: " + std::string(e.what()));
        }

        pad_token_id = special_json.value("pad_token_id", 0);
        unk_token_id = special_json.value("unk_token_id", 1);
        cls_token_id = special_json.value("cls_token_id", 101);
        sep_token_id = special_json.value("sep_token_id", 102);
    }

    // 编码文本
    bool encode(const char* text, int start, int end,
                std::vector<int>& input_ids,
                std::vector<int>& attention_mask) {

        try{
            if (!text || start < 0 || end < start) {
                throw std::invalid_argument("Invalid text range");
            }

            input_ids.clear();
            attention_mask.clear();

            // 预分配内存以提高性能
            input_ids.reserve(max_length);
            attention_mask.reserve(max_length);

            // 添加 [CLS]
            input_ids.push_back(cls_token_id);
            attention_mask.push_back(1);

            // 提取文本片段
            std::string segment(text + start, end - start);

            // 预分词
            std::vector<std::string> words = preTokenize(segment);

            for (const auto& word : words) {
                if (input_ids.size() >= max_length - 1) break;

                // UTF-8字符级别分割
                std::vector<std::string> chars = getUtf8Chars(word);

                // 应用BPE
                std::vector<std::string> tokens = applyBPE(chars);

                // 转换为ID
                for (const auto& token : tokens) {
                    if (input_ids.size() >= max_length - 1) break;

                    auto it = vocab.find(token);
                    if (it != vocab.end()) {
                        input_ids.push_back(it->second);
                    } else {
                        input_ids.push_back(unk_token_id);
                    }
                    attention_mask.push_back(1);
                }
            }

            // 添加 [SEP]
            if (input_ids.size() < max_length) {
                input_ids.push_back(sep_token_id);
                attention_mask.push_back(1);
            }

            // Padding
            size_t current_size = input_ids.size();
            for (size_t i = current_size; i < max_length; ++i) {
                input_ids.push_back(pad_token_id);
                attention_mask.push_back(0);
            }
            return true;
        }catch (...){
            return false;
        }
    }

    // 批量编码接口
    void encodeBatch(const std::vector<std::string>& texts,
                     std::vector<std::vector<int>>& batch_input_ids,
                     std::vector<std::vector<int>>& batch_attention_mask) {
        batch_input_ids.clear();
        batch_attention_mask.clear();

        for (const auto& text : texts) {
            std::vector<int> input_ids;
            std::vector<int> attention_mask;

            encode(text.c_str(), 0, text.length(), input_ids, attention_mask);

            batch_input_ids.push_back(std::move(input_ids));
            batch_attention_mask.push_back(std::move(attention_mask));
        }
    }

    // 解码token IDs到文本
    std::string decode(const std::vector<int>& token_ids) {
        std::string result;

        // 创建反向词汇表
        std::unordered_map<int, std::string> reverse_vocab;
        for (const auto& pair : vocab) {
            reverse_vocab[pair.second] = pair.first;
        }

        for (int token_id : token_ids) {
            if (token_id == pad_token_id || token_id == cls_token_id ||
                token_id == sep_token_id) {
                continue; // 跳过特殊token
            }

            auto it = reverse_vocab.find(token_id);
            if (it != reverse_vocab.end()) {
                result += it->second;
            } else {
                result += "[UNK]";
            }
        }

        return result;
    }

    int getMaxLength() const { return max_length; }
    size_t getVocabSize() const { return vocab.size(); }

    // 获取特殊token ID
    int getPadTokenId() const { return pad_token_id; }
    int getUnkTokenId() const { return unk_token_id; }
    int getClsTokenId() const { return cls_token_id; }
    int getSepTokenId() const { return sep_token_id; }

    // 检查token是否在词汇表中
    bool hasToken(const std::string& token) const {
        return vocab.find(token) != vocab.end();
    }
};