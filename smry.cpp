// ============================================================================
// Unified Image-Text Matrix Encoding System - Full Version (Pure C++ BPE Tokenizer)
// unified_system.cpp
// ============================================================================

#ifndef SRC_UNIFIED_SYSTEM_CPP
#define SRC_UNIFIED_SYSTEM_CPP

#include <onnxruntime_cxx_api.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <unordered_map>
#include <cwchar>
#include <locale>
#include <codecvt>

#include "structs.h"
#include "json/nlohmann/json.hpp"
#include "converter.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_WINDOWS_UTF8
#include "stb_image.h"
#include "stb_image_write.h"

using json = nlohmann::json;

// ============ 常量定义 ============
constexpr int MAT_SIZE = 256;
constexpr int MAT_ELEMENTS = MAT_SIZE * MAT_SIZE;
constexpr int EMBED_DIM = 1024;
constexpr int MAX_SEQ_LENGTH = 512;
constexpr double PI = 3.14159265358979323846;

// ============ 基础数据结构 ============
struct Matrix256 {
    double data[MAT_ELEMENTS];
    void clear() { memset(data, 0, sizeof(data)); }
    Matrix256() { clear(); }
};

struct ImageData {
    unsigned char* pixels;
    int width;
    int height;
    int channels;

    // 构造函数
    ImageData() : pixels(nullptr), width(0), height(0), channels(0) {}

    // 析构函数 - 自动释放内存
    ~ImageData() {
        if (pixels) {
            stbi_image_free(pixels); // 非常重要：使用 stb_image 提供的释放函数
            pixels = nullptr;
        }
    }

    // 禁止拷贝（避免双重释放）
    ImageData(const ImageData&) = delete;
    ImageData& operator=(const ImageData&) = delete;

    // 允许移动
    ImageData(ImageData&& other) noexcept
        : pixels(other.pixels), width(other.width), height(other.height), channels(other.channels) {
        other.pixels = nullptr;
        other.width = 0;
        other.height = 0;
        other.channels = 0;
    }
};

// Base64解码函数
std::vector<unsigned char> base64_decode(const std::string& encoded_string) {
    const std::string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

    // 检查是否有数据URI前缀
    std::string clean_encoded = encoded_string;
    size_t data_prefix = clean_encoded.find("base64,");
    if (data_prefix != std::string::npos) {
        clean_encoded = clean_encoded.substr(data_prefix + 7); // 跳过 "base64,"
    }

    // 移除所有非base64字符（如换行符等）
    std::string clean_data;
    for (char c : clean_encoded) {
        if (isalnum(c) || c == '+' || c == '/' || c == '=') {
            clean_data.push_back(c);
        }
    }

    std::vector<unsigned char> decoded_bytes;
    int in_len = clean_data.size();
    int i = 0;
    int j = 0;
    unsigned char char_array_4[4], char_array_3[3];

    while (in_len-- && (clean_data[i] != '=')) {
        char_array_4[j++] = clean_data[i++];
        if (j == 4) {
            for (j = 0; j < 4; j++) {
                char_array_4[j] = base64_chars.find(char_array_4[j]);
            }

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (j = 0; (j < 3); j++) {
                decoded_bytes.push_back(char_array_3[j]);
            }
            j = 0;
        }
    }

    if (j) {
        for (int k = j; k < 4; k++) {
            char_array_4[k] = 0;
        }

        for (int k = 0; k < 4; k++) {
            char_array_4[k] = base64_chars.find(char_array_4[k]);
        }

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (int k = 0; (k < j - 1); k++) {
            decoded_bytes.push_back(char_array_3[k]);
        }
    }

    return decoded_bytes;
}

// Base64编码函数
std::string base64_encode(const std::vector<unsigned char>& data) {
    const std::string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

    std::string encoded;
    size_t len = data.size();
    size_t i = 0;
    
    while (len >= 3) {
        unsigned char char_array_3[3] = {data[i], data[i+1], data[i+2]};
        unsigned char char_array_4[4];
        
        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for (int j = 0; j < 4; j++) {
            encoded += base64_chars[char_array_4[j]];
        }
        
        i += 3;
        len -= 3;
    }

    if (len > 0) {
        unsigned char char_array_3[3] = {0};
        for (size_t j = 0; j < len; j++) {
            char_array_3[j] = data[i + j];
        }

        unsigned char char_array_4[4];
        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for (size_t j = 0; j < len + 1; j++) {
            encoded += base64_chars[char_array_4[j]];
        }

        while (len < 3) {
            encoded += '=';
            len++;
        }
    }

    return encoded;
}

// 图像转Base64字符串
std::string image_to_base64(const ImageData& img, const std::string& format = "png") {
    if (!img.pixels || img.width <= 0 || img.height <= 0) {
        return "";
    }

    // 使用stb_image_write保存到内存
    int data_size;
    unsigned char* image_data = stbi_write_png_to_mem(
        img.pixels, 
        img.width * img.channels,  // stride
        img.width, 
        img.height, 
        img.channels, 
        &data_size
    );

    if (!image_data) {
        return "";
    }

    // 编码为Base64
    std::vector<unsigned char> data(image_data, image_data + data_size);
    std::string base64_result = base64_encode(data);
    
    // 添加数据URI前缀
    std::string result = "data:image/png;base64," + base64_result;
    
    // 释放内存
    STBIW_FREE(image_data);
    
    return result;
}

ImageData base64_to_image(const std::string& base64_string) {
    ImageData result;

    try {
        // 1. Base64 解码
        std::vector<unsigned char> decoded_data = base64_decode(base64_string);

        if (decoded_data.empty()) {
            throw std::runtime_error("Base64解码失败或数据为空");
        }

        // 2. 使用 stb_image 从内存加载图像
        int width, height, channels;
        // desired_channels 设置为 0 表示保持图像原有的通道数
        // 如果你想强制转换为RGBA（4通道），可以传4
        result.pixels = stbi_load_from_memory(
            decoded_data.data(),        // 指向内存中图像数据的指针
            static_cast<int>(decoded_data.size()), // 数据长度
            &width,                     // 用于接收图像宽度
            &height,                    // 用于接收图像高度
            &channels,                  // 用于接收图像原始通道数
            0                           // 期望的通道数 (0表示保持原样)
        );

        // 检查加载是否成功
        if (result.pixels == nullptr) {
            throw std::runtime_error(std::string("STB_image 加载失败: ") + stbi_failure_reason());
        }

        result.width = width;
        result.height = height;
        result.channels = channels;

        std::cout << "成功加载图像: " << width << "x" << height << ", 通道数: " << channels << std::endl;

    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("图像转换失败: ") + e.what());
    }

    return result;
}

// ============ DCT变换 ============
void dct2d(const double* input, double* output, int size) {
    double* temp = new double[size * size];

    for (int i = 0; i < size; i++) {
        for (int u = 0; u < size; u++) {
            double sum = 0.0;
            double cu = (u == 0) ? sqrt(1.0/size) : sqrt(2.0/size);
            for (int x = 0; x < size; x++) {
                sum += input[i*size + x] * cos(PI * u * (2*x + 1) / (2.0*size));
            }
            temp[i*size + u] = cu * sum;
        }
    }

    for (int j = 0; j < size; j++) {
        for (int v = 0; v < size; v++) {
            double sum = 0.0;
            double cv = (v == 0) ? sqrt(1.0/size) : sqrt(2.0/size);
            for (int y = 0; y < size; y++) {
                sum += temp[y*size + j] * cos(PI * v * (2*y + 1) / (2.0*size));
            }
            output[v*size + j] = cv * sum;
        }
    }

    delete[] temp;
}

void idct2d(const double* input, double* output, int size) {
    double* temp = new double[size * size];

    for (int i = 0; i < size; i++) {
        for (int x = 0; x < size; x++) {
            double sum = 0.0;
            for (int u = 0; u < size; u++) {
                double cu = (u == 0) ? sqrt(1.0/size) : sqrt(2.0/size);
                sum += cu * input[i*size + u] * cos(PI * u * (2*x + 1) / (2.0*size));
            }
            temp[i*size + x] = sum;
        }
    }

    for (int j = 0; j < size; j++) {
        for (int y = 0; y < size; y++) {
            double sum = 0.0;
            for (int v = 0; v < size; v++) {
                double cv = (v == 0) ? sqrt(1.0/size) : sqrt(2.0/size);
                sum += cv * temp[v*size + j] * cos(PI * v * (2*y + 1) / (2.0*size));
            }
            output[y*size + j] = sum;
        }
    }

    delete[] temp;
}

// ============ 内容类型检测 ============
enum ContentType {
    TYPE_NATURAL_TEXT,
    TYPE_CODE,
    TYPE_MIXED
};

ContentType detectContentType(const wchar_t* text, int length) {
    std::wstring content(text, std::min(length, 5000));

    int code_indicators = 0;
    const wchar_t* code_patterns[] = {
        L"def ", L"class ", L"function", L"const ", L"let ", L"var ",
        L"#include", L"import ", L"public ", L"private ", L"void ",
        L"int ", L"float ", L"return ", L"if(", L"for(", L"while(",
        L"{", L"}", L"//", L"/*", L"=>", L"->", nullptr
    };

    for (int i = 0; code_patterns[i] != nullptr; i++) {
        if (content.find(code_patterns[i]) != std::wstring::npos) {
            code_indicators++;
        }
    }

    int special_chars = 0;
    int alphanumeric = 0;
    for (wchar_t c : content) {
        if (c == L'{' || c == L'}' || c == L';' || c == L'(' || c == L')') {
            special_chars++;
        }
        if ((c >= L'a' && c <= L'z') || (c >= L'A' && c <= L'Z') ||
            (c >= L'0' && c <= L'9')) {
            alphanumeric++;
        }
    }

    double special_ratio = (double)special_chars / (alphanumeric + 1);

    if (code_indicators > 5 || special_ratio > 0.15) {
        return TYPE_CODE;
    } else if (code_indicators > 2) {
        return TYPE_MIXED;
    }

    return TYPE_NATURAL_TEXT;
}

#include "bpe_tokenizer.cpp"

// ============ E5-Large模型 ============
class E5LargeModel {
private:
    Ort::Env env;
    Ort::Session* session;
    Ort::SessionOptions session_options;
    BPETokenizer* tokenizer;
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;

    void meanPooling(const float* token_embeddings, const int64_t* attention_mask,
                     int seq_length, double* output) {
        memset(output, 0, EMBED_DIM * sizeof(double));
        int sum_mask = 0;
        for (int i = 0; i < seq_length; i++) {
            if (attention_mask[i] == 1) {
                sum_mask++;
                for (int j = 0; j < EMBED_DIM; j++) {
                    output[j] += (double)token_embeddings[i * EMBED_DIM + j];
                }
            }
        }
        if (sum_mask > 0) {
            for (int j = 0; j < EMBED_DIM; j++) output[j] /= sum_mask;
        }

        double norm = 0.0;
        for (int j = 0; j < EMBED_DIM; j++) norm += output[j] * output[j];
        norm = sqrt(norm);
        if (norm > 1e-6) {
            for (int j = 0; j < EMBED_DIM; j++) output[j] /= norm;
        }
    }

public:
    E5LargeModel(const char* model_path, const char* vocab_path,
                 const char* merges_path, const char* special_tokens_path)
        : env(ORT_LOGGING_LEVEL_WARNING, "E5Large"), session(nullptr), tokenizer(nullptr) {

        tokenizer = new BPETokenizer(vocab_path, merges_path, special_tokens_path, MAX_SEQ_LENGTH);
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        #ifdef USE_INTEL_NPU
        session_options.AppendExecutionProvider("OpenVINO", {{"device_type", "NPU"}});
        #endif

        #ifdef _WIN32
        std::wstring wide_path;
        for (char c : std::string(model_path)) wide_path += (wchar_t)c;
        session = new Ort::Session(env, wide_path.c_str(), session_options);
        #else
        session = new Ort::Session(env, model_path, session_options);
        #endif

        input_node_names.push_back("input_ids");
        input_node_names.push_back("attention_mask");
        output_node_names.push_back("last_hidden_state");
    }

    ~E5LargeModel() {
        if (session) delete session;
        if (tokenizer) delete tokenizer;
    }

    bool getEmbedding(const wchar_t* text, int start, int end, double* embedding) {
        if (!session || !tokenizer) return false;

        // 将宽字符转换为UTF-8字符串
        std::wstring wstr(text + start, end - start);
        std::string utf8_str;
        utf8_str.resize(wstr.length() * 4); // 预留足够空间
        
        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        utf8_str = converter.to_bytes(wstr);

        std::vector<int> input_ids;
        std::vector<int> attention_mask;
        if (!tokenizer->encode(utf8_str.c_str(), 0, utf8_str.length(), input_ids, attention_mask)) return false;

        std::vector<int64_t> input_shape = {1, MAX_SEQ_LENGTH};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // 修正：正确传递向量数据
        Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, 
            reinterpret_cast<int64_t*>(input_ids.data()), 
            input_ids.size(), 
            input_shape.data(), 
            input_shape.size());
        Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, 
            reinterpret_cast<int64_t*>(attention_mask.data()), 
            attention_mask.size(), 
            input_shape.data(), 
            input_shape.size());

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(input_ids_tensor));
        input_tensors.push_back(std::move(attention_mask_tensor));

        auto output_tensors = session->Run(Ort::RunOptions{nullptr},
            input_node_names.data(), input_tensors.data(), input_tensors.size(),
            output_node_names.data(), output_node_names.size());

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        meanPooling(output_data, reinterpret_cast<const int64_t*>(attention_mask.data()), MAX_SEQ_LENGTH, embedding);
        return true;
    }

    size_t getVocabSize() const { return tokenizer ? tokenizer->getVocabSize() : 0; }
};

// ============ 语义向量压缩 ============
Matrix256* compressSemanticVector(const double* vec_1024d) {
    if (!vec_1024d) return nullptr;
    Matrix256* mat = new Matrix256();

    int repeat_factor = MAT_ELEMENTS / EMBED_DIM;
    int idx = 0;
    for (int i = 0; i < EMBED_DIM; i++) {
        for (int j = 0; j < repeat_factor && idx < MAT_ELEMENTS; j++) {
            mat->data[idx++] = vec_1024d[i];
        }
    }

    double* smoothed = new double[MAT_ELEMENTS];
    const double gaussian[9] = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};

    for (int i = 0; i < MAT_SIZE; i++) {
        for (int j = 0; j < MAT_SIZE; j++) {
            double sum = 0.0;
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    int ni = i + ki, nj = j + kj;
                    if (ni >= 0 && ni < MAT_SIZE && nj >= 0 && nj < MAT_SIZE) {
                        sum += mat->data[ni * MAT_SIZE + nj] * gaussian[(ki+1)*3 + (kj+1)];
                    }
                }
            }
            smoothed[i * MAT_SIZE + j] = sum;
        }
    }

    memcpy(mat->data, smoothed, MAT_ELEMENTS * sizeof(double));
    delete[] smoothed;

    double* temp = new double[MAT_ELEMENTS];
    memcpy(temp, mat->data, MAT_ELEMENTS * sizeof(double));
    dct2d(temp, mat->data, MAT_SIZE);
    delete[] temp;
    return mat;
}

// ============ 统一输入处理器 ============
class UnifiedInputProcessor {
private:
    struct Node {
        double* embedding;
        double attention;
        double importance;
        int start_pos, end_pos;
        Node() : embedding(nullptr), attention(1.0), importance(1.0), start_pos(0), end_pos(0) {}
        ~Node() { if (embedding) delete[] embedding; }
    };

    E5LargeModel* model;
    Node* nodes;
    int node_count, capacity, current_extract_idx;
    double* global_semantic_vector;
    ContentType content_type;

    bool processCodeContent(const wchar_t* text, int length) {
        int chunk_size = 1500;
        for (int pos = 0; pos < length && node_count < capacity; pos += chunk_size) {
            int end_pos = std::min(pos + chunk_size, length);
            nodes[node_count].embedding = new double[EMBED_DIM];
            nodes[node_count].start_pos = pos;
            nodes[node_count].end_pos = end_pos;

            if (!model->getEmbedding(text, pos, end_pos, nodes[node_count].embedding)) {
                delete[] nodes[node_count].embedding;
                nodes[node_count].embedding = nullptr;
                continue;
            }
            nodes[node_count].importance = 1.0;
            node_count++;
        }
        return node_count > 0;
    }

    bool processNaturalText(const wchar_t* text, int length) {
        if (length <= 3000) {
            nodes[0].embedding = new double[EMBED_DIM];
            nodes[0].start_pos = 0;
            nodes[0].end_pos = length;
            if (model->getEmbedding(text, 0, length, nodes[0].embedding)) {
                nodes[0].importance = 1.5;
                node_count = 1;
                return true;
            }
            return false;
        }

        int chunk_size = 1500;
        for (int pos = 0; pos < length && node_count < capacity; pos += chunk_size) {
            int end_pos = std::min(pos + chunk_size, length);
            if (end_pos < length) {
                for (int i = 0; i < 200 && end_pos + i < length; i++) {
                    wchar_t c = text[end_pos + i];
                    // 支持中文标点的分句
                    if (c == L'.' || c == L'!' || c == L'?' || c == L'\n' ||
                        c == L'。' || c == L'！' || c == L'？' || c == L'；') {
                        end_pos += i + 1;
                        break;
                    }
                }
            }

            nodes[node_count].embedding = new double[EMBED_DIM];
            nodes[node_count].start_pos = pos;
            nodes[node_count].end_pos = end_pos;

            if (!model->getEmbedding(text, pos, end_pos, nodes[node_count].embedding)) {
                delete[] nodes[node_count].embedding;
                nodes[node_count].embedding = nullptr;
                continue;
            }

            double norm = 0.0;
            for (int i = 0; i < EMBED_DIM; i++) {
                norm += nodes[node_count].embedding[i] * nodes[node_count].embedding[i];
            }
            nodes[node_count].importance = sqrt(norm);
            node_count++;
            pos = end_pos - chunk_size;
        }
        return node_count > 0;
    }

public:
    UnifiedInputProcessor(E5LargeModel* e5_model)
        : model(e5_model), nodes(nullptr), node_count(0), capacity(1000),
          current_extract_idx(0), global_semantic_vector(nullptr), content_type(TYPE_NATURAL_TEXT) {
        nodes = new Node[capacity];
        global_semantic_vector = new double[EMBED_DIM];
    }

    ~UnifiedInputProcessor() {
        if (nodes) delete[] nodes;
        if (global_semantic_vector) delete[] global_semantic_vector;
    }

    bool processText(const wchar_t* text, int length) {
        if (!text || length == 0 || !model) return false;
        node_count = 0;
        current_extract_idx = 0;
        content_type = detectContentType(text, length);

        if (!model->getEmbedding(text, 0, std::min(length, 5000), global_semantic_vector)) {
            return false;
        }

        return (content_type == TYPE_CODE) ? processCodeContent(text, length) : processNaturalText(text, length);
    }

    Matrix256* getGlobalAttentionMatrix() { return compressSemanticVector(global_semantic_vector); }

    Matrix256* getNextBlock() {
        if (current_extract_idx >= node_count) return nullptr;
        Matrix256* mat = new Matrix256();
        int nodes_per_block = MAT_ELEMENTS / EMBED_DIM;
        int block_end = std::min(current_extract_idx + nodes_per_block, node_count);

        int idx = 0;
        for (int i = current_extract_idx; i < block_end && idx < MAT_ELEMENTS; i++) {
            double weight = nodes[i].attention * nodes[i].importance;
            for (int j = 0; j < EMBED_DIM && idx < MAT_ELEMENTS; j++) {
                mat->data[idx++] = nodes[i].embedding[j] * weight;
            }
        }
        while (idx < MAT_ELEMENTS) mat->data[idx++] = 0.0;

        double* temp = new double[MAT_ELEMENTS];
        memcpy(temp, mat->data, MAT_ELEMENTS * sizeof(double));
        dct2d(temp, mat->data, MAT_SIZE);
        delete[] temp;

        current_extract_idx = block_end;
        return mat;
    }

    bool hasMoreBlocks() const { return current_extract_idx < node_count; }
    int getTotalBlocks() const { return (node_count + (MAT_ELEMENTS/EMBED_DIM) - 1) / (MAT_ELEMENTS/EMBED_DIM); }

    const char* getContentTypeName() const {
        switch (content_type) {
            case TYPE_CODE: return "code";
            case TYPE_NATURAL_TEXT: return "natural_text";
            case TYPE_MIXED: return "mixed_content";
            default: return "unknown";
        }
    }
};

// ============ 图像处理 ============
class ImageProcessor {
public:
    static Matrix256* encode(const ImageData* img) {
        if (!img || !img->pixels) return nullptr;
        Matrix256* mat = new Matrix256();
        double* temp = new double[MAT_ELEMENTS];
        double scale_x = (double)img->width / MAT_SIZE;
        double scale_y = (double)img->height / MAT_SIZE;

        for (int i = 0; i < MAT_SIZE; i++) {
            for (int j = 0; j < MAT_SIZE; j++) {
                double src_x = j * scale_x, src_y = i * scale_y;
                int x0 = (int)src_x, y0 = (int)src_y;
                int x1 = std::min(x0 + 1, img->width - 1);
                int y1 = std::min(y0 + 1, img->height - 1);
                double fx = src_x - x0, fy = src_y - y0;

                auto getGray = [&](int x, int y) -> double {
                    int idx = (y * img->width + x) * img->channels;
                    return 0.299 * img->pixels[idx] + 0.587 * img->pixels[idx+1] + 0.114 * img->pixels[idx+2];
                };

                double g00 = getGray(x0, y0), g01 = getGray(x1, y0);
                double g10 = getGray(x0, y1), g11 = getGray(x1, y1);
                double gray = (1-fx)*(1-fy)*g00 + fx*(1-fy)*g01 + (1-fx)*fy*g10 + fx*fy*g11;
                temp[i*MAT_SIZE + j] = gray / 255.0;
            }
        }

        dct2d(temp, mat->data, MAT_SIZE);
        delete[] temp;
        return mat;
    }

    static ImageData* decode(const Matrix256* mat) {
        if (!mat) return nullptr;
        double* temp = new double[MAT_ELEMENTS];
        idct2d(mat->data, temp, MAT_SIZE);

        ImageData* img = new ImageData();
        img->width = MAT_SIZE;
        img->height = MAT_SIZE;
        img->channels = 3;
        img->pixels = new unsigned char[MAT_ELEMENTS * 3];

        for (int i = 0; i < MAT_ELEMENTS; i++) {
            double val = std::max(0.0, std::min(255.0, temp[i] * 255.0));
            unsigned char gray = (unsigned char)val;
            img->pixels[i*3] = img->pixels[i*3+1] = img->pixels[i*3+2] = gray;
        }

        delete[] temp;
        return img;
    }

    static ImageData* mergeFrames(Matrix256** frames, int frame_count, double threshold = 0.01) {
        if (!frames || frame_count == 0) return nullptr;
        Matrix256 merged;
        int valid_count = 0;

        for (int i = 0; i < frame_count; i++) {
            if (!frames[i]) continue;
            double energy = 0.0;
            for (int j = 0; j < MAT_ELEMENTS; j++) {
                energy += frames[i]->data[j] * frames[i]->data[j];
            }
            if (sqrt(energy / MAT_ELEMENTS) > threshold) {
                for (int j = 0; j < MAT_ELEMENTS; j++) merged.data[j] += frames[i]->data[j];
                valid_count++;
            }
        }

        if (valid_count == 0) return nullptr;
        for (int j = 0; j < MAT_ELEMENTS; j++) merged.data[j] /= valid_count;
        return decode(&merged);
    }

    // 检测帧是否有效
    // 条件1: 和上一帧有差别 (差异阈值可配置)
    // 条件2: 携带有信息 (能量阈值可配置)
    static bool isValidFrame(const Matrix256* current_frame, const Matrix256* previous_frame = nullptr, 
                            double diff_threshold = 0.001, double energy_threshold = 0.01) {
        if (!current_frame) return false;
        
        // 条件2: 检测帧是否携带有信息 (能量检测)
        double energy = 0.0;
        for (int j = 0; j < MAT_ELEMENTS; j++) {
            energy += current_frame->data[j] * current_frame->data[j];
        }
        double avg_energy = sqrt(energy / MAT_ELEMENTS);
        if (avg_energy <= energy_threshold) {
            return false;  // 能量太低，不包含有效信息
        }
        
        // 条件1: 检测与前一帧的差异 (如果提供了前一帧)
        if (previous_frame) {
            double diff_sum = 0.0;
            for (int j = 0; j < MAT_ELEMENTS; j++) {
                double diff = current_frame->data[j] - previous_frame->data[j];
                diff_sum += diff * diff;
            }
            double avg_diff = sqrt(diff_sum / MAT_ELEMENTS);
            if (avg_diff <= diff_threshold) {
                return false;  // 与前一帧差异太小，可能是重复帧
            }
        }
        
        return true;
    }
};

// ============ 全局API ============
static E5LargeModel* g_e5_model = nullptr;

bool initUnifiedSystem(const char* model_path, const char* vocab_path,
                      const char* merges_path, const char* special_tokens_path) {
    if (g_e5_model) delete g_e5_model;
    try {
        g_e5_model = new E5LargeModel(model_path, vocab_path, merges_path, special_tokens_path);
        return true;
    }catch (...) {
        if (g_e5_model) {
            delete g_e5_model;
            g_e5_model = nullptr;
        }
        return false;
    }
}

// 创建文本处理器
UnifiedInputProcessor* createTextProcessor() {
    if (!g_e5_model) return nullptr;
    return new UnifiedInputProcessor(g_e5_model);
}

// 宽字符版本的文本处理函数
bool processTextW(UnifiedInputProcessor* processor, const wchar_t* text, int length) {
    if (!processor || !text || length <= 0) return false;
    return processor->processText(text, length);
}

std::wstring stringToWstring(const std::string& str) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.from_bytes(str);
}

bool processTextString(UnifiedInputProcessor* processor, const std::string& text) {
    return processTextW(processor, stringToWstring(text).c_str(), text.length());
}

// 256x256 double矩阵转文本函数
// 基于converter.h中的实现，提供多种输出格式
bool matrix_to_text(const Matrix256* matrix, std::string& result, int format = 0) {
    if (!matrix) return false;
    
    SimpleString output;
    bool success = false;
    
    switch (format) {
        case 0: // 默认：使用特征提取和量化编码（压缩格式）
            {
                DoubleBuffer features;
                if (extract_features(matrix->data, 256, 256, features)) {
                    success = features_to_utf8(features, 8, output);
                }
            }
            break;
            
        case 1: // 完整矩阵编码（无损，但体积大）
            success = matrix_to_utf8_full(matrix->data, 256, 256, output);
            break;
            
        case 2: // 简单文本格式（可读性好）
            {
                // 提取关键特征并格式化为可读文本
                DoubleBuffer features;
                if (extract_features(matrix->data, 256, 256, features)) {
                    // 将特征转换为16进制字符串
                    SimpleString hex_output;
                    char* hex_str = new (std::nothrow) char[features.size() * 3 + 1];
                    if (hex_str) {
                        size_t idx = 0;
                        for (size_t i = 0; i < features.size(); ++i) {
                            int hex_val = static_cast<int>((features[i] + 1.0) * 127.5); // 映射到0-255
                            if (hex_val < 0) hex_val = 0;
                            if (hex_val > 255) hex_val = 255;
                            idx += sprintf(hex_str + idx, "%02X ", hex_val);
                        }
                        hex_output = SimpleString(hex_str);
                        delete[] hex_str;
                        success = true;
                        output = std::move(hex_output);
                    }
                }
            }
            break;
            
        default:
            return false;
    }
    
    if (success && !output.empty()) {
        result = std::string(output.c_str());
        return true;
    }
    
    return false;
}

// 便捷函数：直接获取矩阵的文本表示
std::string matrix_to_string(const Matrix256* matrix, int format = 0) {
    std::string result;
    if (matrix_to_text(matrix, result, format)) {
        return result;
    }
    return "";
}

#endif