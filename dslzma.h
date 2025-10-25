//
// Created by ASUS on 10/3/2025.
//

#ifndef SRC_DSLZMA_H
#define SRC_DSLZMA_H

#ifndef SERIALIZER_H
#define SERIALIZER_H

#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <lzma.h>
#endif

// ===== LZMA压缩/解压工具 =====
class LZMACompressor {
public:
    // 压缩数据
    static std::vector<uint8_t> compress(const void* data, size_t size) {
        const uint8_t* input = static_cast<const uint8_t*>(data);

        // 预分配输出缓冲区(最坏情况:比输入大)
        std::vector<uint8_t> output(size + size / 10 + 128);

        // LZMA编码器
        lzma_stream strm = LZMA_STREAM_INIT;
        lzma_ret ret = lzma_easy_encoder(&strm, 6, LZMA_CHECK_CRC64);

        if (ret != LZMA_OK) {
            throw std::runtime_error("LZMA encoder init failed");
        }

        strm.next_in = input;
        strm.avail_in = size;
        strm.next_out = output.data();
        strm.avail_out = output.size();

        ret = lzma_code(&strm, LZMA_FINISH);

        if (ret != LZMA_STREAM_END) {
            lzma_end(&strm);
            throw std::runtime_error("LZMA compression failed");
        }

        size_t compressed_size = output.size() - strm.avail_out;
        output.resize(compressed_size);

        lzma_end(&strm);
        return output;
    }

    // 解压数据
    static std::vector<uint8_t> decompress(const void* data, size_t compressed_size,
                                          size_t original_size) {
        const uint8_t* input = static_cast<const uint8_t*>(data);
        std::vector<uint8_t> output(original_size);

        lzma_stream strm = LZMA_STREAM_INIT;
        lzma_ret ret = lzma_stream_decoder(&strm, UINT64_MAX, LZMA_CONCATENATED);

        if (ret != LZMA_OK) {
            throw std::runtime_error("LZMA decoder init failed");
        }

        strm.next_in = input;
        strm.avail_in = compressed_size;
        strm.next_out = output.data();
        strm.avail_out = output.size();

        ret = lzma_code(&strm, LZMA_FINISH);

        if (ret != LZMA_STREAM_END) {
            lzma_end(&strm);
            throw std::runtime_error("LZMA decompression failed");
        }

        lzma_end(&strm);
        return output;
    }
};

// ===== 通用序列化器 =====
template<typename T>
class Serializer {
public:
    // 序列化并压缩保存
    static bool save(const T& obj, const std::string& filepath) {
        try {
            // 1. 原始大小
            size_t size = sizeof(T);

            // 2. LZMA压缩
            auto compressed = LZMACompressor::compress(&obj, size);

            // 3. 写入文件(带头部)
            std::ofstream ofs(filepath, std::ios::binary);
            if (!ofs) return false;

            // 头部:魔数(4字节) + 原始大小(8字节) + 压缩大小(8字节)
            uint32_t magic = 0x4C5A4D41; // "LZMA"
            uint64_t original_size = size;
            uint64_t compressed_size = compressed.size();

            ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
            ofs.write(reinterpret_cast<const char*>(&original_size), sizeof(original_size));
            ofs.write(reinterpret_cast<const char*>(&compressed_size), sizeof(compressed_size));
            ofs.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());

            return ofs.good();
        } catch (const std::exception& e) {
            std::cerr << "Save failed: " << e.what() << std::endl;
            return false;
        }
    }

    // 读取并解压反序列化
    static bool load(T& obj, const std::string& filepath) {
        try {
            std::ifstream ifs(filepath, std::ios::binary);
            if (!ifs) return false;

            // 读取头部
            uint32_t magic;
            uint64_t original_size, compressed_size;

            ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
            ifs.read(reinterpret_cast<char*>(&original_size), sizeof(original_size));
            ifs.read(reinterpret_cast<char*>(&compressed_size), sizeof(compressed_size));

            // 验证魔数
            if (magic != 0x4C5A4D41) {
                throw std::runtime_error("Invalid file format");
            }

            // 验证大小
            if (original_size != sizeof(T)) {
                throw std::runtime_error("Size mismatch");
            }

            // 读取压缩数据
            std::vector<uint8_t> compressed(compressed_size);
            ifs.read(reinterpret_cast<char*>(compressed.data()), compressed_size);

            // 解压
            auto decompressed = LZMACompressor::decompress(
                compressed.data(), compressed_size, original_size
            );

            // 拷贝到对象
            std::memcpy(&obj, decompressed.data(), sizeof(T));

            return true;
        } catch (const std::exception& e) {
            std::cerr << "Load failed: " << e.what() << std::endl;
            return false;
        }
    }

    // 获取压缩率
    static double getCompressionRatio(const T& obj) {
        size_t original = sizeof(T);
        auto compressed = LZMACompressor::compress(&obj, original);
        return (double)compressed.size() / original;
    }
};

// ===== 使用示例 =====
/*
// 你的数据结构
struct MyData {
    double matrix[256][256];
    int id;
    char name[64];
};

// 保存
MyData data;
// ... 填充数据
Serializer<MyData>::save(data, "data.lzma");

// 加载
MyData loaded;
Serializer<MyData>::load(loaded, "data.lzma");

// 查看压缩率
double ratio = Serializer<MyData>::getCompressionRatio(data);
std::cout << "压缩率: " << (ratio * 100) << "%" << std::endl;
*/

#endif
