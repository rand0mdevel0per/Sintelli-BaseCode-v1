//
// Created by Gemini on 2025/10/02.
//
// 知识片段引擎(KFE)工具集：包含高效哈希算法
//
#pragma once
#ifndef KFE_UTILS_CUH
#define KFE_UTILS_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <iostream>
#include <type_traits> // 用于检查类型是否为 POD (Plain Old Data)

/**
 * @brief CUDA设备端高效 MurmurHash3 32位实现
 * * MurmurHash3 是一个非加密哈希函数，以其极快的速度和优秀的统计特性而闻名，
 * 非常适合在高性能计算环境（如GPU）中用作查找表键或数据校验。
 * * 适用于将任意字节数据 (如 knowledge ID, 坐标数组等) 映射为一个 32 位哈希值。
 * * @param key 输入字节数组的指针
 * @param len 输入字节数组的长度 (字节数)
 * @param seed 哈希种子
 * @return uint32_t 计算得到的 32 位哈希值
 */
__device__ __forceinline__ uint32_t MurmurHash3_32(const void* key, int len, uint32_t seed) {
    const uint8_t* data = (const uint8_t*)key;
    const int nblocks = len / 4;
    uint32_t h1 = seed;

    // FNV-like constants used in MurmurHash3
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873537;

    // -----------------------------------------------------------------------------
    // 1. 处理4字节块 (Process blocks)
    // -----------------------------------------------------------------------------
    const uint32_t* blocks = (const uint32_t*)(data);

    for (int i = 0; i < nblocks; i++) {
        uint32_t k1;
        // 使用 __ldg 保证对全局内存的快速、非缓存加载（如果适用）
        // 也可以直接使用 *blocks++ 来读取数据
        k1 = blocks[i];

        // 乘法、旋转和异或混合操作
        k1 *= c1;
        k1 = (k1 << 15) | (k1 >> (32 - 15)); // 循环左移15位
        k1 *= c2;

        h1 ^= k1;
        h1 = (h1 << 13) | (h1 >> (32 - 13)); // 循环左移13位
        h1 = h1 * 5 + 0xe6546b64;
    }

    // -----------------------------------------------------------------------------
    // 2. 处理尾部字节 (Handle the remainder)
    // -----------------------------------------------------------------------------
    const uint8_t* tail = (const uint8_t*)(data + nblocks * 4);
    uint32_t k1 = 0;

    switch (len & 3) {
        case 3: k1 ^= tail[2] << 16;
        case 2: k1 ^= tail[1] << 8;
        case 1: k1 ^= tail[0];
                k1 *= c1;
                k1 = (k1 << 15) | (k1 >> (32 - 15));
                k1 *= c2;
                h1 ^= k1;
                break;
        default: break;
    }

    // -----------------------------------------------------------------------------
    // 3. 最终混合 (Finalization)
    // -----------------------------------------------------------------------------
    h1 ^= len;

    // Finalization mix - C++实现中常用的快速位操作
    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >> 16;

    return h1;
}

#endif // KFE_UTILS_CUH


#ifndef SHA256_UTILS
#define SHA256_UTILS
// 移除重复的头文件保护，合并到KFE_UTILS_CUH中

// =================================================================
// 核心 SHA-256 实现 (基于 C++ 标准库)
// 注意: 这是 CPU 端的实现，不使用 CUDA 设备端函数
// =================================================================

namespace SHA256 {

    // SHA-256 常量 K
    constexpr uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    // 初始哈希值 H (每次调用时重新初始化，避免状态污染)
    inline void initialize_H(uint32_t H_state[8]) {
        H_state[0] = 0x6a09e667;
        H_state[1] = 0xbb67ae85;
        H_state[2] = 0x3c6ef372;
        H_state[3] = 0xa54ff53a;
        H_state[4] = 0x510e527f;
        H_state[5] = 0x9b05688c;
        H_state[6] = 0x1f83d9ab;
        H_state[7] = 0x5be0cd19;
    }

    // 宏定义: 右旋 (Rotate Right)
    #define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

    // SHA-256 核心函数 (Sigma, Choice, Majority)
    #define SHR(x, n) ((x) >> (n))
    #define SIGMA0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
    #define SIGMA1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
    #define sig0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ SHR(x, 3))
    #define sig1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ SHR(x, 10))
    #define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
    #define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))

    /**
     * @brief SHA-256 核心压缩函数。处理一个 512-bit (64-byte) 的数据块。
     * @param W 消息调度数组 (64个32位字)
     * @param H_state 当前的哈希状态 (8个32位字)
     */
    void process_block(const uint32_t W[64], uint32_t H_state[8]) {
        uint32_t a = H_state[0];
        uint32_t b = H_state[1];
        uint32_t c = H_state[2];
        uint32_t d = H_state[3];
        uint32_t e = H_state[4];
        uint32_t f = H_state[5];
        uint32_t g = H_state[6];
        uint32_t h = H_state[7];

        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = SIGMA1(e);
            uint32_t ch = CH(e, f, g);
            uint32_t temp1 = h + S1 + ch + K[i] + W[i];

            uint32_t S0 = SIGMA0(a);
            uint32_t maj = MAJ(a, b, c);
            uint32_t temp2 = S0 + maj;

            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        // 更新哈希值
        H_state[0] += a;
        H_state[1] += b;
        H_state[2] += c;
        H_state[3] += d;
        H_state[4] += e;
        H_state[5] += f;
        H_state[6] += g;
        H_state[7] += h;
    }

    /**
     * @brief 将输入字节流转换为 SHA-256 哈希字符串。
     * @param input_bytes 待哈希的字节向量。
     * @return std::string 64字符的十六进制哈希值。
     */
    std::string hash(const std::vector<uint8_t>& input_bytes) {
        // 1. 初始化哈希状态 (每次重新初始化)
        uint32_t H_state[8];
        initialize_H(H_state);

        // 2. 消息填充
        std::vector<uint8_t> padded_data = input_bytes;

        // 消息长度 (位)
        uint64_t L = (uint64_t)input_bytes.size() * 8;

        // 追加 '1' 位 (即 0x80 字节)
        padded_data.push_back(0x80);

        // 如果当前长度模 64 字节大于 56 字节，则需要增加一个额外的 64 字节块
        size_t current_len_mod_64 = padded_data.size() % 64;
        size_t padding_len = 0;
        if (current_len_mod_64 <= 56) {
            padding_len = 56 - current_len_mod_64;
        } else { // 64 - (len % 64) + 56
            padding_len = 64 - current_len_mod_64 + 56;
        }

        // 填充零字节
        for (size_t i = 0; i < padding_len; ++i) {
            padded_data.push_back(0x00);
        }

        // 追加原始消息长度 L (64 位，大端序)
        for (int i = 0; i < 8; ++i) {
            padded_data.push_back((uint8_t)(L >> (56 - i * 8)));
        }

        // 3. 处理数据块
        size_t num_blocks = padded_data.size() / 64;
        for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            uint32_t W[64];
            const uint8_t* block_start = &padded_data[block_idx * 64];

            // a) 将 512-bit 数据块解析为 16 个 32-bit 大端序字 W[0..15]
            for (int i = 0; i < 16; ++i) {
                // 大端序转换: 字节 [0, 1, 2, 3] -> (0 << 24 | 1 << 16 | 2 << 8 | 3)
                W[i] = ((uint32_t)block_start[i * 4] << 24) |
                       ((uint32_t)block_start[i * 4 + 1] << 16) |
                       ((uint32_t)block_start[i * 4 + 2] << 8) |
                       ((uint32_t)block_start[i * 4 + 3]);
            }

            // b) 消息调度：扩展 W[16..63]
            for (int i = 16; i < 64; ++i) {
                uint32_t s0 = sig0(W[i - 15]);
                uint32_t s1 = sig1(W[i - 2]);
                W[i] = W[i - 16] + s0 + W[i - 7] + s1;
            }

            // c) 压缩函数
            process_block(W, H_state);
        }

        // 4. 输出结果 (8个32位字 -> 64个十六进制字符)
        std::stringstream ss;
        ss << std::hex << std::setfill('0');
        for (int i = 0; i < 8; ++i) {
            ss << std::setw(8) << H_state[i];
        }

        return ss.str();
    }

} // namespace SHA256

/**
 * @brief 模板函数: 计算任何 POD 类型或其指针所指向数据的 SHA-256 哈希值。
 * * 使用 reinterpret_cast 获取数据的字节表示。
 * * 警告: 仅对 POD/标准布局类型安全。对包含虚函数或指针的复杂类可能不适用。
 * @tparam T 待哈希的类型。
 * @param data 待哈希的输入数据。
 * @return std::string 64字符的十六进制 SHA-256 哈希值。
 */
template<typename T>
std::string sha256_hash(const T& data) {
    static_assert(std::is_standard_layout<T>::value || std::is_arithmetic<T>::value,
                  "sha256_hash can only be used with standard layout or arithmetic types to ensure hash stability and security.");

    // 将输入数据的内存视图转换为字节向量
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&data);
    std::vector<uint8_t> input_bytes(bytes, bytes + sizeof(T));

    // 调用核心哈希函数
    return SHA256::hash(input_bytes);
}

// 针对 char* 的重载
template<>
inline std::string sha256_hash<char*>(char* const& data) {
    if (!data) return SHA256::hash(std::vector<uint8_t>()); // 空指针返回空哈希

    size_t len = std::strlen(data);
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data);
    std::vector<uint8_t> input_bytes(bytes, bytes + len);

    return SHA256::hash(input_bytes);
}

// 针对 const char* 的重载
template<>
inline std::string sha256_hash<const char*>(const char* const& data) {
    if (!data) return SHA256::hash(std::vector<uint8_t>()); // 空指针返回空哈希

    size_t len = std::strlen(data);
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data);
    std::vector<uint8_t> input_bytes(bytes, bytes + len);

    return SHA256::hash(input_bytes);
}

// 针对 std::string 的重载
template<>
inline std::string sha256_hash<std::string>(const std::string& data) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data.data());
    std::vector<uint8_t> input_bytes(bytes, bytes + data.length());

    return SHA256::hash(input_bytes);
}

// =================================================================
// 示例用法: (如果需要测试，可以放在 main.cpp 或单元测试中)
// =================================================================
/*
#include <iostream>

struct MyStruct {
    double value;
    int index;
    // 注意: 确保结构体是 POD 或 Standard Layout
};

void test_sha256() {
    // 1. 哈希基本类型
    int number = 12345;
    std::string hash_num = sha256_hash(number);
    std::cout << "Hash(int 12345): " << hash_num << std::endl;

    // 2. 哈希字符串
    std::string text = "Hello world!";
    std::string hash_text = sha256_hash(text);
    std::cout << "Hash(\"Hello world!\"): " << hash_text << std::endl;
    // 预期结果: c0535e4be2b79ffd93291305436bf889314e4a3faec05ecffcbb7df31ad9e51a

    // 3. 哈希结构体
    MyStruct data = {3.14159, 42};
    std::string hash_struct = sha256_hash(data);
    std::cout << "Hash(MyStruct): " << hash_struct << std::endl;
}
*/

#endif

