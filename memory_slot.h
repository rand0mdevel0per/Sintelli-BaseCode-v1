// ========== memory_slot.h ==========
#ifndef MEMORY_SLOT_H
#define MEMORY_SLOT_H

#include <string>
#include <cstring>
#include <istream>
#include "hasher.h"

/**
 * @brief 记忆槽位结构（存储到ExternalStorage）
 */
struct MemorySlot {
    char memory_id[128]{};           // 记忆ID
    std::string content;
    double importance{};             // 重要性
    uint64_t timestamp{};            // 时间戳
    char context[512]{};             // 上下文描述
    int access_count{};              // 访问次数
    double embedding[768]{};         // E5特征向量（用于快速检索）

    MemorySlot() {
        memset(this, 0, sizeof(MemorySlot));
    }

    // 用于ExternalStorage的hash
    [[nodiscard]] std::string hash() const {
        // 简单hash：使用memory_id
        return sha256_hash<MemorySlot>(*this);
    }

    // 序列化支持
    bool serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(this), sizeof(MemorySlot));
        return os.good();
    }

    bool deserialize(std::istream& is) {
        is.read(reinterpret_cast<char*>(this), sizeof(MemorySlot));
        return is.good();
    }
};

#endif // MEMORY_SLOT_H