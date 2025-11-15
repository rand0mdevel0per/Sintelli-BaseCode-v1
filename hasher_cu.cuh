//
// Created by ASUS on 11/9/2025.
//

#ifndef SRC_HASHER_CU_CUH
#define SRC_HASHER_CU_CUH

// ---------------------- GPU-only SHA-256 (pure device) ----------------------
// Put this inside namespace SHA256 or at top-level in hasher.h

// Device-side rotate/right helpers
#if defined(__CUDA_ARCH__)
__device__ __forceinline__ uint32_t dev_ROTR(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}
__device__ __forceinline__ uint32_t dev_SHR(uint32_t x, int n) {
    return x >> n;
}
__device__ __forceinline__ uint32_t dev_SIGMA0(uint32_t x) { return dev_ROTR(x,2) ^ dev_ROTR(x,13) ^ dev_ROTR(x,22); }
__device__ __forceinline__ uint32_t dev_SIGMA1(uint32_t x) { return dev_ROTR(x,6) ^ dev_ROTR(x,11) ^ dev_ROTR(x,25); }
__device__ __forceinline__ uint32_t dev_sig0(uint32_t x) { return dev_ROTR(x,7) ^ dev_ROTR(x,18) ^ dev_SHR(x,3); }
__device__ __forceinline__ uint32_t dev_sig1(uint32_t x) { return dev_ROTR(x,17) ^ dev_ROTR(x,19) ^ dev_SHR(x,10); }
__device__ __forceinline__ uint32_t dev_CH(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
__device__ __forceinline__ uint32_t dev_MAJ(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }

// K constants (same as host) - 在头文件中仅声明，在 hasher.cu 中定义
extern __device__ __constant__ const uint32_t dev_K[64];

// Initialize H (device inline)
__device__ __forceinline__ void dev_initialize_H(uint32_t H[8]) {
    H[0] = 0x6a09e667u; H[1] = 0xbb67ae85u; H[2] = 0x3c6ef372u; H[3] = 0xa54ff53au;
    H[4] = 0x510e527fu; H[5] = 0x9b05688cu; H[6] = 0x1f83d9abu; H[7] = 0x5be0cd19u;
}

// Process single 512-bit block (block is pointer to 64 bytes)
__device__ __forceinline__ void dev_process_block(const uint8_t block[64], uint32_t H[8]) {
    uint32_t W[64];
    // big-endian decode
    for (int i = 0; i < 16; ++i) {
        W[i] = (static_cast<uint32_t>(block[i*4]) << 24) |
               (static_cast<uint32_t>(block[i*4 + 1]) << 16) |
               (static_cast<uint32_t>(block[i*4 + 2]) << 8) |
               (static_cast<uint32_t>(block[i*4 + 3]));
    }
    for (int i = 16; i < 64; ++i) {
        W[i] = dev_sig0(W[i-15]) + W[i-16] + dev_sig1(W[i-2]) + W[i-7];
    }

    uint32_t a = H[0], b = H[1], c = H[2], d = H[3];
    uint32_t e = H[4], f = H[5], g = H[6], h = H[7];

    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = dev_SIGMA1(e);
        uint32_t ch = dev_CH(e, f, g);
        uint32_t temp1 = h + S1 + ch + dev_K[i] + W[i];
        uint32_t S0 = dev_SIGMA0(a);
        uint32_t maj = dev_MAJ(a, b, c);
        uint32_t temp2 = S0 + maj;
        h = g; g = f; f = e;
        e = d + temp1;
        d = c; c = b; b = a;
        a = temp1 + temp2;
    }

    H[0] += a; H[1] += b; H[2] += c; H[3] += d;
    H[4] += e; H[5] += f; H[6] += g; H[7] += h;
}

// Top-level device digest: raw 32 bytes (big-endian)
__device__ void sha256_digest_device(const uint8_t* data, size_t len, uint8_t out32[32]) {
    uint32_t H[8];
    dev_initialize_H(H);

    // 1) process full 64-byte blocks
    size_t nblocks = len / 64;
    for (size_t i = 0; i < nblocks; ++i) {
        dev_process_block(&data[i * 64], H);
    }

    // 2) handle tail + padding (may produce 1 or 2 blocks)
    uint8_t block[64];
    size_t tail_len = len - nblocks * 64;
    // copy tail bytes
    for (size_t i = 0; i < tail_len; ++i) block[i] = data[nblocks * 64 + i];
    // append 0x80
    block[tail_len] = 0x80u;
    // zero the rest
    for (size_t i = tail_len + 1; i < 64; ++i) block[i] = 0u;

    const uint64_t bitlen = static_cast<uint64_t>(len) * 8ull;
    if (tail_len <= 55) {
        // can fit length in this single block
        // write 64-bit big-endian length at bytes 56..63
        for (int i = 0; i < 8; ++i) {
            block[56 + i] = static_cast<uint8_t>((bitlen >> ((7 - i) * 8)) & 0xFFu);
        }
        dev_process_block(block, H);
    } else {
        // need two blocks: current block (already has 0x80 and zeros), process it
        dev_process_block(block, H);
        // form second block with zeros and length at end
        for (int i = 0; i < 56; ++i) block[i] = 0u;
        for (int i = 0; i < 8; ++i) {
            block[56 + i] = static_cast<uint8_t>((bitlen >> ((7 - i) * 8)) & 0xFFu);
        }
        dev_process_block(block, H);
    }

    // write H to out32 as big-endian bytes
    for (int i = 0; i < 8; ++i) {
        out32[i*4 + 0] = static_cast<uint8_t>((H[i] >> 24) & 0xFFu);
        out32[i*4 + 1] = static_cast<uint8_t>((H[i] >> 16) & 0xFFu);
        out32[i*4 + 2] = static_cast<uint8_t>((H[i] >> 8) & 0xFFu);
        out32[i*4 + 3] = static_cast<uint8_t>((H[i] >> 0) & 0xFFu);
    }
}

// Optional: device helper to output hex string (null-terminated 65 chars)
__device__ void sha256_digest_hex_device(const uint8_t* data, size_t len, char out_hex[65]) {
    uint8_t raw[32];
    sha256_digest_device(data, len, raw);
    const char hexmap[] = "0123456789abcdef";
    for (int i = 0; i < 32; ++i) {
        out_hex[i*2 + 0] = hexmap[(raw[i] >> 4) & 0xF];
        out_hex[i*2 + 1] = hexmap[raw[i] & 0xF];
    }
    out_hex[64] = '\0';
}

#endif // device-only section


#endif //SRC_HASHER_CU_CUH