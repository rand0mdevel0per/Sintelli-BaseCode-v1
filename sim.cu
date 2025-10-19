//
// Created by ASUS on 10/3/2025.
//

#ifndef VECTOR_SIMILARITY_CUH
#define VECTOR_SIMILARITY_CUH

#include <cuda_runtime.h>
#include <cmath>    // for sqrt
#include <cfloat>   // for DBL_EPSILON

#define N_DIM 65536     // 256 * 256 维
#define BLOCK_SIZE 1024 // 线程块大小 (推荐使用 256, 512, 或 1024)

// =========================================================================
// 优化后的 Block-wise Reduction Function (使用 Warp Shuffle Intrinsics)
// =========================================================================

/**
 * @brief 使用 Warp Shuffle Intrinsics 对线程块内的 double 数组进行规约求和。
 * * * 规约算法结合了 Warp Shuffle (Warp 内) 和 Shared Memory (Warp 间)。
 * * @param sum 线程本地的 partial sum。
 * @return double 线程块的总和 (只有 threadIdx.x == 0 的线程返回正确值)。
 */
__device__ double blockReduceSum(double sum) {
    // 1. Warp 内部规约 (寄存器级别)
    // 假设 Warp Size = 32
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // 2. Warp 之间规约 (使用 Shared Memory)
    __shared__ double sdata[32]; // 足够存储 BLOCK_SIZE / 32 = 32 个 Warp Leader 的结果

    unsigned int warp_id = threadIdx.x / 32;
    unsigned int lane_id = threadIdx.x % 32;

    // Warp Leader (lane_id == 0) 将其 Warp 的结果写入 Shared Memory
    if (lane_id == 0) {
        sdata[warp_id] = sum;
    }

    __syncthreads();

    // 3. Block Leader (threadIdx.x == 0) 完成最终规约
    if (warp_id == 0 && lane_id < (BLOCK_SIZE / 32)) {
        // 在第一个 Warp 内部对 Shared Memory 中的值再次进行 Shuffle 规约
        for (int offset = (BLOCK_SIZE / 64); offset > 0; offset /= 2) {
            sdata[lane_id] += __shfl_down_sync(0xFFFFFFFF, sdata[lane_id], offset);
        }
    }

    // 最终结果在 sdata[0]
    return sdata[0];
}


// =========================================================================
// 256x256 (65536维) 向量的并行余弦相似度计算 Kernel
// =========================================================================

/**
 * @brief 高效并行计算两个 65536 维向量的余弦相似度。
 * * * 这是在 Host 端启动 Kernel 来批量计算相似度的推荐方法。
 * * 必须使用一个线程块 (BLOCK_SIZE=1024) 且一个 Grid (dim3(1,1,1)) 来调用。
 * * @param A_flat 第一个向量的平坦指针 (长度 N_DIM)
 * @param B_flat 第二个向量的平坦指针 (长度 N_DIM)
 * @param result_out 输出余弦相似度结果的指针 (长度 1)
 */
__global__ void cosineSimilarityKernel(
    const double* __restrict__ A_flat,
    const double* __restrict__ B_flat,
    double* __restrict__ result_out
) {
    const int tid = threadIdx.x;
    const int numElementsPerThread = N_DIM / BLOCK_SIZE; // 65536 / 1024 = 64

    // 1. 线程本地计算部分和
    double partial_dot = 0.0;
    double partial_norm_A_sq = 0.0;
    double partial_norm_B_sq = 0.0;

    // 循环迭代，每个线程处理 64 个元素
    for (int i = 0; i < numElementsPerThread; ++i) {
        int index = tid + i * BLOCK_SIZE;
        double val_A = A_flat[index];
        double val_B = B_flat[index];

        // 使用 FMA (Fused Multiply-Add) 提升精度和性能
        partial_dot = __fma_rn(val_A, val_B, partial_dot);
        partial_norm_A_sq = __fma_rn(val_A, val_A, partial_norm_A_sq);
        partial_norm_B_sq = __fma_rn(val_B, val_B, partial_norm_B_sq);
    }

    // 2. 块级规约 (所有线程协作完成求和)
    __syncthreads();

    double total_dot = blockReduceSum(partial_dot);
    double total_norm_A_sq = blockReduceSum(partial_norm_A_sq);
    double total_norm_B_sq = blockReduceSum(partial_norm_B_sq);

    // 3. 线程 0 完成最终计算并写入结果
    if (tid == 0) {
        double norm_A = sqrt(total_norm_A_sq);
        double norm_B = sqrt(total_norm_B_sq);

        double product_of_norms = norm_A * norm_B;

        // 避免除以零和浮点数精度问题
        if (product_of_norms < DBL_EPSILON) {
            *result_out = 0.0;
        } else {
            *result_out = total_dot / product_of_norms;
        }
    }
}


// =========================================================================
// 兼容性函数：原始的单线程版本 (供 Neuron.cu 内部逻辑使用)
// =========================================================================

/**
 * @brief 单线程计算两个 256x256 矩阵的余弦相似度。
 * * * 此函数是为了兼容 Neuron 内部的 __device__ 调用，效率低于 Kernel 版本。
 */
__device__ double calculate_cosine_similarity_single_thread(
    const double A[256][256],
    const double B[256][256]
) {
    double dot_product = 0.0;
    double norm_A_sq = 0.0;
    double norm_B_sq = 0.0;

    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            double val_A = A[i][j];
            double val_B = B[i][j];

            dot_product = __fma_rn(val_A, val_B, dot_product);
            norm_A_sq = __fma_rn(val_A, val_A, norm_A_sq);
            norm_B_sq = __fma_rn(val_B, val_B, norm_B_sq);
        }
    }

    double norm_A = sqrt(norm_A_sq);
    double norm_B = sqrt(norm_B_sq);
    double product_of_norms = norm_A * norm_B;

    if (product_of_norms == 0.0) {
        return 0.0;
    }

    return dot_product / product_of_norms;
}


#endif // VECTOR_SIMILARITY_CUH

//
// Created by Gemini on 10/3/2025.
//
// CUDA Device Utility functions for vector/matrix operations.
//

#ifndef SRC_UTILS_CUH
#define SRC_UTILS_CUH

#include <cmath> // 引入数学库，用于 sqrt 函数

/**
 * @brief 计算两个 16x16 双精度矩阵之间的余弦相似度。
 * * 矩阵被视为扁平的 256 维向量。这是用于比较 KFE_STM_Slot::conv16 字段的关键函数。
 * * @param A 第一个 16x16 矩阵的指针。
 * @param B 第二个 16x16 矩阵的指针。
 * @return 矩阵 A 和 B 之间的余弦相似度 [-1.0, 1.0]。
 */
__device__ double cosineSimilarity16x16(const double A[16][16], const double B[16][16]) {
    double dot_product = 0.0;
    double norm_a_sq = 0.0; // A的L2范数的平方 (用于开方)
    double norm_b_sq = 0.0; // B的L2范数的平方 (用于开方)

    // 迭代 16x16 = 256 个元素
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            double a_val = A[i][j];
            double b_val = B[i][j];

            // 累加点积 (分子)
            dot_product += a_val * b_val;

            // 累加平方和 (分母的组成部分)
            norm_a_sq += a_val * a_val;
            norm_b_sq += b_val * b_val;
        }
    }

    // 计算范数 (分母)
    double norm_a = sqrt(norm_a_sq);
    double norm_b = sqrt(norm_b_sq);

    // 如果任一范数接近零，则相似度未定义 (即其中一个矩阵是零向量)，返回 0.0。
    // 使用一个小的 epsilon 来防止浮点数误差。
    if (norm_a < 1e-12 || norm_b < 1e-12) {
        return 0.0;
    }

    // 计算最终的余弦相似度
    double similarity = dot_product / (norm_a * norm_b);

    // 限制结果在 [-1.0, 1.0] 范围内，以处理极小的浮点误差
    // 避免返回 1.0000000000000002 这样的值
    if (similarity > 1.0) return 1.0;
    if (similarity < -1.0) return -1.0;

    return similarity;
}

#endif // SRC_UTILS_CUH
