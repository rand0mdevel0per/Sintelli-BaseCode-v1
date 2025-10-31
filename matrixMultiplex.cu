//
// Created by ASUS on 9/29/2025.
//

#ifndef MATRIX_MULTIPLEX_CU
#define MATRIX_MULTIPLEX_CU

#include <cuda_runtime.h>

// 尝试包含CUTLASS，如果失败则使用基础CUDA实现
#ifdef CUTLASS_AVAILABLE
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/arch/arch.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#endif

// ==================== Double 精度矩阵乘法 ====================

// 简单的CUDA矩阵乘法实现，作为备用方案
__global__ void matmul_kernel_256x256(const double *A, const double *B, double *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 矩阵数据传输辅助函数 - 简化版本
__host__ bool copyMatrixToDevice(const double *host_matrix, double *device_matrix, int rows, int cols) {
    size_t size = rows * cols * sizeof(double);
    return cudaMemcpy(device_matrix, host_matrix, size, cudaMemcpyHostToDevice) == cudaSuccess;
}

__host__ bool copyMatrixToHost(const double *device_matrix, double *host_matrix, int rows, int cols) {
    size_t size = rows * cols * sizeof(double);
    return cudaMemcpy(host_matrix, device_matrix, size, cudaMemcpyDeviceToHost) == cudaSuccess;
}

// 矩阵初始化函数 - 简化版本
__host__ bool initMatrixOnDevice(double **device_matrix, int rows, int cols) {
    size_t size = rows * cols * sizeof(double);
    return cudaMalloc(device_matrix, size) == cudaSuccess;
}

__host__ void freeMatrixOnDevice(double *device_matrix) {
    if (device_matrix) cudaFree(device_matrix);
}

// 矩阵内存拷贝函数（设备到设备）- 简化版本
__host__ bool copyMatrixDeviceToDevice(const double *src_device_matrix, double *dst_device_matrix, int rows, int cols) {
    size_t size = rows * cols * sizeof(double);
    return cudaMemcpy(dst_device_matrix, src_device_matrix, size, cudaMemcpyDeviceToDevice) == cudaSuccess;
}

// 简化的矩阵乘法接口：C = A * B (优先使用CUTLASS，备选基础CUDA实现)
__host__ bool matmul_double(const double *device_A, const double *device_B, double *device_C, int M, int N, int K) {
#ifdef CUTLASS_AVAILABLE
    // 使用CUTLASS实现高性能矩阵乘法
    try {
        // CUTLASS 256x256 双精度矩阵乘法定义
        using CutlassGemmDouble256 = cutlass::gemm::device::Gemm<
            double, cutlass::layout::ColumnMajor,
            double, cutlass::layout::ColumnMajor,
            double, cutlass::layout::ColumnMajor,
            double,
            cutlass::arch::OpClassSimt,
            cutlass::arch::Sm80,  // 使用sm80架构，兼容sm89/sm90
            cutlass::gemm::GemmShape<128, 128, 8>,
            cutlass::gemm::GemmShape<64, 64, 8>,
            cutlass::gemm::GemmShape<1, 1, 1>,
            cutlass::epilogue::thread::LinearCombination<double, 1, double, double>,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
            2
        >;

        const double alpha = 1.0;
        const double beta = 0.0;

        // 配置GEMM参数
        typename CutlassGemmDouble256::Arguments args(
            {M, N, K}, // 问题规模
            {device_A, M}, // A矩阵和leading dimension
            {device_B, K}, // B矩阵和leading dimension
            {nullptr, M}, // 源C矩阵(不使用)
            {device_C, M}, // 目标C矩阵
            {alpha, beta} // alpha和beta系数
        );

        // 初始化GEMM操作
        CutlassGemmDouble256 gemm_op;

        // 分配workspace(如果需要)
        size_t workspace_size = CutlassGemmDouble256::get_workspace_size(args);
        void *workspace_ptr = nullptr;
        if (workspace_size > 0 && cudaMalloc(&workspace_ptr, workspace_size) != cudaSuccess) {
            // 如果workspace分配失败，回退到基础实现
            goto fallback;
        }

        // 初始化并执行
        cutlass::Status status = gemm_op.initialize(args, workspace_ptr);
        if (status != cutlass::Status::kSuccess) {
            if (workspace_ptr) cudaFree(workspace_ptr);
            goto fallback;
        }

        status = gemm_op();
        cudaDeviceSynchronize();

        if (workspace_ptr) cudaFree(workspace_ptr);
        return (status == cutlass::Status::kSuccess);
    } catch (...) {
        // 如果CUTLASS执行失败，回退到基础实现
        goto fallback;
    }
    
    fallback:
#endif
    // 使用基础CUDA内核实现矩阵乘法（备选方案）
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    
    matmul_kernel_256x256<<<gridSize, blockSize>>>(device_A, device_B, device_C, M);
    cudaDeviceSynchronize();
    
    return cudaGetLastError() == cudaSuccess;
}

// 重载版本，使用默认参数
__host__ bool matmul_double(const double *device_A, const double *device_B, double *device_C) {
    return matmul_double(device_A, device_B, device_C, 256, 256, 256);
}

// ==================== Float 精度矩阵乘法 ====================

// 简单的CUDA矩阵乘法实现，作为备用方案 (float版本)
__global__ void matmul_kernel_256x256_float(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 简化的矩阵乘法接口：C = A * B (优先使用CUTLASS，备选基础CUDA实现)
__host__ bool matmul_float(const float *device_A, const float *device_B, float *device_C, int M, int N, int K) {
#ifdef CUTLASS_AVAILABLE
    // 使用CUTLASS实现高性能矩阵乘法
    try {
        // CUTLASS 256x256 单精度矩阵乘法定义
        using CutlassGemmFloat256 = cutlass::gemm::device::Gemm<
            float, cutlass::layout::ColumnMajor,
            float, cutlass::layout::ColumnMajor,
            float, cutlass::layout::ColumnMajor,
            float,
            cutlass::arch::OpClassSimt,
            cutlass::arch::Sm80,  // 使用sm80架构，兼容sm89/sm90
            cutlass::gemm::GemmShape<128, 128, 8>,
            cutlass::gemm::GemmShape<64, 64, 8>,
            cutlass::gemm::GemmShape<1, 1, 1>,
            cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
            2
        >;

        const float alpha = 1.0f;
        const float beta = 0.0f;

        // 配置GEMM参数
        typename CutlassGemmFloat256::Arguments args(
            {M, N, K}, // 问题规模
            {device_A, M}, // A矩阵和leading dimension
            {device_B, K}, // B矩阵和leading dimension
            {nullptr, M}, // 源C矩阵(不使用)
            {device_C, M}, // 目标C矩阵
            {alpha, beta} // alpha和beta系数
        );

        // 初始化GEMM操作
        CutlassGemmFloat256 gemm_op;

        // 分配workspace(如果需要)
        size_t workspace_size = CutlassGemmFloat256::get_workspace_size(args);
        void *workspace_ptr = nullptr;
        if (workspace_size > 0 && cudaMalloc(&workspace_ptr, workspace_size) != cudaSuccess) {
            // 如果workspace分配失败，回退到基础实现
            goto fallback_float;
        }

        // 初始化并执行
        cutlass::Status status = gemm_op.initialize(args, workspace_ptr);
        if (status != cutlass::Status::kSuccess) {
            if (workspace_ptr) cudaFree(workspace_ptr);
            goto fallback_float;
        }

        status = gemm_op();
        cudaDeviceSynchronize();

        if (workspace_ptr) cudaFree(workspace_ptr);
        return (status == cutlass::Status::kSuccess);
    } catch (...) {
        // 如果CUTLASS执行失败，回退到基础实现
        goto fallback_float;
    }
    
    fallback_float:
#endif
    // 使用基础CUDA内核实现矩阵乘法（备选方案）
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    
    matmul_kernel_256x256_float<<<gridSize, blockSize>>>(device_A, device_B, device_C, M);
    cudaDeviceSynchronize();
    
    return cudaGetLastError() == cudaSuccess;
}

// 重载版本，使用默认参数
__host__ bool matmul_float(const float *device_A, const float *device_B, float *device_C) {
    return matmul_float(device_A, device_B, device_C, 256, 256, 256);
}

#endif // MATRIX_MULTIPLEX_CU