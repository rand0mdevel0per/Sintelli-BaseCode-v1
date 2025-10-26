//
// Created by ASUS on 9/29/2025.
//

#ifndef MATRIX_MULTIPLEX_CPP
#define MATRIX_MULTIPLEX_CPP

#include <cuda_runtime.h>
#include "cutlass/include/cutlass/gemm/device/gemm.h"
#include "cutlass/include/cutlass/arch/arch.h"
#include "cutlass/include/cutlass/layout/matrix.h"
#include "cutlass/include/cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h"

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

// CUTLASS 256x256 双精度矩阵乘法定义
using CutlassGemmDouble256 = cutlass::gemm::device::Gemm<
    double, cutlass::layout::ColumnMajor,
    double, cutlass::layout::ColumnMajor,
    double, cutlass::layout::ColumnMajor,
    double,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<double, 1, double, double>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
>;

/**

 * 简化的矩阵乘法接口：C = A * B

 *

 * @param device_A 设备端A矩阵指针 (M x K)

 * @param device_B 设备端B矩阵指针 (K x N)

 * @param device_C 设备端C矩阵指针 (M x N)

 * @param M 矩阵A的行数

 * @param N 矩阵B的列数

 * @param K 矩阵A的列数/矩阵B的行数

 * @return true if成功, false if失败

 */

__host__ __device__ inline bool matmul_double(

    const double *device_A,

    const double *device_B,

    double *device_C,

    int M = 256,

    int N = 256,

    int K = 256

) {
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
        return false;
    }


    // 初始化并执行

    cutlass::Status status = gemm_op.initialize(args, workspace_ptr);

    if (status != cutlass::Status::kSuccess) {
        if (workspace_ptr) cudaFree(workspace_ptr);

        return false;
    }


    status = gemm_op();

    cudaDeviceSynchronize();

    if (workspace_ptr) cudaFree(workspace_ptr);

    return (status == cutlass::Status::kSuccess);
}

// ==================== Float 精度矩阵乘法 ====================

using CutlassGemmFloat256 = cutlass::gemm::device::Gemm<
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
>;

__host__ __device__ inline bool matmul_float(

    const float *device_A,

    const float *device_B,

    float *device_C,

    int M = 256,

    int N = 256,

    int K = 256

) {
    const float alpha = 1.0f;

    const float beta = 0.0f;


    typename CutlassGemmFloat256::Arguments args(

        {M, N, K},

        {device_A, M},

        {device_B, K},

        {nullptr, M},

        {device_C, M},

        {alpha, beta}

    );


    CutlassGemmFloat256 gemm_op;

    size_t workspace_size = CutlassGemmFloat256::get_workspace_size(args);

    void *workspace_ptr = nullptr;


    if (workspace_size > 0 && cudaMalloc(&workspace_ptr, workspace_size) != cudaSuccess) {
        return false;
    }


    cutlass::Status status = gemm_op.initialize(args, workspace_ptr);

    if (status != cutlass::Status::kSuccess) {
        if (workspace_ptr) cudaFree(workspace_ptr);

        return false;
    }


    status = gemm_op();

    cudaDeviceSynchronize();

    if (workspace_ptr) cudaFree(workspace_ptr);

    return (status == cutlass::Status::kSuccess);
}

// ==================== 使用示例 ====================
/*
// 在Host端分配并初始化数据
double *h_A, *h_B, *h_C;
double *d_A, *d_B, *d_C;

// 分配device内存
cudaMalloc(&d_A, 256 * 256 * sizeof(double));
cudaMalloc(&d_B, 256 * 256 * sizeof(double));
cudaMalloc(&d_C, 256 * 256 * sizeof(double));

// 拷贝数据到device
cudaMemcpy(d_A, h_A, 256 * 256 * sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, 256 * 256 * sizeof(double), cudaMemcpyHostToDevice);

// 执行矩阵乘法：C = A * B
if (matmul_double(d_A, d_B, d_C)) {
    // 成功
    cudaMemcpy(h_C, d_C, 256 * 256 * sizeof(double), cudaMemcpyDeviceToHost);
} else {
    // 失败处理
}

// 清理
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
*/

#endif // MATRIX_MULTIPLEX_CPP
