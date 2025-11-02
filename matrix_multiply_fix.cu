#ifndef MATRIX_MULTIPLY_FIX_CU
#define MATRIX_MULTIPLY_FIX_CU

#include "Neuron.cu"

// 修正后的矩阵乘法实现：C = input_matrix * n.P_Matrix
// 每个线程计算输出矩阵的一个元素
__device__ void fixed_matrix_multiply(
    double input_matrix[256][256],
    double P_Matrix[256][256],
    double *transformed_input,
    ull tid
) {
    // 每个线程计算输出矩阵的一个元素
    int tx = tid % 256;
    int ty = tid / 256;

    if (ty < 256 && tx < 256) {
        double sum = 0.0;
        // 每个线程计算输出矩阵的一个元素
        for (int k = 0; k < 256; k++) {
            sum += input_matrix[ty][k] * P_Matrix[k][tx];
        }
        transformed_input[ty * 256 + tx] = sum;
    }
}

#endif // MATRIX_MULTIPLY_FIX_CU