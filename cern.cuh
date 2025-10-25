//
// Created by ASUS on 10/2/2025.
//

#ifndef SRC_CERN_CUH
#define SRC_CERN_CUH

// ===== 卷积核定义 =====
struct ConvKernel {
    double kernel[8][8];
    double bias;

    __device__ void randomInit(curandStatePhilox4_32_10_t* rand_state) {
        // He初始化
        double std_dev = sqrt(2.0 / 64.0);
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                kernel[i][j] = curand_normal_double(rand_state) * std_dev;
            }
        }
        bias = 0.0;
    }
};

#endif //SRC_CERN_CUH