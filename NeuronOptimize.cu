//
// Created by ASUS on 11/9/2025.
//
#pragma once

#ifdef USE_OPTIMIZED_KERNELS

#define ll long long
#define ull unsigned ll
#include "Neuron.cu"
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
using namespace nvcuda;

__device__ __forceinline__ double warpReduceSum(double val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ ull warpReduceSum(ull val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<typename T>
__device__ T *allocate(T *pool, ull &count, ull pool_size, ull request_size) {
    ull idx = atomicAdd(&count, request_size);
    if (idx >= pool_size) {
        atomicAdd(&count, -request_size);
        return nullptr;
    }
    return retpc<T[request_size]>(&pool[idx - request_size]);
}

template<typename T>
__device__ T *get_shared(T *shared_pool, ull request_id, ull *share_sizes, ull *share_ref_counts, bool wait = true) {
    ull count = 25565;
    while (share_sizes[request_id] == 0 || count == 0 || !wait) {
        count--;
        __nanosleep(5);
    }
    if (!wait && share_sizes[request_id] == 0)
        return nullptr;
    atomicAdd(&share_ref_counts[request_id], 1);
    ull idx = 0;
    for (int i = 0; i < request_id; i++) {
        idx += share_sizes[i];
    }
    return retpc<T[share_sizes[request_id]]>(&shared_pool[idx]);
}

template<typename T>
__device__ T *alloc_shared(
    T *shared_pool,
    ull &pool_count,
    ull pool_size,
    ull request_size,
    ull *share_offsets,
    ull *share_sizes,
    ull *share_ref_counts,
    ull &share_counts,
    ull share_id = 0
) {
    ull memory_idx = atomicAdd(&pool_count, request_size);

    if (memory_idx + request_size > pool_size) {
        atomicAdd(&pool_count, -request_size);
        return nullptr;
    }

    if (share_id == 0) {
        share_id = atomicAdd(&share_counts, 1ULL);
    }

    share_offsets[share_id] = memory_idx;
    share_sizes[share_id] = request_size;
    share_ref_counts[share_id] = 1;

    return retpc<T *>(&shared_pool[memory_idx]);
}

template<typename T>
__device__ T *get_shared(
    T *shared_pool,
    ull share_id,
    ull *share_offsets,
    ull *share_sizes,
    ull *share_ref_counts,
    bool wait = true
) {
    if (wait) {
        int spin_count = 0;
        while (share_sizes[share_id] == 0 && spin_count < 100000) {
            __nanosleep(100);
            spin_count++;
        }
        if (spin_count >= 100000) return nullptr;
    } else {
        if (share_sizes[share_id] == 0) return nullptr;
    }

    atomicAdd(&share_ref_counts[share_id], 1ULL);

    ull offset = share_offsets[share_id];

    return reinterpret_cast<T *>(&shared_pool[offset]);
}

template<typename T>
__device__ T *alloc_shared(T *shared_pool, ull &count, ull pool_size, ull request_size, ull *share_sizes,
                           ull *share_ref_counts, ull &share_counts, ull share_id = 0) {
    ull idx = atomicAdd(&count, request_size) - request_size;
    if (idx + request_size > pool_size) {
        atomicAdd(&count, -request_size);
        return nullptr;
    }
    share_id = (share_id != 0) ? atomicAdd(&share_counts, 1) : share_id;
    share_ref_counts[share_id] = 1;
    share_sizes[share_id] = request_size;
    return retpc<T[request_size]>(&shared_pool[idx]);
}

template<typename T>
__device__ T *release_shared(T *ptr, ull &count, ull request_id, ull *share_sizes, ull *share_ref_counts) {
    atomicAdd(&share_ref_counts[request_id], -1);
    if (share_ref_counts[request_id] <= 0) {
        ull size = share_sizes[request_id];
        atomicAdd(&count, -size);
        for (ull i = 0; i < size; i++) {
            ptr[i].~T();
        }
        memset(ptr, 0, sizeof(T) * size);
        for (ull i = 0; i < size; i++) {
            ptr[i] = T();
        }
        share_sizes[request_id] = 0;
        share_ref_counts[request_id] = 0;
    }
}

template<typename T>
__device__ void free(T *ptr, ull &count, ull size) {
    atomicAdd(&count, -size);
    for (ull i = 0; i < size; i++) {
        ptr[i].~T();
    }
    memset(ptr, 0, sizeof(T) * size);
    for (ull i = 0; i < size; i++) {
        ptr[i] = T();
    }
}

struct KVCache {
    half *k_cache; // [512][256]
    half *v_cache; // [512][256]
    int seq_len;
    int max_len;
};

struct SharedContext {
    double sum_weights;
    double deviation_norm;
    double stm_utility;
    double mean, std;
    bool trigger_gemm;
    int port_sync_counter;
    double usum, isum, vsum;
    ull history_index;
    double prev_diff_norm;
};

__device__ __forceinline__ double blockReduceSum(double val, double *shared) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    val = warpReduceSum(val);
    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    val = (threadIdx.x < 32) ? shared[threadIdx.x] : 0.0;
    if (warp_id == 0) val = warpReduceSum(val);
    return val;
}

__device__ __forceinline__ void selectiveSSM(
    Neuron &n,
    const double *transformed_input, // [256*256]
    ull port,
    int tid
) {
    __shared__ double s_B[256];
    __shared__ double s_C[256];
    __shared__ double s_input_mean[256];

    if (tid < 256) {
        int i = tid;
        double row_sum = 0.0;

        for (int j = 0; j < 256; j++) {
            row_sum += transformed_input[i * 256 + j];
        }

        s_input_mean[i] = row_sum / 256.0;

        s_B[i] = Neuron::gelu(s_input_mean[i]);
        s_C[i] = Neuron::gelu(-s_input_mean[i]);
    }
    __syncthreads();

    const int ELEMS = 64;
    for (int local_idx = 0; local_idx < ELEMS; local_idx++) {
        int global_idx = local_idx * 1024 + tid;
        int i = global_idx / 256;
        int j = global_idx % 256;

        if (global_idx < 256 * 256) {
            if (j == 0 && tid < 256) {
                double delta = s_B[i] * __half2float(n.PS_aggregate[i][0]);
                n.h_state[i] = 0.9 * n.h_state[i] + delta;
            }

            double c_contrib = s_C[i] * n.h_state[i];
            atomicAdd(&n.P_Matrix[i][j], c_contrib);
        }
    }
}

__device__ __forceinline__ void computeWKV(
    Neuron &n,
    double *aggregated_input, // [256*256]
    double score,
    int tid
) {
    const int ELEMS = 64;

    for (int vec_idx = 0; vec_idx < 256; vec_idx++) {
        if ((vec_idx % 32) != (tid / 32)) continue;

        double wkv = 0.0;
        double state = n.h_state[vec_idx];
        double w_decay = -exp(n.time_decay[vec_idx]);
        double time_first = __half2float(n.time_first[vec_idx]);

        for (int j = 0; j < 256; j++) {
            int idx = vec_idx * 256 + j;

            double k_val = __half2float(n.PS_aggregate[vec_idx][j]);
            double v_val = __half2float(n.PS_aggregate[vec_idx][j]);

            wkv += exp(time_first + k_val) * v_val;
            state = state * exp(w_decay) + exp(k_val) * v_val;

            double trans_val = aggregated_input[idx];
            double update = trans_val * w_decay * trans_val * score +
                            wkv / (wkv + state + 1e-9);

            double old_val = __half2float(n.PS_aggregate[vec_idx][j]);
            n.PS_aggregate[vec_idx][j] = __float2half(old_val + update);
        }
    }
}

__device__ __forceinline__ void extractConvFeatures(
    Neuron &n,
    const double *input, // [256*256]
    ull port,
    int tid
) {
    if (tid < 8) {
        int kernel_idx = tid;
        for (int out_i = 0; out_i < 32; out_i++) {
            for (int out_j = 0; out_j < 32; out_j++) {
                double sum = 0.0;
                for (int ki = 0; ki < 8; ki++) {
                    for (int kj = 0; kj < 8; kj++) {
                        int in_i = out_i * 8 + ki;
                        int in_j = out_j * 8 + kj;

                        if (in_i < 256 && in_j < 256) {
                            double input_val = input[in_i * 256 + in_j];
                            double kernel_val = n.input_conv_kernels[port][kernel_idx].kernel[ki][kj];
                            sum += input_val * kernel_val;
                        }
                    }
                }
                sum += n.input_conv_kernels[port][kernel_idx].bias; // bias
                n.conv_feature_maps[port][kernel_idx][out_i][out_j] = Neuron::gelu(sum);
            }
        }
    }
    __syncthreads();
    __shared__ double s_deconv_output[256][256];
    const int ELEMS = 64;
    for (int local_idx = 0; local_idx < ELEMS; local_idx++) {
        int idx = local_idx * 1024 + tid;
        int i = idx / 256;
        int j = idx % 256;
        if (idx < 256 * 256) {
            s_deconv_output[i][j] = 0.0;
        }
    }
    __syncthreads();
    for (int k = 0; k < 8; k++) {
        for (int fi = 0; fi < 32; fi++) {
            for (int fj = 0; fj < 32; fj++) {
                double feature_val = n.conv_feature_maps[port][k][fi][fj];
                for (int ki = 0; ki < 8; ki++) {
                    for (int kj = 0; kj < 8; kj++) {
                        int out_i = fi * 8 + ki;
                        int out_j = fj * 8 + kj;

                        if (out_i < 256 && out_j < 256) {
                            double kernel_val = n.input_conv_kernels[port][k].kernel[ki][kj];
                            atomicAdd(&s_deconv_output[out_i][out_j],
                                      feature_val * kernel_val);
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
    for (int local_idx = 0; local_idx < ELEMS; local_idx++) {
        int idx = local_idx * 1024 + tid;
        int i = idx / 256;
        int j = idx % 256;
        if (idx < 256 * 256) {
            const_cast<double *>(input)[idx] = s_deconv_output[i][j] / 8.0;
        }
    }
}

__device__ __forceinline__ void computeKFEAttention(
    Neuron &n,
    SharedContext &ctx,
    int tid
) {
    __shared__ double s_dot_products[16];
    __shared__ double s_warp_reduce[32];

    if (tid == 0) {
        for (int i = 0; i < 16; i++) s_dot_products[i] = 0.0;
        ctx.stm_utility = 0.0;
        ctx.usum = 0.0;
        ctx.isum = 0.0;
        ctx.vsum = 0.0;
    }
    __syncthreads();
    int kfe_idx = (tid / 64); // 0-15
    int sub_tid = tid % 64;

    if (kfe_idx < 16) {
        int k = kfe_idx;

        if (n.kfe_local[k].Icore >= 0.01) {
            // Vmem · Deviation
            double local_dot = 0.0;
            for (int idx = sub_tid; idx < 256 * 256; idx += 64) {
                int i = idx / 256;
                int j = idx % 256;
                local_dot += n.kfe_local[k].Vmem[i][j] * n.Deviation[i][j];
            }
            for (int offset = 32; offset > 0; offset /= 2) {
                local_dot += __shfl_down_sync(0xffffffff, local_dot, offset);
            }
            if (sub_tid == 0) {
                s_dot_products[k] = local_dot;
            }
        }
    }
    __syncthreads();
    if (tid < 16) {
        int k = tid;

        if (n.kfe_local[k].Icore >= 0.01) {
            double dot_prod = s_dot_products[k];
            double attn_weight = 1.0 / (1.0 + exp(-dot_prod));
            double weighted_attn = attn_weight * n.kfe_local[k].Icore;

            atomicAdd(&ctx.stm_utility, weighted_attn);
            atomicAdd(&ctx.usum, n.kfe_local[k].Ulocal);
            atomicAdd(&ctx.isum, n.kfe_local[k].Icore);
            atomicAdd(&ctx.vsum, n.kfe_local[k].V);
            n.kfe_mutex.lock();
            n.kfe_local[k].Ulocal += 0.01 * (attn_weight - n.kfe_local[k].Ulocal);
            n.kfe_local[k].Icore += 0.01 * (0.5 - n.kfe_local[k].Icore);
            n.kfe_local[k].V -= 0.01 * (1.0 - n.kfe_local[k].V);
            n.kfe_local[k].Rcycles = n.cycle_counter;
            n.kfe_mutex.unlock();
        }
    }
    __syncthreads();
    const int ELEMS = 64;
    for (int k = 0; k < 16; k++) {
        if (n.kfe_local[k].Icore >= 0.01) {
            double attn_weight = 1.0 / (1.0 + exp(-s_dot_products[k]));
            double weighted_attn = attn_weight * n.kfe_local[k].Icore;

            for (int local_idx = 0; local_idx < ELEMS; local_idx++) {
                int idx = local_idx * 1024 + tid;
                int i = idx / 256;
                int j = idx % 256;

                if (idx < 256 * 256) {
                    double contrib = weighted_attn * n.kfe_local[k].Vmem[i][j];
                    atomicAdd(&n.M_KFE[i][j], contrib);
                }
            }
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void queryExternalKFE(
    Neuron &n,
    SharedContext &ctx,
    int tid
) {
    if (tid != 0) return;
    bool need_external = (ctx.usum < 4.0 && ctx.isum < 6.0) || ctx.vsum > 12.0;
    if (!need_external) return;
    double curr_conv16[16][16];
    ConvResidualProcessor::conv2d_16x16(n.W_predict, ConvKernel16(), curr_conv16);
    double max_sim = -1.0;
    ull max_idx = 0;

    for (ull k = 0; k < n.ext_kfe_slots.size(); k++) {
        auto &slot = n.ext_kfe_slots[k];
        double sim = cosineSimilarity16x16(slot.conv16, curr_conv16);
        if (sim > max_sim) {
            max_sim = sim;
            max_idx = k;
        }
    }
    KFE_STM_Slot ext_kfe = {};
    bool success = false;

    if (n.kfe_query_queue && n.kfe_result_queue) {
        if (n.kfe_query_queue->push(n.ext_kfe_slots[max_idx].hash.data())) {
            for (int wait = 0; wait < 1000; wait++) {
                if (n.kfe_result_queue->size() > 0) {
                    if (n.kfe_result_queue->pop(ext_kfe)) {
                        success = true;
                        break;
                    }
                }
                __nanosleep(100);
            }
        }
    }
    if (!success || ext_kfe.Icore < 0.01) return;
    double dot_product = 0.0;
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            dot_product += ext_kfe.Vmem[i][j] * n.Deviation[i][j];
        }
    }

    double attn_weight = 1.0 / (1.0 + exp(-dot_product));
    double weighted_attn = attn_weight * ext_kfe.Icore;

    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            atomicAdd(&n.M_KFE[i][j], weighted_attn * ext_kfe.Vmem[i][j]);
        }
    }

    ctx.stm_utility += weighted_attn;
}

__device__ __forceinline__ void KVCacheAttention(
    KVCache &kvc,
    Neuron &n,
    int tid
) {
    __shared__ float s_scores[512];
    __shared__ float s_max_score;
    __shared__ float s_sum_exp;
    __shared__ double s_output[256];

    if (tid == 0) {
        s_max_score = -1e9f;
        s_sum_exp = 0.0f;
    }
    __syncthreads();

    // Query, Key, Value
    __shared__ double s_query[256];
    __shared__ double s_new_key[256];
    __shared__ double s_new_value[256];

    if (tid < 256) {
        int i = tid;
        s_query[i] = n.P_Matrix[0][i];
        double key_sum = 0.0;
        for (int j = 0; j < 256; j++) {
            key_sum += __half2float(n.PS_aggregate[i][j]);
        }
        s_new_key[i] = key_sum / 256.0;
        double val_sum = 0.0;
        for (int j = 0; j < 256; j++) {
            val_sum += n.P_Matrix[i][j];
        }
        s_new_value[i] = val_sum / 256.0;

        s_output[i] = 0.0;
    }
    __syncthreads();
    if (tid == 0) {
        if (kvc.seq_len < kvc.max_len) {
            for (int i = 0; i < 256; i++) {
                kvc.k_cache[kvc.seq_len * 256 + i] = __float2half_rn(s_new_key[i]);
                kvc.v_cache[kvc.seq_len * 256 + i] = __float2half_rn(s_new_value[i]);
            }
            kvc.seq_len++;
        } else {
            for (int seq = 0; seq < kvc.max_len - 1; seq++) {
                for (int i = 0; i < 256; i++) {
                    kvc.k_cache[seq * 256 + i] = kvc.k_cache[(seq + 1) * 256 + i];
                    kvc.v_cache[seq * 256 + i] = kvc.v_cache[(seq + 1) * 256 + i];
                }
            }
            for (int i = 0; i < 256; i++) {
                kvc.k_cache[(kvc.max_len - 1) * 256 + i] = __float2half_rn(s_new_key[i]);
                kvc.v_cache[(kvc.max_len - 1) * 256 + i] = __float2half_rn(s_new_value[i]);
            }
        }
    }
    __syncthreads();

    int seq_len = kvc.seq_len;
    for (int seq = tid; seq < seq_len; seq += 1024) {
        float score = 0.0f;

        for (int i = 0; i < 256; i++) {
            float q = (float) s_query[i];
            float k = __half2float(kvc.k_cache[seq * 256 + i]);
            score += q * k;
        }

        score /= sqrtf(256.0f);
        s_scores[seq] = score;
    }
    __syncthreads();
    if (tid == 0) {
        for (int seq = 0; seq < seq_len; seq++) {
            s_max_score = fmaxf(s_max_score, s_scores[seq]);
        }
    }
    __syncthreads();
    __shared__ float s_warp_sums[32];
    float local_sum = 0.0f;

    for (int seq = tid; seq < seq_len; seq += 1024) {
        float exp_score = expf(s_scores[seq] - s_max_score);
        s_scores[seq] = exp_score;
        local_sum += exp_score;
    }

    local_sum = warpReduceSum((double) local_sum);
    if (tid % 32 == 0) {
        s_warp_sums[tid / 32] = local_sum;
    }
    __syncthreads();

    if (tid == 0) {
        s_sum_exp = 0.0f;
        for (int w = 0; w < 32; w++) {
            s_sum_exp += s_warp_sums[w];
        }
    }
    __syncthreads();
    for (int seq = tid; seq < seq_len; seq += 1024) {
        s_scores[seq] /= s_sum_exp;
    }
    __syncthreads();

    if (tid < 256) {
        int i = tid;
        double out_val = 0.0;

        for (int seq = 0; seq < seq_len; seq++) {
            float score = s_scores[seq];
            float v = __half2float(kvc.v_cache[seq * 256 + i]);
            out_val += score * v;
        }

        s_output[i] = out_val;
    }
    __syncthreads();

    if (tid < 256) {
        for (int j = 0; j < 256; j++) {
            atomicAdd(&n.M_KFE[tid][j], s_output[tid] * 0.1);
        }
    }
}

__device__ __forceinline__ void drcIterativeCorrection(
    Neuron &n,
    double *r_P_next, // [ELEMS_PER_THREAD]
    SharedContext &ctx,
    double *s_reduce,
    int tid
) {
    const int ELEMS = 64;
    double alpha = 0.7;
    double epsilon = 1e-4;
    double eta_base = 0.1;
    double lambda = 0.9;

    if (tid == 0) {
        ctx.prev_diff_norm = 0.0;
        ctx.history_index = 0;
    }
    __syncthreads();

    double r_T_fixed[ELEMS];
    for (int i = 0; i < ELEMS; i++) {
        int idx = i * 1024 + tid;
        int row = idx / 256;
        int col = idx % 256;

        if (idx < 256 * 256) {
            double ps_agg = __half2float(n.PS_aggregate[row][col]);
            r_T_fixed[i] = alpha * ps_agg + (1.0 - alpha) * r_P_next[i];
        }
    }

    double r_P_current[ELEMS];
    for (int i = 0; i < ELEMS; i++) {
        r_P_current[i] = r_P_next[i];
    }

    for (int iter = 0; iter < 16; iter++) {
        double local_diff_sq = 0.0;

        for (int i = 0; i < ELEMS; i++) {
            int idx = i * 1024 + tid;
            int row = idx / 256;
            int col = idx % 256;

            if (idx < 256 * 256) {
                // V_corr
                double V_corr = (r_T_fixed[i] - r_P_current[i]) * eta_base;

                // M_attn (计算局部特征)
                double local_feat = 0.0;
                for (int p = 0; p < 4; p++) {
                    if (!n.port_in[p].empty()) {
                        NeuronInput temp = n.port_in[p].front();
                        local_feat += temp.array[row][col];
                    }
                }
                local_feat /= 4.0;

                double attn_w = 1.0 / (1.0 + exp(-(local_feat * r_P_current[i])));
                double M_attn = attn_w * V_corr;

                double V_hist = 0.0;
                if (iter > 0) {
                    for (int h = 1; h <= min(iter, 3); h++) {
                        ull hist_idx = (ctx.history_index - h + 5) % 5;
                        ull prev_idx = (hist_idx - 1 + 5) % 5;
                        double delta = __half2float(n.P_history[hist_idx][row][col]) -
                                       __half2float(n.P_history[prev_idx][row][col]);
                        V_hist += pow(lambda, (double) h) * delta;
                    }
                }

                double P_new = r_P_current[i] + V_corr + M_attn + V_hist;
                double diff = P_new - r_P_current[i];
                local_diff_sq += diff * diff;

                r_P_current[i] = P_new;
            }
        }

        local_diff_sq = blockReduceSum(local_diff_sq, s_reduce);

        __shared__ double s_diff_norm;
        if (tid == 0) {
            s_diff_norm = sqrt(local_diff_sq / (256.0 * 256.0));
        }
        __syncthreads();

        // 更新历史
        if (tid == 0) {
            ctx.history_index = (ctx.history_index + 1) % 5;
        }
        __syncthreads();

        for (int i = 0; i < ELEMS; i++) {
            int idx = i * 1024 + tid;
            int row = idx / 256;
            int col = idx % 256;

            if (idx < 256 * 256) {
                n.P_history[ctx.history_index][row][col] = __float2half(r_P_current[i]);
            }
        }

        if (s_diff_norm < epsilon) break;
        if (iter > 8 && s_diff_norm > ctx.prev_diff_norm) break;

        if (tid == 0) {
            ctx.prev_diff_norm = s_diff_norm;
        }
        __syncthreads();
    }

    for (int i = 0; i < ELEMS; i++) {
        r_P_next[i] = r_P_current[i];
    }
}

__device__ __forceinline__ void ddpmDenoising(
    Neuron &n,
    double *r_P_matrix, // [ELEMS_PER_THREAD]
    int tid
) {
    const int ELEMS = 64;
    __shared__ double s_beta_schedule[16];

    // 计算beta schedule (cosine)
    if (tid < 16) {
        double alpha_t = cos(PI * tid / 32.0);
        s_beta_schedule[tid] = 1.0 - alpha_t * alpha_t;
    }
    __syncthreads();

    // 保存副本用于去噪
    double r_P_noisy[ELEMS];
    for (int i = 0; i < ELEMS; i++) {
        r_P_noisy[i] = r_P_matrix[i];
    }

    // 反向去噪过程
    for (int t = 15; t >= 0; t--) {
        double beta = s_beta_schedule[t];

        // 预测噪声 (使用神经网络)
        __shared__ double s_noise_pred[256][256];

        // 重建矩阵到shared memory
        for (int i = 0; i < ELEMS; i++) {
            int idx = i * 1024 + tid;
            int row = idx / 256;
            int col = idx % 256;

            if (idx < 256 * 256) {
                s_noise_pred[row][col] = r_P_noisy[i];
            }
        }
        __syncthreads();

        // 调用噪声预测网络 (只一个线程)
        if (tid == 0) {
            double noise_output[256][256];
            n.predictNoise((double(*)[256]) s_noise_pred, noise_output);

            // 拷贝回shared
            for (int row = 0; row < 256; row++) {
                for (int col = 0; col < 256; col++) {
                    s_noise_pred[row][col] = noise_output[row][col];
                }
            }
        }
        __syncthreads();

        // DDPM去噪步骤
        for (int i = 0; i < ELEMS; i++) {
            int idx = i * 1024 + tid;
            int row = idx / 256;
            int col = idx % 256;

            if (idx < 256 * 256) {
                double noise_pred = s_noise_pred[row][col];
                r_P_noisy[i] = (r_P_noisy[i] - sqrt(beta) * noise_pred) / sqrt(1.0 - beta);
            }
        }
        __syncthreads();
    }

    // 融合去噪结果
    const double alpha_c = 0.7;
    for (int i = 0; i < ELEMS; i++) {
        r_P_matrix[i] = (alpha_c * r_P_matrix[i] + (1.0 - alpha_c) * r_P_noisy[i]) / 2.0;
    }
}

__device__ __forceinline__ void tensorCoreGEMM(
    Neuron &n,
    double *r_output, // [ELEMS_PER_THREAD]
    double *temp_buffer,
    int tid
) {
    const int ELEMS = 64;

#if __CUDA_ARCH__ >= 700
    half *P_half = (half *) temp_buffer;
    half *W_half = P_half + 256 * 256;
    float *C_float = (float *) (W_half + 256 * 256);

    // FP64 -> FP16
    for (int i = 0; i < ELEMS; i++) {
        int idx = i * 1024 + tid;
        if (idx < 256 * 256) {
            int row = idx / 256;
            int col = idx % 256;
            P_half[idx] = __double2half_rn(n.P_Matrix[row][col]);
            W_half[idx] = __double2half_rn(n.W_predict[row][col]);
        }
    }
    __syncthreads();

    int warp_id = tid / 32;
    int tiles_per_warp = (16 * 16 + 31) / 32; // 每个warp 8个tiles

    for (int tile_iter = 0; tile_iter < tiles_per_warp; tile_iter++) {
        int tile_id = warp_id * tiles_per_warp + tile_iter;
        if (tile_id >= 16 * 16) break;

        int tile_m = (tile_id / 16) * 16;
        int tile_n = (tile_id % 16) * 16;

        // WMMA fragments
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

        wmma::fill_fragment(c_frag, 0.0f);

#pragma unroll
        for (int k_tile = 0; k_tile < 16; k_tile++) {
            int k = k_tile * 16;

            wmma::load_matrix_sync(a_frag, P_half + tile_m * 256 + k, 256);
            wmma::load_matrix_sync(b_frag, W_half + k * 256 + tile_n, 256);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        wmma::store_matrix_sync(C_float + tile_m * 256 + tile_n, c_frag, 256,
                                wmma::mem_row_major);
    }
    __syncthreads();

    for (int i = 0; i < ELEMS; i++) {
        int idx = i * 1024 + tid;
        int row = idx / 256;
        int col = idx % 256;

        if (idx < 256 * 256) {
            float gemm_out = C_float[idx];
            double x = (double) gemm_out + n.M_KFE[row][col];
            r_output[i] = Neuron::gelu(x);
        }
    }

#else

    for (int i = 0; i < ELEMS; i++) {
        int idx = i * 1024 + tid;
        int row = idx / 256;
        int col = idx % 256;

        if (idx < 256 * 256) {
            double sum = 0.0;

#pragma unroll 16
            for (int k = 0; k < 256; k++) {
                sum += n.P_Matrix[row][k] * n.W_predict[k][col];
            }

            r_output[i] = Neuron::gelu(sum + n.M_KFE[row][col]);
        }
    }

#endif
}

__global__ void step_complete_fusion(
    Neuron *neurons,
    ull neuron_count,
    double score,
    KVCache *kv_caches,
    double *global_pool,
    ull pool_size,
    bool *active_flags
) {
    ull nid = blockIdx.x / 4;
    ull port = blockIdx.x % 4;
    if (nid >= neuron_count) return;

    ull tid = threadIdx.x;
    ull wid = tid / 32;
    ull lane = tid % 32;

    if (active_flags[nid] == false) return;

    Neuron &n = neurons[nid];
    KVCache &kvc = kv_caches[nid];

    // ========================================
    // Shared Memory
    // ========================================
    __shared__ double s_reduce[32];
    __shared__ SharedContext ctx;

    if (tid == 0 && port == 0) {
        ctx.sum_weights = 0.0;
        ctx.deviation_norm = 0.0;
        ctx.stm_utility = 0.0;
        ctx.trigger_gemm = false;
        ctx.port_sync_counter = 0;
    }
    __syncthreads();

    // ========================================
    // 寄存器数组
    // ========================================
    const int ELEMS = 64;
    double r_data[ELEMS];

    // ========================================
    // PHASE 1: 输入加载 + 变换
    // ========================================

    bool has_input = !n.port_in[port].empty();
    double weight = 0.0;

    if (has_input) {
        NeuronInput inp = n.port_in[port].front();
        weight = inp.weight * inp.activity;

        for (int i = 0; i < ELEMS; i++) {
            int idx = i * 1024 + tid;
            int row = idx / 256;
            int col = idx % 256;

            if (idx < 256 * 256) {
                double input_val = inp.array[row][col];
                double mult_val = n.input_multiplex_array[port][row][col];
                r_data[i] = input_val * mult_val * weight;
            } else {
                r_data[i] = 0.0;
            }
        }
    } else {
        for (int i = 0; i < ELEMS; i++) {
            r_data[i] = 0.0;
        }
    }

    if (has_input) {
        double *input_matrix = global_pool + nid * 256 * 256 * 4 + port * 256 * 256;

        for (int i = 0; i < ELEMS; i++) {
            int idx = i * 1024 + tid;
            if (idx < 256 * 256) {
                input_matrix[idx] = r_data[i];
            }
        }
        __syncthreads();

        extractConvFeatures(n, input_matrix, port, tid);
        __syncthreads();

        for (int i = 0; i < ELEMS; i++) {
            int idx = i * 1024 + tid;
            if (idx < 256 * 256) {
                r_data[i] = input_matrix[idx];
            }
        }
    }
    double *agg_buffer = global_pool + nid * 256 * 256;
    double *weight_buffer = global_pool + neuron_count * 256 * 256 + nid;

    if (has_input) {
        atomicAdd(weight_buffer, weight);

        for (int i = 0; i < ELEMS; i++) {
            int idx = i * 1024 + tid;
            if (idx < 256 * 256) {
                atomicAdd(&agg_buffer[idx], r_data[i]);
            }
        }
    }

    __threadfence();

    if (tid == 0) {
        atomicAdd(&ctx.port_sync_counter, 1);
    }

    while (ctx.port_sync_counter < 4) {
        __nanosleep(10);
    }

    if (port != 0) return;

    double total_weight = *weight_buffer;

    for (int i = 0; i < ELEMS; i++) {
        int idx = i * 1024 + tid;
        if (idx < 256 * 256) {
            if (total_weight > 1e-6) {
                agg_buffer[idx] /= total_weight;
            }
        }
    }
    __syncthreads();

    selectiveSSM(n, agg_buffer, 0, tid);
    __syncthreads();

    __shared__ double s_score;
    if (tid == 0) {
        s_score = 0.0;
    }
    __syncthreads();

    double local_score = 0.0;
    for (int i = 0; i < ELEMS; i++) {
        int idx = i * 1024 + tid;
        int row = idx / 256;
        int col = idx % 256;

        if (idx < 256 * 256) {
            local_score += n.P_Matrix[row][col] * agg_buffer[idx];
        }
    }

    local_score = blockReduceSum(local_score, s_reduce);
    if (tid == 0) {
        s_score = local_score / (256.0 * 256.0);
    }
    __syncthreads();

    computeWKV(n, agg_buffer, s_score, tid);
    __syncthreads();

    for (int i = 0; i < ELEMS; i++) {
        int idx = i * 1024 + tid;
        int row = idx / 256;
        int col = idx % 256;

        if (idx < 256 * 256) {
            r_data[i] = __half2float(n.PS_aggregate[row][col]);
        }
    }

    double dev_sq_sum = 0.0;

    for (int i = 0; i < ELEMS; i++) {
        int idx = i * 1024 + tid;
        int row = idx / 256;
        int col = idx % 256;

        if (idx < 256 * 256) {
            double dev = r_data[i] - n.P_stable[row][col];
            n.Deviation[row][col] = dev;
            dev_sq_sum += dev * dev;
        }
    }

    dev_sq_sum = blockReduceSum(dev_sq_sum, s_reduce);
    if (tid == 0) {
        ctx.deviation_norm = sqrt(dev_sq_sum / (256.0 * 256.0));
    }
    __syncthreads();

    if (tid == 0) {
        memset(n.M_KFE, 0, sizeof(n.M_KFE));
    }
    __syncthreads();

    computeKFEAttention(n, ctx, tid);
    __syncthreads();

    queryExternalKFE(n, ctx, tid);
    __syncthreads();

    if (tid == 0) {
        n.STM_aggregate_utility = ctx.stm_utility;
    }

    // ========================================
    // PHASE 9: KV-Cache Attention
    // ========================================

    KVCacheAttention(kvc, n, tid);
    __syncthreads();

    if (tid == 0) {
        ctx.trigger_gemm = (n.cycle_counter % 16 == 0) ||
                           (ctx.deviation_norm > 0.5) ||
                           (n.core_vulnerability > 0.7) ||
                           (ctx.stm_utility > 0.6);
    }
    __syncthreads();

    if (!ctx.trigger_gemm) {
        // Micro Correction
        double alpha = 0.3;
        double eta_micro = 0.05 + max(min(n.getLearningRate(3), 1.0), 0.0) * 0.0001;

        for (int i = 0; i < ELEMS; i++) {
            int idx = i * 1024 + tid;
            int row = idx / 256;
            int col = idx % 256;

            if (idx < 256 * 256) {
                double ps_agg = r_data[i];
                double p_curr = n.P_Matrix[row][col];
                double target = alpha * ps_agg + (1.0 - alpha) * p_curr;

                p_curr += eta_micro * (target - p_curr);
                p_curr += max(min(n.noise, 1.0), 0.0) * 0.0001 *
                        (Neuron::randomInRange(0, 1) - 0.5);

                n.P_Matrix[row][col] = p_curr;
                r_data[i] = p_curr;
            }
        }
    } else {
        // GEMM + DRC + DDPM + LayerNorm
        ll pos = n.local_coord[0] * n.GRID_SIZE * n.GRID_SIZE +
                 n.local_coord[1] * n.GRID_SIZE +
                 n.local_coord[2];

        for (int i = 0; i < ELEMS; i++) {
            int idx = i * 1024 + tid;
            int row = idx / 256;
            int col = idx % 256;

            if (idx < 256 * 256) {
                int d = row * 256 + col;
                double freq = 1.0 / pow(10000.0, 2.0 * d / 65536.0);
                double pe = (d % 2 == 0) ? sin(pos * freq) : cos(pos * freq);
                n.P_Matrix[row][col] += 0.1 * pe;
            }
        }
        __syncthreads();

        // Dropout
        if (n.training) {
            for (int i = 0; i < ELEMS; i++) {
                int idx = i * 1024 + tid;
                int row = idx / 256;
                int col = idx % 256;

                if (idx < 256 * 256) {
                    if (curand_uniform(&n.rand_state) < 0.05) {
                        n.W_predict[row][col] = 0.0;
                    }
                }
            }
            __syncthreads();
        }

        // Tensor Core GEMM
        double *temp_buf = global_pool + nid * 256 * 256 * 8;
        tensorCoreGEMM(n, r_data, temp_buf, tid);
        __syncthreads();

        // DRC
        drcIterativeCorrection(n, r_data, ctx, s_reduce, tid);
        __syncthreads();

        // DDPM Denoising
        ddpmDenoising(n, r_data, tid);
        __syncthreads();

        // Layer Normalization
        double sum_val = 0.0;
        for (int i = 0; i < ELEMS; i++) {
            sum_val += r_data[i];
        }

        sum_val = blockReduceSum(sum_val, s_reduce);
        if (tid == 0) {
            ctx.mean = sum_val / (256.0 * 256.0);
        }
        __syncthreads();

        double var_val = 0.0;
        for (int i = 0; i < ELEMS; i++) {
            double diff = r_data[i] - ctx.mean;
            var_val += diff * diff;
        }

        var_val = blockReduceSum(var_val, s_reduce);
        if (tid == 0) {
            ctx.std = sqrt(var_val / (256.0 * 256.0) + 1e-6);
        }
        __syncthreads();

        for (int i = 0; i < ELEMS; i++) {
            r_data[i] = (r_data[i] - ctx.mean) / ctx.std;
        }
    }

    for (int i = 0; i < ELEMS; i++) {
        int idx = i * 1024 + tid;
        int row = idx / 256;
        int col = idx % 256;

        if (idx < 256 * 256) {
            n.P_Matrix[row][col] = r_data[i];
            n.P_stable[row][col] = r_data[i];
        }
    }
    __syncthreads();

    double vuln_sq = 0.0;
    for (int i = 0; i < ELEMS; i++) {
        int idx = i * 1024 + tid;
        int row = idx / 256;
        int col = idx % 256;

        if (idx < 256 * 256) {
            double diff = n.P_Matrix[row][col] - n.P_stable[row][col];
            vuln_sq += diff * diff;
        }
    }

    vuln_sq = blockReduceSum(vuln_sq, s_reduce);
    if (tid == 0) {
        double vuln = sqrt(vuln_sq / (256.0 * 256.0));
        n.core_vulnerability = tanh(vuln);
    }
    __syncthreads();

    if (tid == 0) {
        n.broadcastOutput();
        n.cycle_counter++;
    }

    if (tid < 4) {
        n.updateConvKernels(tid);
    }
}

#endif
