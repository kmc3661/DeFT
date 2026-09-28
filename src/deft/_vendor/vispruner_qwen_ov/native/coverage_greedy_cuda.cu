#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

__global__ void global_greedy_kernel(
    const float *rel, const float *glob,
    const float *user_sim, const float *key_sim,
    int n, int keep, uint8_t *available,
    float *user_res, float *key_res, int64_t *out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int first = 0;
    for (int i = 1; i < n; ++i) if (rel[i] > rel[first]) first = i;
    for (int i = 0; i < n; ++i) {
        available[i] = 1;
        float u = 1.0f - user_sim[(size_t)i * n + first];
        float k = 1.0f - key_sim[(size_t)i * n + first];
        user_res[i] = fminf(2.0f, fmaxf(0.0f, u));
        key_res[i] = fminf(2.0f, fmaxf(0.0f, k));
    }
    available[first] = 0; out[0] = first;
    for (int step = 1; step < keep; ++step) {
        float sum_u = 0.0f, sum_k = 0.0f; int count = 0;
        for (int i = 0; i < n; ++i) if (available[i]) {
            float a = rel[i] * user_res[i];
            float b = glob[i] * key_res[i];
            sum_u += a > 0.0f ? a : 0.0f;
            sum_k += b > 0.0f ? b : 0.0f;
            ++count;
        }
        float den_u = count ? sum_u / (float)count : 1.0f;
        float den_k = count ? sum_k / (float)count : 1.0f;
        if (den_u < 1e-8f) den_u = 1e-8f;
        if (den_k < 1e-8f) den_k = 1e-8f;
        int best = -1; float best_value = -1.0f;
        for (int i = 0; i < n; ++i) if (available[i]) {
            float a = rel[i] * user_res[i]; if (a < 0.0f) a = 0.0f;
            float b = glob[i] * key_res[i]; if (b < 0.0f) b = 0.0f;
            float utility = 0.5f * (a / den_u + b / den_k);
            if (best < 0 || utility > best_value) { best = i; best_value = utility; }
        }
        if (best_value <= 0.0f) {
            best = -1; best_value = -1.0f;
            for (int i = 0; i < n; ++i) if (available[i]) {
                float utility = rel[i] + glob[i];
                if (best < 0 || utility > best_value) { best = i; best_value = utility; }
            }
        }
        if (best < 0) return;
        out[step] = best; available[best] = 0;
        for (int i = 0; i < n; ++i) {
            float u = 1.0f - user_sim[(size_t)i * n + best];
            float k = 1.0f - key_sim[(size_t)i * n + best];
            u = fminf(2.0f, fmaxf(0.0f, u));
            k = fminf(2.0f, fmaxf(0.0f, k));
            if (u < user_res[i]) user_res[i] = u;
            if (k < key_res[i]) key_res[i] = k;
        }
    }
}

extern "C" int vispruner_global_greedy_cuda(
    const float *rel, const float *glob,
    const float *user_sim, const float *key_sim,
    int n, int keep, uint8_t *available,
    float *user_res, float *key_res, int64_t *out,
    void *stream_ptr) {
    if (!rel || !glob || !user_sim || !key_sim || !available || !user_res || !key_res || !out || n <= 0 || keep <= 0 || keep > n) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    global_greedy_kernel<<<1, 1, 0, stream>>>(rel, glob, user_sim, key_sim, n, keep, available, user_res, key_res, out);
    return (int)cudaGetLastError();
}
