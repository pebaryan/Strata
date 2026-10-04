// src/kernels/cuda/glm_decode_trunk.cu - see include/strata/kernels/glm_decode_trunk.hpp.
#include "strata/kernels/glm_decode_trunk.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <mutex>
#include <unordered_map>

namespace strata::kernels::glm {
namespace {

struct Trunk {
    int ne = 0;
    float *cur = nullptr, *mid = nullptr, *nxt = nullptr;       // [HC][ne] each
    float *layer_in = nullptr, *attn_out = nullptr, *ffn_out = nullptr;   // [ne]
    float *flat = nullptr;                                      // [HC*ne] normalised streams (hc_pre scratch)
    float *mixes = nullptr;                                     // [HC_MIX_DIM]
    float *mix[2] = {nullptr, nullptr};                         // per site: pre[4] | post[4] | comb[dst][src] (16)
    cudaStream_t stream = nullptr;
    std::unordered_map<const float*, float*> weights;           // host pointer -> resident device copy
    std::mutex mutex;
    ~Trunk() {
        release();
        for (auto& kv : weights) cudaFree(kv.second);
        if (stream) cudaStreamDestroy(stream);
    }
    void release() {
        float** all[] = {&cur, &mid, &nxt, &layer_in, &attn_out, &ffn_out, &flat, &mixes, &mix[0], &mix[1]};
        for (float** p : all) { if (*p) cudaFree(*p); *p = nullptr; }
        ne = 0;
    }
    bool alloc(int ne_) {
        if (ne == ne_ && cur) return true;
        release();
        const size_t dim = (size_t) HC * ne_;
        auto mk = [&](float*& p, size_t n) { return cudaMalloc(&p, n * sizeof(float)) == cudaSuccess; };
        if (!mk(cur, dim) || !mk(mid, dim) || !mk(nxt, dim) || !mk(layer_in, ne_) || !mk(attn_out, ne_) || !mk(ffn_out, ne_) ||
            !mk(flat, dim) || !mk(mixes, HC_MIX_DIM) || !mk(mix[0], 24) || !mk(mix[1], 24)) { release(); return false; }
        if (!stream && cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) { release(); return false; }
        ne = ne_;
        return true;
    }
    float* resident(const float* host, size_t n) {
        auto it = weights.find(host);
        if (it != weights.end()) return it->second;
        float* d = nullptr;
        if (cudaMalloc(&d, n * sizeof(float)) != cudaSuccess) return nullptr;
        if (cudaMemcpy(d, host, n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) { cudaFree(d); return nullptr; }
        weights.emplace(host, d);
        return d;
    }
};
Trunk& trunk() { static Trunk t; return t; }

__device__ __forceinline__ float sigmoidf(float v) { return 1.0f / (1.0f + expf(-v)); }

__device__ __forceinline__ double block_sum_double(double v, double* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    double total = 0.0;
    if (threadIdx.x == 0) { for (int w = 0; w < (int) (blockDim.x >> 5); ++w) total += red[w]; red[0] = total; }
    __syncthreads();
    total = red[0];
    __syncthreads();
    return total;
}

// flat = x * rms, where rms = 1/sqrt(mean(x^2) + eps) over the whole [HC][ne] block (double sum like the host)
__global__ void hc_rms_flat(const float* x, float* flat, int dim) {
    __shared__ double red[32];
    double acc = 0.0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) acc += (double) x[i] * (double) x[i];
    const double ss = block_sum_double(acc, red);
    const float rms = 1.0f / sqrtf((float) (ss / dim) + HC_RMS_EPS);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) flat[i] = x[i] * rms;
}

// mixes[m] = fn[m] . flat   (one block per mix row)
__global__ void hc_mixes(const float* fn, const float* flat, float* mixes, int dim) {
    __shared__ float red[32];
    const float* row = fn + (size_t) blockIdx.x * dim;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) acc += row[i] * flat[i];
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int w = 0; w < (int) (blockDim.x >> 5); ++w) total += red[w];
        mixes[blockIdx.x] = total;
    }
}

// pre / post / comb from the 24 mixes, with the Sinkhorn normalisation exactly as the host does it (one thread: 16 elements)
__global__ void hc_params(const float* mixes, const float* base, const float* scale, float* mix) {
    if (threadIdx.x != 0) return;
    float pre[HC], post[HC], comb[HC][HC];
    for (int h = 0; h < HC; ++h) {
        pre[h] = sigmoidf(mixes[h] * scale[0] + base[h]) + HC_EPS;
        post[h] = 2.0f * sigmoidf(mixes[HC + h] * scale[1] + base[HC + h]);
    }
    for (int src = 0; src < HC; ++src)
        for (int dst = 0; dst < HC; ++dst) {
            const int k = 2 * HC + dst + src * HC;
            comb[dst][src] = mixes[k] * scale[2] + base[k];
        }
    // softmax over dst per src column, + eps
    for (int src = 0; src < HC; ++src) {
        float m = comb[0][src];
        for (int dst = 1; dst < HC; ++dst) m = fmaxf(m, comb[dst][src]);
        float sum = 0.0f;
        for (int dst = 0; dst < HC; ++dst) { comb[dst][src] = expf(comb[dst][src] - m); sum += comb[dst][src]; }
        for (int dst = 0; dst < HC; ++dst) comb[dst][src] = comb[dst][src] / sum + HC_EPS;
    }
    // norm_cols: divide by the sum over src; norm_rows: divide by the sum over dst
    for (int dst = 0; dst < HC; ++dst) {
        float sum = HC_EPS;
        for (int src = 0; src < HC; ++src) sum += comb[dst][src];
        for (int src = 0; src < HC; ++src) comb[dst][src] /= sum;
    }
    for (int it = 1; it < SINKHORN_ITERS; ++it) {
        for (int src = 0; src < HC; ++src) {
            float sum = HC_EPS;
            for (int dst = 0; dst < HC; ++dst) sum += comb[dst][src];
            for (int dst = 0; dst < HC; ++dst) comb[dst][src] /= sum;
        }
        for (int dst = 0; dst < HC; ++dst) {
            float sum = HC_EPS;
            for (int src = 0; src < HC; ++src) sum += comb[dst][src];
            for (int src = 0; src < HC; ++src) comb[dst][src] /= sum;
        }
    }
    for (int h = 0; h < HC; ++h) { mix[h] = pre[h]; mix[HC + h] = post[h]; }
    for (int dst = 0; dst < HC; ++dst)
        for (int src = 0; src < HC; ++src) mix[2 * HC + dst * HC + src] = comb[dst][src];
}

// layer_in[e] = sum_h pre[h] * x[h][e]
__global__ void hc_layer_in(const float* x, const float* mix, float* layer_in, int ne) {
    const int e = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= ne) return;
    float acc = 0.0f;
    for (int h = 0; h < HC; ++h) acc += mix[h] * x[(size_t) h * ne + e];
    layer_in[e] = acc;
}

// y = rms_norm(y, eps) * w, in place (glm_stage_hc_norm's second half)
__global__ void hc_norm_inplace(float* y, const float* w, int n, float eps) {
    __shared__ double red[32];
    double acc = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += (double) y[i] * (double) y[i];
    const double ss = block_sum_double(acc, red);
    const float rms = 1.0f / sqrtf((float) (ss / n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = y[i] * rms * w[i];
}

// out[dst][e] = post[dst]*site[e] + sum_src comb[dst][src]*res[src][e]   (the host's accumulation order)
__global__ void hc_post_kernel(const float* site, const float* res, const float* mix, float* out, int ne) {
    const int e = (int) (blockIdx.x * blockDim.x + threadIdx.x), dst = (int) blockIdx.y;
    if (e >= ne) return;
    float acc = mix[HC + dst] * site[e];
    for (int src = 0; src < HC; ++src) acc += mix[2 * HC + dst * HC + src] * res[(size_t) src * ne + e];
    out[(size_t) dst * ne + e] = acc;
}

bool fail_with(char* error, size_t cap, const char* what, cudaError_t e) {
    if (error && cap) std::snprintf(error, cap, "%s: %s", what, cudaGetErrorString(e));
    return false;
}

}  // namespace

bool dtrunk_begin(const float* streams_host, int n_embd, char* error, size_t error_capacity) {
    Trunk& t = trunk();
    std::lock_guard<std::mutex> lock(t.mutex);
    if (!t.alloc(n_embd)) return fail_with(error, error_capacity, "decode trunk allocation", cudaErrorMemoryAllocation);
    const cudaError_t e = cudaMemcpyAsync(t.cur, streams_host, (size_t) HC * n_embd * sizeof(float), cudaMemcpyHostToDevice, t.stream);
    return e == cudaSuccess ? true : fail_with(error, error_capacity, "decode trunk upload", e);
}

void* dtrunk_stream() { return (void*) trunk().stream; }

bool dtrunk_hc_pre(int site, const float* fn, const float* base, const float* scale, const float* norm_w, float eps,
                   int n_embd, char* error, size_t error_capacity) {
    Trunk& t = trunk();
    std::lock_guard<std::mutex> lock(t.mutex);
    if (site < 0 || site > 1 || t.ne != n_embd) return fail_with(error, error_capacity, "decode trunk hc_pre arguments", cudaErrorInvalidValue);
    const int dim = HC * n_embd;
    const float* d_fn = t.resident(fn, (size_t) HC_MIX_DIM * dim);
    const float* d_base = t.resident(base, (size_t) HC_MIX_DIM);
    const float* d_scale = t.resident(scale, 3);
    const float* d_norm = norm_w ? t.resident(norm_w, (size_t) n_embd) : nullptr;
    if (!d_fn || !d_base || !d_scale || (norm_w && !d_norm))
        return fail_with(error, error_capacity, "decode trunk weight upload", cudaErrorMemoryAllocation);
    const float* x = site == 0 ? t.cur : t.mid;
    cudaStream_t s = t.stream;
    hc_rms_flat<<<1, 1024, 0, s>>>(x, t.flat, dim);
    hc_mixes<<<HC_MIX_DIM, 256, 0, s>>>(d_fn, t.flat, t.mixes, dim);
    hc_params<<<1, 32, 0, s>>>(t.mixes, d_base, d_scale, t.mix[site]);
    hc_layer_in<<<(n_embd + 255) / 256, 256, 0, s>>>(x, t.mix[site], t.layer_in, n_embd);
    if (d_norm) hc_norm_inplace<<<1, 1024, 0, s>>>(t.layer_in, d_norm, n_embd, eps);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? true : fail_with(error, error_capacity, "decode trunk hc_pre", e);
}

const float* dtrunk_layer_in() { return trunk().layer_in; }
float* dtrunk_attn_out() { return trunk().attn_out; }

bool dtrunk_hc_post(int site, int n_embd, char* error, size_t error_capacity) {
    Trunk& t = trunk();
    std::lock_guard<std::mutex> lock(t.mutex);
    if (site < 0 || site > 1 || t.ne != n_embd) return fail_with(error, error_capacity, "decode trunk hc_post arguments", cudaErrorInvalidValue);
    const dim3 grid((unsigned) ((n_embd + 255) / 256), HC);
    if (site == 0) {
        hc_post_kernel<<<grid, 256, 0, t.stream>>>(t.attn_out, t.cur, t.mix[0], t.mid, n_embd);
    } else {
        hc_post_kernel<<<grid, 256, 0, t.stream>>>(t.ffn_out, t.mid, t.mix[1], t.nxt, n_embd);
        float* tmp = t.cur; t.cur = t.nxt; t.nxt = tmp;       // the next layer's input streams
    }
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? true : fail_with(error, error_capacity, "decode trunk hc_post", e);
}

bool dtrunk_fetch_ffn_in(float* host, int n_embd, char* error, size_t error_capacity) {
    Trunk& t = trunk();
    std::lock_guard<std::mutex> lock(t.mutex);
    cudaError_t e = cudaMemcpyAsync(host, t.layer_in, (size_t) n_embd * sizeof(float), cudaMemcpyDeviceToHost, t.stream);
    if (e == cudaSuccess) e = cudaStreamSynchronize(t.stream);
    return e == cudaSuccess ? true : fail_with(error, error_capacity, "decode trunk fetch ffn_in", e);
}

bool dtrunk_put_ffn_out(const float* host, int n_embd, char* error, size_t error_capacity) {
    Trunk& t = trunk();
    std::lock_guard<std::mutex> lock(t.mutex);
    const cudaError_t e = cudaMemcpyAsync(t.ffn_out, host, (size_t) n_embd * sizeof(float), cudaMemcpyHostToDevice, t.stream);
    return e == cudaSuccess ? true : fail_with(error, error_capacity, "decode trunk put ffn_out", e);
}

int dtrunk_debug_fetch(int which, float* host, int n_embd) {
    Trunk& t = trunk();
    std::lock_guard<std::mutex> lock(t.mutex);
    const float* src = which == 0 ? t.cur : which == 1 ? t.mid : which == 2 ? t.layer_in : which == 3 ? t.attn_out : which == 4 ? t.ffn_out : nullptr;
    if (!src || t.ne != n_embd) return 0;
    const int n = which <= 1 ? HC * n_embd : n_embd;
    cudaStreamSynchronize(t.stream);
    return cudaMemcpy(host, src, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess ? n : 0;
}

bool dtrunk_fetch_streams(float* host, int n_embd, char* error, size_t error_capacity) {
    Trunk& t = trunk();
    std::lock_guard<std::mutex> lock(t.mutex);
    cudaError_t e = cudaMemcpyAsync(host, t.cur, (size_t) HC * n_embd * sizeof(float), cudaMemcpyDeviceToHost, t.stream);
    if (e == cudaSuccess) e = cudaStreamSynchronize(t.stream);
    return e == cudaSuccess ? true : fail_with(error, error_capacity, "decode trunk fetch streams", e);
}

}  // namespace strata::kernels::glm
