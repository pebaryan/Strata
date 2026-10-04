// src/kernels/cuda/glm_indexer_device.cu - see include/strata/kernels/glm_indexer_device.hpp.
#include "strata/kernels/glm_indexer_device.hpp"

#include <cuda_runtime.h>
#include <cub/device/device_segmented_radix_sort.cuh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <unordered_map>

#include "strata/kernels/native_mmvq.hpp"

namespace strata::kernels::glm {
namespace {

VramReclaimFn g_reclaim = nullptr;

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

// row[pos0 + t] = [ LayerNorm(kraw[t]) * w + b | graw[t] ]   (blockDim = 128: one thread per channel)
__global__ void idx_row_kernel(const float* kraw, const float* graw, const float* nw, const float* nb, float* rows, int pos0) {
    __shared__ double red[32];
    const int t = (int) blockIdx.x, i = (int) threadIdx.x;
    const float key = kraw[(size_t) t * IDX_D + i];
    const double mean = block_sum_double((double) key, red) / IDX_D;
    const double c = (double) key - mean;
    const double sq = block_sum_double(c * c, red);
    const float inv = 1.0f / sqrtf((float) (sq / IDX_D) + 1e-5f);
    float* row = rows + (size_t) (pos0 + t) * (2 * IDX_D);
    row[i] = (float) ((double) key - mean) * inv * nw[i] + nb[i];
    row[IDX_D + i] = graw[(size_t) t * IDX_D + i];
}

// pooled[b][ch] = sum_m softmax_m(gate[m][ch] + ape[m][ch]) * key[m][ch]   (one block per pool, 128 channels)
__global__ void idx_pool_kernel(const float* rows, const float* ape, float* pooled, int b0) {
    const int b = b0 + (int) blockIdx.x, ch = (int) threadIdx.x;
    float lg[IDX_KPOOL];
    float mx = -INFINITY;
    for (int m = 0; m < IDX_KPOOL; ++m) {
        lg[m] = rows[(size_t) (b * IDX_KPOOL + m) * (2 * IDX_D) + IDX_D + ch] + ape[m * IDX_D + ch];
        mx = fmaxf(mx, lg[m]);
    }
    double sum = 0.0;
    float num = 0.0f;
    for (int m = 0; m < IDX_KPOOL; ++m) {
        const float e = expf(lg[m] - mx);
        sum += e;
        num += e * rows[(size_t) (b * IDX_KPOOL + m) * (2 * IDX_D) + ch];
    }
    pooled[(size_t) b * IDX_D + ch] = (float) ((double) num / sum);
}

// wts[t][h] = (proj[h] . x[t]) / sqrt(d * nh)      grid (nh, T)
__global__ void idx_wts_kernel(const float* proj, const float* x, float* wts, int ne, float scale) {
    __shared__ float red[32];
    const int h = (int) blockIdx.x, t = (int) blockIdx.y;
    const float* row = proj + (size_t) h * ne;
    const float* xt = x + (size_t) t * ne;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < ne; i += blockDim.x) acc += row[i] * xt[i];
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int w = 0; w < (int) (blockDim.x >> 5); ++w) total += red[w];
        wts[(size_t) t * IDX_NH + h] = total * scale;
    }
}

// scores[t][b] = sum_h wts[t][h] * relu(q[t][h] . pooled[b]) for the pools visible to the query, -inf for the rest.
// grid (ceil(np / 8), T), 256 threads: a warp per pool, the query staged in shared memory.
__global__ void idx_scores_kernel(const float* q, const float* wts, const float* pooled, float* scores, int np, int pos0) {
    extern __shared__ float qs[];                     // [nh * d]
    __shared__ float ws[IDX_NH];
    const int t = (int) blockIdx.y;
    for (int i = threadIdx.x; i < IDX_NH * IDX_D; i += blockDim.x) qs[i] = q[(size_t) t * IDX_NH * IDX_D + i];
    if (threadIdx.x < IDX_NH) ws[threadIdx.x] = wts[(size_t) t * IDX_NH + threadIdx.x];
    __syncthreads();
    const int warp = (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31);
    const int b = (int) blockIdx.x * 8 + warp;
    if (b >= np) return;
    const int n_vis = (pos0 + t + 1) / IDX_KPOOL;
    float s;
    if (b >= n_vis) {
        s = -INFINITY;
    } else {
        float pv[4];
        for (int j = 0; j < 4; ++j) pv[j] = pooled[(size_t) b * IDX_D + lane + 32 * j];
        float acc = 0.0f;
        for (int h = 0; h < IDX_NH; ++h) {
            float dot = 0.0f;
            for (int j = 0; j < 4; ++j) dot += qs[h * IDX_D + lane + 32 * j] * pv[j];
            for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
            acc += ws[h] * fmaxf(dot, 0.0f);
        }
        s = acc;
    }
    if (lane == 0) scores[(size_t) t * np + b] = s;
}

__global__ void idx_fill_ids(int* ids, int np, long long total) {
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long) gridDim.x * blockDim.x)
        ids[i] = (int) (i % np);
}
__global__ void idx_fill_offsets(int* offsets, int np, int T) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i <= T) offsets[i] = i * np;
}

// The cells token t reads.  n_vis <= 512 pools visible: every cell 0..q (dense).  Otherwise the 512 best pools' cells, then the
// trailing partial pool's cells [n_vis*4 .. q].
__global__ void idx_cells_kernel(const int* sorted_ids, int np, int pos0, int* cells, int* ncells) {
    const int t = (int) blockIdx.x;
    const int q = pos0 + t, n_vis = (q + 1) / IDX_KPOOL;
    int* out = cells + (size_t) t * IDX_CELL_STRIDE;
    if (n_vis <= IDX_N_SEL) {
        for (int j = threadIdx.x; j <= q; j += blockDim.x) out[j] = j;
        if (threadIdx.x == 0) ncells[t] = q + 1;
        return;
    }
    const int* ids = sorted_ids + (size_t) t * np;
    for (int idx = threadIdx.x; idx < IDX_N_SEL * IDX_KPOOL; idx += blockDim.x) out[idx] = ids[idx >> 2] * IDX_KPOOL + (idx & 3);
    const int tail_start = n_vis * IDX_KPOOL, tail = q - tail_start + 1;
    if ((int) threadIdx.x < tail) out[IDX_N_SEL * IDX_KPOOL + threadIdx.x] = tail_start + threadIdx.x;
    if (threadIdx.x == 0) ncells[t] = IDX_N_SEL * IDX_KPOOL + (tail > 0 ? tail : 0);
}

struct LayerIdx {
    float* rows = nullptr;      size_t rows_cap = 0;      // positions
    float* pooled = nullptr;    size_t pooled_cap = 0;    // pools
    int valid_upto = 0;         // rows [0, valid_upto) are written and contiguous
};

struct Module {
    int capacity = 32768;
    std::unordered_map<const void*, LayerIdx> layers;
    std::unordered_map<const float*, float*> fweights;
    // write-side scratch
    float* kraw = nullptr; size_t wr_tokens = 0;
    void* xq = nullptr; size_t xq_bytes = 0;
    // read-side scratch
    float *qbuf = nullptr, *wts = nullptr, *scores = nullptr, *keys_out = nullptr;
    int *vals_in = nullptr, *vals_out = nullptr, *offsets = nullptr, *cells = nullptr, *ncells = nullptr;
    void *qq = nullptr, *sort_tmp = nullptr;
    size_t rd_tokens = 0, rd_items = 0, qq_bytes = 0, sort_tmp_bytes = 0, wts_cap = 0, off_cap = 0, cells_cap = 0, ncells_cap = 0;
    std::mutex mutex;
    ~Module() {
        for (auto& kv : layers) { cudaFree(kv.second.rows); cudaFree(kv.second.pooled); }
        for (auto& kv : fweights) cudaFree(kv.second);
        cudaFree(kraw); cudaFree(xq);
        cudaFree(qbuf); cudaFree(wts); cudaFree(scores); cudaFree(keys_out);
        cudaFree(vals_in); cudaFree(vals_out); cudaFree(offsets); cudaFree(cells); cudaFree(ncells); cudaFree(qq); cudaFree(sort_tmp);
    }
    float* fweight(const float* host, size_t n) {
        auto it = fweights.find(host);
        if (it != fweights.end()) return it->second;
        void* d = nullptr;
        if (!vram_malloc(&d, n * sizeof(float))) return nullptr;
        if (cudaMemcpy(d, host, n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) { cudaFree(d); return nullptr; }
        fweights.emplace(host, (float*) d);
        return (float*) d;
    }
    // positions rows / pooled keys, grown (and copied) by doubling
    bool ensure_layer(LayerIdx& L, size_t positions) {
        if ((size_t) capacity < positions) return false;
        if (positions > L.rows_cap) {
            size_t cap = std::max<size_t>(L.rows_cap ? L.rows_cap : 4096, 4096);
            while (cap < positions) cap *= 2;
            cap = std::min<size_t>(cap, (size_t) capacity);
            void* nr = nullptr; void* np = nullptr;
            const size_t pcap = cap / IDX_KPOOL + 1;
            if (!vram_malloc(&nr, cap * 2 * IDX_D * sizeof(float))) return false;
            if (!vram_malloc(&np, pcap * IDX_D * sizeof(float))) { cudaFree(nr); return false; }
            if (L.rows) {
                cudaMemcpy(nr, L.rows, L.rows_cap * 2 * IDX_D * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(np, L.pooled, L.pooled_cap * IDX_D * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaFree(L.rows); cudaFree(L.pooled);
            }
            L.rows = (float*) nr; L.rows_cap = cap; L.pooled = (float*) np; L.pooled_cap = pcap;
        }
        return true;
    }
};
Module& mod() { static Module m; return m; }

template <class T>
bool grow(T*& p, size_t& cap_elems, size_t need) {
    if (p && need <= cap_elems) return true;
    if (p) cudaFree(p);
    p = nullptr; cap_elems = 0;
    void* q = nullptr;
    if (!vram_malloc(&q, need * sizeof(T))) return false;
    p = (T*) q; cap_elems = need;
    return true;
}

}  // namespace

void vram_set_reclaim(VramReclaimFn fn) { g_reclaim = fn; }

bool vram_malloc(void** p, size_t bytes) {
    if (cudaMalloc(p, bytes) == cudaSuccess) return true;
    (void) cudaGetLastError();
    if (g_reclaim != nullptr && g_reclaim(bytes) && cudaMalloc(p, bytes) == cudaSuccess) return true;
    (void) cudaGetLastError();
    *p = nullptr;
    return false;
}

bool idx_available(const MlaWeights& w) {
    using strata::kernels::native_mmvq_supported;
    return w.idx_attn_k && w.idx_attn_q_b && w.idx_c_gate && w.idx_k_norm_w && w.idx_k_norm_b && w.idx_proj && w.idx_ape &&
           w.idx_attn_k_type && w.idx_attn_q_b_type && w.idx_c_gate_type && native_mmvq_supported(w.idx_attn_k_type) &&
           native_mmvq_supported(w.idx_attn_q_b_type) && native_mmvq_supported(w.idx_c_gate_type);
}

void idx_set_capacity(int positions) {
    Module& m = mod();
    std::lock_guard<std::mutex> lock(m.mutex);
    m.capacity = std::max(positions, IDX_SPARSE_FROM);
}

int idx_max_tokens() {
    Module& m = mod();
    const size_t np_max = (size_t) m.capacity / IDX_KPOOL + 1;
    const size_t t = (64ull << 20) / (np_max * 16);
    return (int) std::max<size_t>(8, std::min<size_t>(512, t));
}

int idx_write_device(const MlaWeights& w, const MlaGeometry& g, const void* layer_key, const float* d_x, int T, int pos0,
                     void* stream, char* error, size_t error_capacity) {
    auto decline = [&](const char* msg) { if (error && error_capacity) std::snprintf(error, error_capacity, "%s", msg); return 0; };
    auto fail = [&](const char* msg) { if (error && error_capacity) std::snprintf(error, error_capacity, "%s", msg); return -1; };
    if (!idx_available(w) || !layer_key || !d_x || T < 1 || pos0 < 0) return decline("indexer unavailable");
    const int ne = g.n_embd;
    if (ne % 32 || ne > 16384) return decline("unsupported width");
    Module& m = mod();
    std::lock_guard<std::mutex> lock(m.mutex);
    LayerIdx& L = m.layers[layer_key];
    if (pos0 > L.valid_upto) return decline("indexer rows have a gap (positions ran through a path that does not write them)");
    if (!m.ensure_layer(L, (size_t) pos0 + T)) return decline("indexer cache allocation failed / context beyond capacity");
    const float *nw = m.fweight(w.idx_k_norm_w, IDX_D), *nb = m.fweight(w.idx_k_norm_b, IDX_D),
                *ape = m.fweight(w.idx_ape, (size_t) IDX_KPOOL * IDX_D);
    if (!nw || !nb || !ape) return decline("indexer weight upload failed");
    const size_t xq_need = native_q8_1_bytes(ne, 8);
    if (m.xq == nullptr || m.xq_bytes < xq_need) {
        if (m.xq) cudaFree(m.xq);
        m.xq = nullptr;
        if (!vram_malloc(&m.xq, xq_need)) return decline("indexer scratch allocation failed");
        m.xq_bytes = xq_need;
    }
    if (!grow(m.kraw, m.wr_tokens, (size_t) 2 * T * IDX_D)) return decline("indexer scratch allocation failed");
    float* graw = m.kraw + (size_t) T * IDX_D;     // [T][128] gate halves sit after the [T][128] key halves
    using namespace strata::kernels;
    cudaStream_t st = (cudaStream_t) stream;
    for (int gi = 0; gi < T; gi += 8) {
        const int gr = std::min(8, T - gi);
        native_quantize_q8_1(d_x + (size_t) gi * ne, m.xq, ne, gr, stream);
        native_mmvq(w.idx_attn_k_type, w.idx_attn_k, m.xq, m.kraw + (size_t) gi * IDX_D, ne, IDX_D, gr, stream);
        native_mmvq(w.idx_c_gate_type, w.idx_c_gate, m.xq, graw + (size_t) gi * IDX_D, ne, IDX_D, gr, stream);
    }
    idx_row_kernel<<<T, IDX_D, 0, st>>>(m.kraw, graw, nw, nb, L.rows, pos0);
    const int b_lo = pos0 / IDX_KPOOL, b_hi = (pos0 + T) / IDX_KPOOL;     // pools completed by these rows
    if (b_hi > b_lo) idx_pool_kernel<<<b_hi - b_lo, IDX_D, 0, st>>>(L.rows, ape, L.pooled, b_lo);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return fail(cudaGetErrorString(e));
    L.valid_upto = pos0 + T;
    return 1;
}

int idx_cells_device(const MlaWeights& w, const MlaGeometry& g, const void* layer_key, const float* d_x, const float* d_qr, int T,
                     int pos0, int32_t** d_cells, int32_t** d_ncells, void* stream, char* error, size_t error_capacity) {
    auto decline = [&](const char* msg) { if (error && error_capacity) std::snprintf(error, error_capacity, "%s", msg); return 0; };
    auto fail = [&](const char* msg) { if (error && error_capacity) std::snprintf(error, error_capacity, "%s", msg); return -1; };
    if (!idx_available(w) || !layer_key || !d_x || !d_qr || T < 1 || pos0 < 0 || T > idx_max_tokens())
        return decline("indexer unavailable or too many tokens");
    const int ne = g.n_embd, ql = g.q_lora;
    if (ne % 32 || ql % 32) return decline("unsupported width");
    Module& m = mod();
    std::lock_guard<std::mutex> lock(m.mutex);
    auto lit = m.layers.find(layer_key);
    if (lit == m.layers.end() || lit->second.valid_upto < pos0 + T) return decline("indexer rows not written");
    LayerIdx& L = lit->second;
    const int np = (pos0 + T) / IDX_KPOOL;                 // pools the last token of the block can see
    if (np < 1) return decline("no complete pool");
    const float* proj = m.fweight(w.idx_proj, (size_t) IDX_NH * ne);
    if (!proj) return decline("indexer weight upload failed");
    const size_t items = (size_t) T * np;
    // scratch: q [T][4096], wts [T][32], scores/keys_out [T*np], ids in/out [T*np], offsets [T+1], cells [T][stride], ncells [T]
    size_t c0 = m.rd_tokens;
    if (!grow(m.qbuf, c0, (size_t) T * IDX_NH * IDX_D)) return decline("indexer scratch allocation failed");
    m.rd_tokens = c0;
    if (m.rd_items < items) {
        for (float** p : {&m.scores, &m.keys_out}) { if (*p) cudaFree(*p); *p = nullptr; }
        for (int** p : {&m.vals_in, &m.vals_out}) { if (*p) cudaFree(*p); *p = nullptr; }
        void *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr;
        if (!vram_malloc(&a, items * 4) || !vram_malloc(&b, items * 4) || !vram_malloc(&c, items * 4) || !vram_malloc(&d, items * 4)) {
            cudaFree(a); cudaFree(b); cudaFree(c); cudaFree(d);
            m.rd_items = 0;
            return decline("indexer scratch allocation failed");
        }
        m.scores = (float*) a; m.keys_out = (float*) b; m.vals_in = (int*) c; m.vals_out = (int*) d;
        m.rd_items = items;
    }
    if (!grow(m.wts, m.wts_cap, (size_t) T * IDX_NH) || !grow(m.offsets, m.off_cap, (size_t) T + 1) ||
        !grow(m.cells, m.cells_cap, (size_t) T * IDX_CELL_STRIDE) || !grow(m.ncells, m.ncells_cap, (size_t) T))
        return decline("indexer scratch allocation failed");
    const size_t qq_need = native_q8_1_bytes(ql, 8);
    if (m.qq == nullptr || m.qq_bytes < qq_need) {
        if (m.qq) cudaFree(m.qq);
        m.qq = nullptr;
        if (!vram_malloc(&m.qq, qq_need)) return decline("indexer scratch allocation failed");
        m.qq_bytes = qq_need;
    }
    using namespace strata::kernels;
    cudaStream_t st = (cudaStream_t) stream;
    for (int gi = 0; gi < T; gi += 8) {
        const int gr = std::min(8, T - gi);
        native_quantize_q8_1(d_qr + (size_t) gi * ql, m.qq, ql, gr, stream);
        native_mmvq(w.idx_attn_q_b_type, w.idx_attn_q_b, m.qq, m.qbuf + (size_t) gi * IDX_NH * IDX_D, ql, IDX_NH * IDX_D, gr, stream);
    }
    idx_wts_kernel<<<dim3(IDX_NH, T), 256, 0, st>>>(proj, d_x, m.wts, ne, 1.0f / sqrtf((float) IDX_D * (float) IDX_NH));
    idx_scores_kernel<<<dim3((np + 7) / 8, T), 256, (size_t) IDX_NH * IDX_D * sizeof(float), st>>>(m.qbuf, m.wts, L.pooled, m.scores, np, pos0);
    idx_fill_ids<<<(unsigned) std::min<size_t>((items + 255) / 256, 4096), 256, 0, st>>>(m.vals_in, np, (long long) items);
    idx_fill_offsets<<<(T + 1 + 127) / 128, 128, 0, st>>>(m.offsets, np, T);
    size_t tmp_bytes = 0;
    cudaError_t e = cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, tmp_bytes, m.scores, m.keys_out, m.vals_in, m.vals_out,
                                                                       (int) items, T, m.offsets, m.offsets + 1, 0, 32, st);
    if (e != cudaSuccess) return fail(cudaGetErrorString(e));
    if (m.sort_tmp == nullptr || m.sort_tmp_bytes < tmp_bytes) {
        if (m.sort_tmp) cudaFree(m.sort_tmp);
        m.sort_tmp = nullptr;
        if (!vram_malloc(&m.sort_tmp, tmp_bytes ? tmp_bytes : 1)) return decline("indexer sort scratch allocation failed");
        m.sort_tmp_bytes = tmp_bytes;
    }
    e = cub::DeviceSegmentedRadixSort::SortPairsDescending(m.sort_tmp, tmp_bytes, m.scores, m.keys_out, m.vals_in, m.vals_out,
                                                           (int) items, T, m.offsets, m.offsets + 1, 0, 32, st);
    if (e != cudaSuccess) return fail(cudaGetErrorString(e));
    idx_cells_kernel<<<T, 256, 0, st>>>(m.vals_out, np, pos0, m.cells, m.ncells);
    e = cudaGetLastError();
    if (e != cudaSuccess) return fail(cudaGetErrorString(e));
    *d_cells = m.cells;
    *d_ncells = m.ncells;
    return 1;
}

bool idx_debug_fetch_rows(const void* layer_key, int n, float* host) {
    Module& m = mod();
    std::lock_guard<std::mutex> lock(m.mutex);
    auto it = m.layers.find(layer_key);
    if (it == m.layers.end() || it->second.rows_cap < (size_t) n) return false;
    cudaDeviceSynchronize();
    return cudaMemcpy(host, it->second.rows, (size_t) n * 2 * IDX_D * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
}

}  // namespace strata::kernels::glm
