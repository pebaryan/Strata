// src/kernels/glm_mla_parity.cpp - does our MLA block match the reference?
//
// The oracle is tools/glm5_mla_reference.py, transcribed from llama.cpp's GLM5-Next `build_mla_layer`, and
// the fixture carries the real weights of a block plus a random cache:
//
//   python tools/glm5_mla_reference.py --gguf <shard1> --layer 3 --tokens 5 --raw-fixture /tmp/mla.bin
//   build-volta/glm_mla_parity /tmp/mla.bin
//
// The intermediates are compared too (qr, qcur, kv, attn), so a mismatch points at the stage that made it
// rather than only at the output.
#include "strata/kernels/glm_mla.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace glm = strata::kernels::glm;

namespace {

std::vector<float> read_floats(std::FILE* f, size_t n) {
    std::vector<float> v(n);
    if (n && std::fread(v.data(), sizeof(float), n, f) != n) {
        std::fprintf(stderr, "fixture is truncated\n");
        std::exit(1);
    }
    return v;
}

/// Gated on the error relative to the ARRAY's own scale: float32 round-off is proportional to magnitude,
/// so a per-element ratio is meaningless for the small entries a normalized operator produces.
void compare(const char* what, const std::vector<float>& got, const float* want, size_t n, bool& ok) {
    double scale = 1e-30, ma = 0.0, mr = 0.0;
    for (size_t i = 0; i < n; ++i) scale = std::max(scale, std::fabs((double) want[i]));
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double) got[i] - (double) want[i]);
        ma = std::max(ma, d);
        if (std::fabs((double) want[i]) >= 0.1 * scale)
            mr = std::max(mr, d / std::fabs((double) want[i]));
    }
    const double rel_of_scale = ma / scale;
    const bool good = rel_of_scale < 1e-5 && mr < 1e-4;
    std::printf("  %-8s max abs %.3e (= %.2e of scale)   worst element rel %.3e   %s\n", what, ma,
                rel_of_scale, mr, good ? "PASS" : "FAIL");
    ok = ok && good;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--attention-selftest") {
        const int heads=4,head_dim=256,kv_lora=512;
        bool all_ok=true;
        for (int n_cache : {1,5,257,8192}) {
            std::vector<float> q((size_t)heads*kv_lora),cache((size_t)n_cache*kv_lora);
            for(size_t i=0;i<q.size();++i) q[i]=(float)((int)((i*17+3)%101)-50)*0.001f;
            for(size_t i=0;i<cache.size();++i) cache[i]=(float)((int)((i*29+7)%127)-63)*0.001f;
            std::vector<float> cpu(q.size(),0.0f),gpu(q.size(),0.0f);
            const float scale=1.0f/std::sqrt((float)head_dim);
            for(int h=0;h<heads;++h) {
                std::vector<float> scores((size_t)n_cache);
                float best=-INFINITY;
                for(int t=0;t<n_cache;++t) {
                    float sum=0.0f;
                    for(int i=0;i<kv_lora;++i) sum+=q[(size_t)h*kv_lora+i]*cache[(size_t)t*kv_lora+i];
                    scores[(size_t)t]=sum*scale; best=std::max(best,scores[(size_t)t]);
                }
                double denom=0.0;
                for(float& s:scores) { s=std::exp(s-best); denom+=s; }
                for(int t=0;t<n_cache;++t) for(int i=0;i<kv_lora;++i)
                    cpu[(size_t)h*kv_lora+i]+=(float)(scores[(size_t)t]/denom)*cache[(size_t)t*kv_lora+i];
            }
            char error[256]={};
            const bool call_ok=glm::mla_attention_cuda(q.data(),cache.data(),n_cache,heads,head_dim,kv_lora,
                                                        gpu.data(),error,sizeof(error));
            double max_abs=0.0,scale_ref=1e-30;
            if(call_ok) for(size_t i=0;i<cpu.size();++i) {
                max_abs=std::max(max_abs,std::fabs((double)gpu[i]-cpu[i]));
                scale_ref=std::max(scale_ref,std::fabs((double)cpu[i]));
            }
            const bool pass=call_ok&&max_abs/scale_ref<5e-5;
            std::printf("  cache %-5d max abs %.3e (%.2e of scale) %s%s\n",n_cache,max_abs,max_abs/scale_ref,
                        pass?"PASS":"FAIL",call_ok?"":error);
            all_ok=all_ok&&pass;
        }
        std::printf("MLA CUDA attention selftest: %s\n",all_ok?"PASS":"FAIL");
        return all_ok?0:1;
    }
    if (argc < 2) {
        std::printf("usage: glm_mla_parity <fixture.bin> | --attention-selftest\n");
        return 2;
    }
    std::FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }

    int32_t hdr[7] = {0};
    if (std::fread(hdr, sizeof(int32_t), 7, f) != 7) { std::fprintf(stderr, "bad header\n"); return 1; }
    glm::MlaGeometry g;
    g.n_embd = hdr[0];
    g.n_head = hdr[2];
    g.head_dim = hdr[3];
    g.kv_lora = hdr[4];
    g.q_lora = hdr[5];
    const int n_cache = hdr[6];
    if (g.n_embd <= 0 || g.n_head <= 0 || g.head_dim <= 0 || g.kv_lora <= 0 || g.q_lora <= 0 || n_cache <= 0) {
        std::fprintf(stderr, "bad geometry in the fixture\n");
        return 1;
    }
    const int q_dim = g.n_head * g.head_dim;

    glm::MlaWeights w;
    std::vector<float> wq_a = read_floats(f, (size_t) g.q_lora * g.n_embd);
    std::vector<float> q_a_norm = read_floats(f, (size_t) g.q_lora);
    std::vector<float> wq_b = read_floats(f, (size_t) q_dim * g.q_lora);
    std::vector<float> wk_b = read_floats(f, (size_t) g.n_head * g.kv_lora * g.head_dim);
    std::vector<float> kv_a = read_floats(f, (size_t) g.kv_lora * g.n_embd);
    std::vector<float> kv_a_norm = read_floats(f, (size_t) g.kv_lora);
    std::vector<float> wv_b = read_floats(f, (size_t) g.n_head * g.head_dim * g.kv_lora);
    std::vector<float> wo = read_floats(f, (size_t) g.n_embd * q_dim);
    std::vector<float> x = read_floats(f, (size_t) g.n_embd);
    std::vector<float> cache = read_floats(f, (size_t) n_cache * g.kv_lora);
    const std::vector<float> e_qr = read_floats(f, (size_t) g.q_lora);
    const std::vector<float> e_qcur = read_floats(f, (size_t) g.n_head * g.kv_lora);
    const std::vector<float> e_kv = read_floats(f, (size_t) g.kv_lora);
    const std::vector<float> e_attn = read_floats(f, (size_t) g.n_head * g.kv_lora);
    const std::vector<float> e_out = read_floats(f, (size_t) g.n_embd);
    std::fclose(f);

    w.wq_a = wq_a.data();
    w.q_a_norm = q_a_norm.data();
    w.wq_b = wq_b.data();
    w.wk_b = wk_b.data();
    w.kv_a = kv_a.data();
    w.kv_a_norm = kv_a_norm.data();
    w.wv_b = wv_b.data();
    w.wo = wo.data();

    std::printf("MLA parity vs tools/glm5_mla_reference.py: n_embd %d, %d head(s) x %d, kv_lora %d, "
                "q_lora %d, cache %d\n", g.n_embd, g.n_head, g.head_dim, g.kv_lora, g.q_lora, n_cache);

    std::vector<float> got_qr((size_t) g.q_lora), got_qcur((size_t) g.n_head * g.kv_lora);
    std::vector<float> got_kv((size_t) g.kv_lora), got_attn((size_t) g.n_head * g.kv_lora);
    std::vector<float> got_out((size_t) g.n_embd);
    glm::MlaIntermediates want;
    want.qr = got_qr.data();
    want.qcur = got_qcur.data();
    want.kv = got_kv.data();
    want.attn = got_attn.data();
    glm::mla_forward(w, g, x.data(), n_cache, cache.data(), got_out.data(), want);

    bool ok = true;
    compare("qr", got_qr, e_qr.data(), e_qr.size(), ok);
    compare("kv", got_kv, e_kv.data(), e_kv.size(), ok);
    compare("qcur", got_qcur, e_qcur.data(), e_qcur.size(), ok);
    compare("attn", got_attn, e_attn.data(), e_attn.size(), ok);
    compare("out", got_out, e_out.data(), e_out.size(), ok);

    // structural invariants, independent of the fixture: with a one-entry cache attention must return that
    // entry exactly for every head, and every attended row must be a convex combination of cache rows.
    {
        std::vector<float> one_out((size_t) g.n_embd), one_attn((size_t) g.n_head * g.kv_lora);
        glm::MlaIntermediates w2;
        w2.attn = one_attn.data();
        glm::mla_forward(w, g, x.data(), 1, cache.data(), one_out.data(), w2);
        double worst = 0.0;
        for (int h = 0; h < g.n_head; ++h)
            for (int i = 0; i < g.kv_lora; ++i)
                worst = std::max(worst, std::fabs((double) one_attn[(size_t) (h * g.kv_lora + i)] -
                                                  (double) cache[(size_t) i]));
        std::printf("invariants: with a 1-entry cache every head returns that entry (worst %.3e)\n", worst);
        if (worst > 1e-5) { std::printf("  invariants FAIL\n"); ok = false; }
    }

    std::printf("glm_mla_parity: %s\n", ok ? "0 failures" : "FAILURES");
    return ok ? 0 : 1;
}
