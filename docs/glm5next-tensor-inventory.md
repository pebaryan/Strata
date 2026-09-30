# GLM-5.3-Flash tensor inventory (generated)

Produced by `tools/glm5_inventory.py` from unsloth/GLM-5.3-Flash-GGUF UD-IQ1_S (3 shards, 93.1 GB),
reading only the metadata and tensor-info tables. Regenerate with:

```bash
python3 tools/glm5_inventory.py /path/to/GLM-5.3-Flash-UD-IQ1_S-0000{1,2,3}-of-00003.gguf
```

# tensor inventory

## GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf

- GGUF v3, 0 tensors, 72 metadata keys

## GLM-5.3-Flash-UD-IQ1_S-00002-of-00003.gguf

- GGUF v3, 771 tensors, 3 metadata keys
  - embedding_head           2 tensors
  - hyper_connection       150 tensors
  - linear_attn_ssm        209 tensors
  - mla_attention          128 tensors
  - moe_dense_stem           9 tensors
  - moe_experts             67 tensors
  - moe_router              45 tensors
  - moe_shared_expert       67 tensors
  - norm                    52 tensors
  - sparse_indexer          42 tensors

## GLM-5.3-Flash-UD-IQ1_S-00003-of-00003.gguf

- GGUF v3, 641 tensors, 3 metadata keys
  - hyper_connection       120 tensors
  - linear_attn_ssm        165 tensors
  - mla_attention          104 tensors
  - moe_experts             62 tensors
  - moe_router              41 tensors
  - moe_shared_expert       62 tensors
  - mtp_nextn                4 tensors
  - norm                    41 tensors
  - sparse_indexer          42 tensors

## totals by role (all shards)

| role | tensors | approx bytes | distinct dtypes |
|---|---:|---:|---|
| moe_experts | 129 | 86.41 GB | IQ1_Sx56, IQ3_XXSx39, IQ2_XXSx28, IQ4_XSx3, Q2_Kx2, Q3_Kx1 |
| mla_attention | 232 | 4.31 GB | Q5_Kx158, Q8_0x48, F32x24, Q6_Kx2 |
| moe_shared_expert | 129 | 0.80 GB | Q5_Kx84, Q6_Kx44, Q8_0x1 |
| embedding_head | 2 | 0.71 GB | Q4_Kx2 |
| moe_dense_stem | 9 | 0.33 GB | Q5_Kx6, Q6_Kx3 |
| moe_router | 86 | 0.20 GB | F32x86 |
| linear_attn_ssm | 374 | 0.14 GB | F32x204, Q8_0x170 |
| sparse_indexer | 84 | 0.10 GB | F32x48, Q8_0x36 |
| hyper_connection | 270 | 0.04 GB | F32x180, Q8_0x90 |
| mtp_nextn | 4 | 0.04 GB | F32x3, Q8_0x1 |
| norm | 93 | 0.00 GB | F32x93 |

* 1412 tensors total; quant types seen: F32x638, Q8_0x346, Q5_Kx248, IQ1_Sx56, Q6_Kx49, IQ3_XXSx39, IQ2_XXSx28, IQ4_XSx3, Q4_Kx2, Q2_Kx2, Q3_Kx1

## architecture metadata (the guard must know these)

- `general.architecture` = glm5next
- `glm5next.attention.head_count` = 64
- `glm5next.attention.head_count_kv` = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,...
- `glm5next.attention.indexer.head_count` = 32
- `glm5next.attention.indexer.key_length` = 128
- `glm5next.attention.indexer.kpool` = 4
- `glm5next.attention.indexer.top_k` = 2048
- `glm5next.attention.key_length` = 512
- `glm5next.attention.key_length_mla` = 256
- `glm5next.attention.kv_lora_rank` = 512
- `glm5next.attention.layer_norm_epsilon` = 9.999999974752427e-07
- `glm5next.attention.layer_norm_rms_epsilon` = 9.999999747378752e-06
- `glm5next.attention.q_lora_rank` = 1536
- `glm5next.attention.value_length` = 512
- `glm5next.attention.value_length_mla` = 256
- `glm5next.block_count` = 46
- `glm5next.context_length` = 1048576
- `glm5next.embedding_length` = 4096
- `glm5next.expert_count` = 288
- `glm5next.expert_feed_forward_length` = 2048
- `glm5next.expert_gating_func` = 2
- `glm5next.expert_group_count` = 1
- `glm5next.expert_group_used_count` = 1
- `glm5next.expert_shared_count` = 1
- `glm5next.expert_shared_feed_forward_length` = 2048
- `glm5next.expert_used_count` = 8
- `glm5next.expert_weights_norm` = True
- `glm5next.expert_weights_scale` = 2.5
- `glm5next.hyper_connection.count` = 4
- `glm5next.hyper_connection.epsilon` = 9.999999974752427e-07
- `glm5next.hyper_connection.sinkhorn_iterations` = 20
- `glm5next.kda.gate_lower_bound` = -5.0
- `glm5next.kda.head_dim` = 128
- `glm5next.leading_dense_block_count` = 3
- `glm5next.nextn_predict_layers` = 1
- `glm5next.rope.dimension_count` = 0
- `glm5next.vocab_size` = 154880

## per-layer role mix (first 3 and last 2 layers)

- blk.0: hyper_connection x6, linear_attn_ssm x11, mla_attention x4, moe_dense_stem x3, norm x2
- blk.1: hyper_connection x6, linear_attn_ssm x11, mla_attention x4, moe_dense_stem x3, norm x2
- blk.2: hyper_connection x6, linear_attn_ssm x11, mla_attention x4, moe_dense_stem x3, norm x2
- blk.44: hyper_connection x6, linear_attn_ssm x11, mla_attention x4, moe_experts x3, moe_router x2, moe_shared_expert x3, norm x2
- blk.45: mla_attention x8, moe_experts x3, moe_router x2, moe_shared_expert x3, mtp_nextn x4, norm x2, sparse_indexer x7
