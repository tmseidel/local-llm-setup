# vLLM Performance Optimization Review

**Date:** 2026-06-05
**Current performance:** ~20 tok/s output (Qwen3-32B-AWQ)
**Target:** Identify bottlenecks and concrete improvements

---

## Current Configuration Summary

| Setting                    | Value                          |
|----------------------------|--------------------------------|
| Hardware                   | 2× RTX 5060 Ti 16GB (sm_120)  |
| RAM                        | 64 GB                          |
| CPU                        | AMD Ryzen 9 9950X              |
| Model                      | Qwen/Qwen3-32B-AWQ (~19 GB)   |
| Quantization               | awq_marlin                     |
| Attention backend          | TRITON_ATTN                    |
| Parallelism                | PP=2, TP=1                    |
| CUDA graphs                | Enabled (enforce_eager=false)  |
| GPU memory utilization     | 0.92                           |
| Max model len              | 65536 (64K)                    |
| KV cache dtype             | fp8                            |
| RoPE scaling               | YaRN factor 2.0               |
| Max num seqs               | 32                             |
| Prefix caching             | NOT enabled                    |
| Chunked prefill            | NOT explicitly enabled         |
| Speculative decoding       | NOT enabled                    |
| NCCL_P2P_DISABLE           | 1                              |
| FlashInfer                 | Removed                        |

---

## Root Cause Analysis: Why 20 tok/s?

### Bottleneck 1: Pipeline Parallelism Bubble (BIGGEST — ~40-50% loss)

With PP=2, the model's ~60 layers are split: GPU0 runs layers 0-29, GPU1 runs
layers 30-59. For a **single request**, the pipeline looks like:

```
GPU0: [compute layers 0-29] ────idle──── [compute layers 0-29] ...
GPU1: ────idle──── [compute layers 30-59] ────idle──── [compute ...]
```

Each GPU is idle ~50% of the time waiting for the other to finish its stage.
At 20 tok/s with PP=2, the actual per-GPU compute speed is ~40 tok/s — the
pipeline bubble halves it.

**This is the dominant bottleneck for single-stream generation.**

The bubble fills with more concurrent requests (microbatching), but at low
concurrency (1-4 requests, typical for a personal agent), utilization stays poor.

**Mitigation options (ranked by impact):**

a) **Use a smaller model on a single GPU** — eliminates ALL inter-GPU overhead.
   Qwen3-14B-AWQ (~10 GB) fits on one 16 GB card. Expected: 40-70 tok/s.
   Trade-off: model quality.

b) **Switch to TP=2** — both GPUs compute every layer simultaneously (split by
   attention heads/dims), so no bubble. The cost is an all-reduce per layer over
   PCIe. On your hardware (PCIe 4.0 x8 per card, ~16 GB/s each direction, going
   through host RAM with P2P disabled), the all-reduce for a 32B model is
   ~0.5-1ms per layer × ~60 layers ≈ 30-60ms overhead per token. That's
   comparable to PP's bubble penalty, but both GPUs are always busy.

   Net effect: TP=2 likely matches or slightly beats PP=2 at low concurrency
   (1-2 requests), and clearly wins at very low concurrency (1 request).
   PP=2 wins at higher concurrency (8+ requests) where microbatching fills
   the bubble.

   **Recommendation: test TP=2.** Your workload (agent, few parallel requests)
   favors TP over PP.

c) **Speculative decoding** — fills the bubble by generating multiple tokens
   per forward pass. See Optimization 3 below.

### Bottleneck 2: Missing Prefix Caching (free TTFT improvement)

`--enable-prefix-caching` is NOT in your start script. For agent workloads that
repeat system prompts (tool definitions, persona, context), this caches the KV
computation for the shared prefix. First request pays full cost; subsequent
requests skip the cached portion.

Impact: 50-80% TTFT reduction for repeated prompts. No throughput impact on
decode, but frees prefill compute for decode batching.

### Bottleneck 3: Over-Provisioned Context Length (limits batching)

`max_model_len=65536` reserves KV cache blocks for 64K tokens even if typical
requests use 4K-16K. With fp8 KV at 0.125 MiB/token:

- 64K context → 8 GB KV cache reserved (4 GB/GPU at PP=2)
- 16K context → 2 GB KV cache reserved (1 GB/GPU at PP=2)

The unused blocks sit idle. If your actual max context usage is ~16K-32K,
reducing max_model_len frees significant VRAM for:
- More concurrent sequences (better batching)
- Higher gpu_memory_utilization headroom

**Question: what's your typical and maximum context size in practice?**

### Bottleneck 4: AWQ Marlin vs NVFP4 on Blackwell (17% speed gap)

Recent benchmarks (arXiv:2601.09527, Jan 2026) show **NVFP4 quantization is
~17% faster than AWQ Marlin on SM_120** Blackwell consumer GPUs. NVFP4 uses
the native E2M1 codebook with FP8 block scaling, which maps directly to
Blackwell's hardware 4-bit datapath.

If an NVFP4 checkpoint of Qwen3-32B exists (or can be created), switching to
`--quantization nvfp4` would be a direct ~17% speedup with no other changes.

### Bottleneck 5: No Chunked Prefill Configuration

V1 engine (default since vLLM 0.8.x) enables chunked prefill by default, but
it's not explicitly configured. For long agent prompts (tool definitions can be
thousands of tokens), explicit tuning helps:

- `--max-num-batched-tokens 4096` — controls chunk size
- `--max-num-partial-prefills 2` — allows overlapping prefill with decode

### Bottleneck 6: gpu_memory_utilization Could Be Higher

0.92 is conservative for a single-tenant setup. Going to 0.95 gains ~0.5 GB
more VRAM per GPU for KV cache blocks — small but free.

---

## Concrete Optimizations (Ranked by Impact)

### Optimization 1: Switch to TP=2 (estimated +15-30% for low concurrency)

Replace PP=2 with TP=2 in your config:

```yaml
# group_vars/all.yml
vllm_tensor_parallel_size: 2
vllm_pipeline_parallel_size: 1
```

Why: At 1-4 concurrent requests (typical agent workload), TP=2 keeps both GPUs
busy 100% of the time. PP=2's bubble wastes ~50% at low concurrency. The
all-reduce cost over PCIe is comparable to the bubble cost, but without the
idle time.

If you find throughput worse at higher concurrency (8+ parallel agents), switch
back to PP=2.

### Optimization 2: Enable Prefix Caching (estimated +30-50% TTFT)

Add to vllm_start.sh.j2:

```bash
--enable-prefix-caching \
```

And add to group_vars/all.yml:

```yaml
vllm_enable_prefix_caching: true
```

This is free performance for any workload with repeated prompt prefixes.

### Optimization 3: Speculative Decoding (estimated +50-100% decode speed)

Use a small same-family draft model. Qwen3 has 0.6B, 1.7B, 4B, and 8B variants
that could serve as drafts. Best option: **Qwen3-4B-AWQ** (~3 GB) or even
**Qwen3-0.6B** (FP16, ~1.2 GB).

```bash
--speculative-model Qwen/Qwen3-4B-AWQ \
--speculative-model-quantization awq_marlin \
--num-speculative-tokens 5 \
--speculative-draft-tensor-parallel-size 1 \
```

The draft model runs on one GPU (TP=1) while the target model spans both.
With PP=2, the draft fits in the idle GPU's spare VRAM during bubble time.
With TP=2, the draft model needs to share VRAM — tighter but possible with
a 0.6B draft.

Expected: 30-50 tok/s (from 20 tok/s baseline) for single-stream generation.

**Caveat:** Speculative decoding adds VRAM for the draft model. With TP=2 and
a 0.6B draft (~1.2 GB FP16), total VRAM would be ~19 GB weights + 1.2 GB draft
+ KV cache = tight in 32 GB. Monitor with nvidia-smi.

### Optimization 4: NVFP4 Quantization (estimated +17% if available)

Check if a Qwen3-32B NVFP4 checkpoint exists:

```bash
# Search HuggingFace for NVFP4 variants
# Example: neuralmagic/Qwen3-32B-FP4 or similar
```

If available:

```yaml
vllm_quantization: "nvfp4"
vllm_model: "<nvfp4-checkpoint>"
```

NVFP4 uses Blackwell's native 4-bit datapath and is measurably faster than
AWQ Marlin on SM_120.

### Optimization 5: Reduce max_model_len (frees VRAM for batching)

If you don't regularly use 64K context:

```yaml
# For 32K context (sufficient for most agent work)
vllm_max_model_len: 32768
vllm_rope_scaling: ''   # Empty = use native 32K, no YaRN overhead
```

VRAM savings at fp8:
- 64K → 8 GB KV total → freed 4 GB
- 32K → 4 GB KV total → freed 2 GB
- 16K → 2 GB KV total → freed 6 GB

Each 2 GB freed ≈ 1000 more KV blocks ≈ more concurrent sequences.

If you DO need 64K for the Hermes agent, keep it — but consider whether
the 64K is actually used in practice or just reserved.

### Optimization 6: Increase GPU Memory Utilization (small but free)

```yaml
vllm_gpu_memory_utilization: "0.95"
```

Safe because nothing else runs on the GPUs. Gains ~0.5 GB/GPU for KV blocks.

### Optimization 7: Enable Metrics for Monitoring

Add to start script:

```bash
--enable-metrics \
```

This exposes Prometheus metrics on the default port. Essential for understanding
where time is spent:

```bash
curl http://localhost:8001/metrics | grep vllm
```

Key metrics:
- `vllm:time_to_first_token_seconds` — TTFT
- `vllm:time_per_output_token_seconds` — decode latency (the 20 tok/s metric)
- `vllm:gpu_cache_usage_perc` — KV cache utilization
- `vllm:num_requests_running` — active concurrency
- `vllm:e2e_request_latency_seconds` — total request time

---

## Recommended Config Changes (all.yml diff)

```diff
 # group_vars/all.yml — Optimization changes

 # Try TP=2 instead of PP=2 for low-concurrency agent workload
-vllm_tensor_parallel_size: 1
-vllm_pipeline_parallel_size: 2
+vllm_tensor_parallel_size: 2
+vllm_pipeline_parallel_size: 1

 # Push memory utilization higher (single-tenant, safe)
-vllm_gpu_memory_utilization: "0.92"
+vllm_gpu_memory_utilization: "0.95"

 # Add prefix caching (free TTFT improvement)
+vllm_enable_prefix_caching: true

 # Reduce context if 64K isn't needed in practice
-# vllm_max_model_len: 65536
+# vllm_max_model_len: 32768   # Uncomment if 64K not needed
+# vllm_rope_scaling: ''       # Remove YaRN overhead if using native 32K

 # Enable metrics for performance monitoring
+vllm_enable_metrics: true
```

## Start Script Additions (vllm_start.sh.j2)

```diff
 exec {{ vllm_venv_dir }}/bin/vllm serve "{{ vllm_model }}" \
   --host "{{ vllm_host }}" \
   --port {{ vllm_port }} \
+{% if vllm_enable_prefix_caching | default(false) | bool %}
+  --enable-prefix-caching \
+{% endif %}
+{% if vllm_enable_metrics | default(false) | bool %}
+  --enable-metrics \
+{% endif %}
 {% if vllm_enable_tool_calling | bool %}
   --enable-auto-tool-choice \
   --tool-call-parser {{ vllm_tool_call_parser }} \
```

---

## Expected Performance After Optimization

| Scenario                        | Current  | Estimated After |
|---------------------------------|----------|-----------------|
| TP=2 + prefix cache + 0.95 util | 20 tok/s | 28-35 tok/s     |
| + speculative decoding          | 20 tok/s | 40-55 tok/s     |
| + NVFP4 (if available)          | 20 tok/s | 47-65 tok/s     |
| 14B model on single GPU         | N/A      | 50-80 tok/s     |

These are estimates. Actual numbers depend on concurrency, prompt length, and
whether CUDA graphs capture all shapes correctly on SM_120.

---

## Quick Test Plan

1. Apply TP=2 + prefix caching + metrics (Optimizations 1, 2, 6)
2. Deploy and measure baseline with:
   ```bash
   curl http://localhost:8001/v1/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"Qwen3-32B-AWQ","prompt":"Write a story about a robot.","max_tokens":256}'
   ```
3. Check metrics: `curl http://localhost:8001/metrics | grep time_per_output`
4. If TP=2 is worse than PP=2 at your concurrency, switch back to PP=2
5. Add speculative decoding (Optimization 3) if further speed needed
6. Search for NVFP4 checkpoint (Optimization 4) as a bonus

---

## What NOT to Change

- **TRITON_ATTN** — correct for SM_120, keep it
- **awq_marlin** — correct quantization kernel, keep it (unless switching to NVFP4)
- **FlashInfer removed** — correct, keep it removed
- **NCCL_P2P_DISABLE=1** — correct for GeForce without NVLink, keep it
- **CUDA graphs enabled** — correct, keep enforce_eager=false
- **VLLM_USE_FLASHINFER_SAMPLER=0** — correct, keep it
