# Local LLM Stack — Analysis & Fix Plan

**Date:** 2026-06-04
**Target host:** `inferencekoriander` (Ubuntu 24.04, AMD Ryzen 9 9950X, 64 GB RAM)
**GPU:** 2× **NVIDIA GeForce RTX 5060 Ti 16 GB** → **32 GB VRAM total**, Blackwell, compute capability **SM 12.0 (sm_120)**, driver 595.71.05 / CUDA 13.2

> **Status (2026-06-04): all fixes in §8 applied** to the ansible tree. All three playbooks pass `--syntax-check`; the rendered `vllm_start.sh` passes `bash -n`. Still requires a human: **rotate the leaked HF token** on huggingface.co and scrub it from git history; decide on vault/SSH-keys for the inventory password.

---

## 0. The single most important design correction

The current config assumes **Ollama and vLLM run side-by-side and share the 32 GB** (low VRAM fractions, "passt knapp neben Ollama" comments everywhere). You told me the opposite:

> I want to run **either** ollama **or** vllm, they won't run in parallel, so every app can use the whole 32 GB VRAM.

But the playbook does `state: started` + `enabled: true` for **both** the `ollama` and `vllm` systemd services. On boot they both start and fight over VRAM — whichever loads second OOMs. This alone can look like "vLLM does not work at all."

**Fix (architectural):** introduce a single switch and make the two engines mutually exclusive.

```yaml
# group_vars/all.yml
inference_engine: "vllm"   # "vllm" | "ollama"
```

- When `vllm`: enable+start vLLM, **stop + disable** Ollama (and vice-versa).
- Open WebUI stays up and points at whichever engine is active.
- Both engines get to use the full GPU → bump VRAM utilization (see §3).

This is the backbone of every other fix below.

---

## 1. Findings, ranked by severity

| # | Severity | File | Problem | Effect |
|---|----------|------|---------|--------|
| 1 | 🔴 Blocker | `vllm/defaults/main.yml:46` | `vllm_service_restart_sec: 10 I` — stray ` I` | Invalid `RestartSec=10 I` → systemd unit fails to load → **vLLM never starts** |
| 2 | 🔴 Blocker | `vllm/templates/vllm_start.sh.j2:24` | `VLLM_ATTENTION_BACKEND=FLASH_ATTN` | On sm_120, FLASH_ATTN throws `undefined symbol` / unsupported. Confirmed broken for RTX 5060 Ti |
| 3 | 🔴 Blocker | site.yml / role design | Ollama **and** vLLM both `started`+`enabled` | Both grab VRAM on boot → second one OOMs |
| 4 | 🟠 Major | `ollama/tasks/main.yml:58,61` | `OLLAMA_GPU_MEMORY_FRACTION` and `OLLAMA_NUM_CTX` are **not real** Ollama env vars | Settings silently ignored; context stays at default 4096 |
| 5 | 🟠 Major | `group_vars/all.yml:53` | **Real HuggingFace token committed in git** | Secret leak — must rotate + remove |
| 6 | 🟠 Major | `inventory.ini:5-6` | `ansible_password=tom` in plaintext | Credential leak |
| 7 | 🟠 Major | `docker/tasks/main.yml` + `vllm` | `ansible_user_id` under `become: true` resolves to **root**, not `tom` | Adds *root* to docker group; vLLM service runs as root with HF cache in root-owned dirs |
| 8 | 🟡 Medium | `nvidia/tasks/main.yml` + `all.yml:59` | `nvidia_driver_min_version: 525` / "CUDA 12.x" | Blackwell needs driver **≥ 570**. Fresh `ubuntu-drivers install` may pick a too-old driver |
| 9 | 🟡 Medium | `vllm/tasks/main.yml:84,29` | `python -m vllm --version` is not a valid invocation | Version never detected (masked by `failed_when: false`) |
| 10 | 🟡 Medium | `vllm` install | Plain `pip install vllm` with system CUDA 13.2 present | Possible `libcudart.so.12` mismatch + FlashInfer JIT needs a toolchain |
| 11 | 🟡 Medium | README + `vllm/defaults` comments | Hallucinated models (`Qwen3.6-27B`, `Qwen3.6-35B-A3B`, "~15 GB") | Misleading; sizes wrong (see §4) |
| 12 | 🟢 Minor | `open_webui/tasks/main.yml:43` | WebUI only wired to Ollama (`OLLAMA_BASE_URL`) | Can't see vLLM when vLLM is the active engine |
| 13 | 🟢 Minor | `vllm_start.sh.j2:4-5` | Mojibake (`�`) — file not UTF-8 | Cosmetic |
| 14 | 🟢 Minor | `vllm/defaults` vs `all.yml` | `vllm_quantization` is `awq` in defaults, `none` in all.yml; `vllm_max_model_len` 32768 vs 30000; util 0.90 vs 0.85 | Confusing duplication; all.yml wins, but easy to misread |

---

## 2. The vLLM Blackwell problem (deep dive)

This is the core of "vLLM does not work at all." There are **two independent blockers** plus tuning.

### 2a. Broken systemd unit (`RestartSec=10 I`)
`vllm/defaults/main.yml:46`:
```yaml
vllm_service_restart_sec: 10 I   # ← stray " I"
```
renders to `RestartSec=10 I` in the unit → systemd rejects the directive. Depending on version the unit fails to load entirely. **Fix:** `vllm_service_restart_sec: 10`.

### 2b. Wrong attention backend for sm_120
`vllm_start.sh.j2` forces:
```bash
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```
Field reports for the **exact RTX 5060 Ti / sm_120** confirm FLASH_ATTN fails (undefined symbol / unsupported head size). Two backends are known-good on Blackwell:

- **`TRITON_ATTN`** + **`awq_marlin`** quantization — works out of the box, no extra toolchain. **Recommended** for this setup.
- **`FLASHINFER`** — also works but needs a JIT compile toolchain present (`gcc`, `python3-dev`, `nvcc`/cuda-toolkit, `ninja-build`) and is more fragile.

**Fix:** switch the attention backend to Triton **and keep the FlashInfer-sampler disabled** (see 2b-bis — these are two independent FlashInfer code paths):
```bash
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
```

### 2b-ter. FlashInfer keeps resurfacing → remove the package (observed 2026-06-04)
Disabling the FlashInfer *sampler* (2b-bis) wasn't enough: with CUDA graphs on
(`enforce_eager: false`), the FlashInfer **attention** backend got selected during
graph capture and failed again — same `check_cuda_arch() → "requires sm75 or higher"`,
this time via `flashinfer.py … fast_plan_decode` during `profile_cudagraph_memory`.
FlashInfer's JIT simply cannot build for sm_120, and vLLM keeps reaching for it on
multiple independent paths (sampler, attention, decode planning). Per vLLM #40153,
SM120 cards are meant to fall back to TRITON_ATTN + Marlin anyway. **Durable fix:
uninstall `flashinfer-python` from the venv** (Ansible `pip state: absent`, in both
install and update tasks). With the package gone, vLLM can't pick FlashInfer on any
path and falls back to Triton attention + native sampler. `VLLM_ATTENTION_BACKEND=
TRITON_ATTN` stays set so it doesn't auto-pick the (also-broken) `FLASH_ATTN`.
Also fixed related `Permission denied` errors: the **venv** (`/opt/vllm/env`), the
**HF cache** (`/opt/vllm/hf_cache`) and the **log dir** (`/var/log/vllm`) were owned by
root from earlier runs when `vllm_user` was root — so pip-(un)install as `tom` failed,
the HF commit-hash write failed, and the service (now running as `tom`) couldn't append
to `vllm.log`. The role now recursively chowns `{{ vllm_install_dir }}` and
`{{ vllm_log_dir }}` to `vllm_user` before the pip steps.

### 2b-bis. FlashInfer **sampler** also breaks on sm_120 (separate from attention)
Setting `TRITON_ATTN` fixes *attention*, but vLLM still auto-selects FlashInfer for
**top-k/top-p sampling** whenever the `flashinfer` package is importable. On sm_120 its
JIT arch check fails during the memory-profiling dummy run:
```
File ".../flashinfer/jit/core.py", line 108, in check_cuda_arch
    raise RuntimeError("FlashInfer requires GPUs with sm75 or higher")
RuntimeError: Worker failed with error 'FlashInfer requires GPUs with sm75 or higher'
→ EngineCore failed to start.
```
(The message is misleading — sm_120 *is* ≥ sm75; FlashInfer's JIT just can't resolve the
Blackwell arch and bails.) **Fix:** force the PyTorch-native sampler with
`export VLLM_USE_FLASHINFER_SAMPLER=0`. This was observed live on 2026-06-04 and is now in
`vllm_start.sh.j2`. (Alternative: `pip uninstall flashinfer-python` from the venv — then
vLLM never takes the FlashInfer path at all.)
> The exact backend token has changed across vLLM versions (`TRITON_ATTN_VLLM_V1` → `TRITON_ATTN`). Pin it to whatever the installed vLLM accepts — verify with `vllm serve --help | grep -i attention` after install, and expose it as a variable `vllm_attention_backend`.

### 2c. Quantization flag for Blackwell AWQ
For AWQ on sm_120 the working kernel is **`awq_marlin`** (not plain `awq`). With Triton attention this combination is the documented stable path. Set:
```yaml
vllm_quantization: "awq_marlin"
```
…and make the start script always emit `--quantization {{ vllm_quantization }}` unless `none`. (Currently `all.yml` sets `none` so vLLM auto-detects; auto-detect usually resolves AWQ→awq_marlin on new vLLM, but pinning is safer.)

### 2d. CUDA 13.2 vs vLLM's CUDA-12 wheels
vLLM's PyPI wheels bundle PyTorch built for **cu128** (ships `libcudart.so.12`). The host has system CUDA **13.2**. In a clean venv this is normally self-contained, **but** if anything (e.g. FlashInfer JIT) reaches for the system toolchain you can hit `libcudart.so.12: cannot open shared object file`. Mitigations:

- Keep vLLM in its **own venv** (already done) and **do not** put system CUDA on its `LD_LIBRARY_PATH`.
- If the mismatch appears, `pip install nvidia-cuda-runtime-cu12` into the venv.
- Sticking with `TRITON_ATTN` avoids the FlashInfer JIT toolchain requirement entirely.

### 2e. `--enforce-eager`
Currently always on. It's safe (skips CUDA graph capture, which has been flaky on early Blackwell) but ~10–20 % slower. Keep it for the first working boot, then expose `vllm_enforce_eager: true` so it can be turned off once stable.

---

## 3. VRAM / utilization (now that it's either-or)

Since only one engine runs at a time, both can claim almost the whole 32 GB:

| Setting | Current | Proposed | Reason |
|---|---|---|---|
| `vllm_gpu_memory_utilization` | 0.85 (all.yml) / 0.90 (defaults) | **0.92** | No co-tenant; leave a little headroom per 16 GB card |
| `ollama_gpu_memory_fraction` | "0.90" (ignored — not a real var) | **remove** | Not a valid Ollama var; Ollama auto-sizes. Use `OLLAMA_GPU_OVERHEAD` if you must reserve VRAM |
| `vllm_max_model_len` | 30000 | 32768 | Qwen3-32B native |
| `vllm_max_num_seqs` | 32 | 32 ✅ | Fine for agent concurrency |
| `vllm_enforce_eager` | (always on) | **false** | CUDA graphs → big speedup; see §3b |
| parallelism | `tensor=2` | **`tensor=1, pipeline=2`** | No-P2P GeForce: pipeline beats tensor; see §3b |

A 32B AWQ model (~19 GB) does not fit on a single 16 GB card, so it must use **both** GPUs either way. The question is *how* it spans them — see §3b.

---

## 3b. Throughput problem & parallelism choice (observed 2026-06-04)

vLLM served correctly but was **~10 tok/s** for the 32B. Two causes from the log:

1. **`--enforce-eager`** → no CUDA graphs → per-step launch overhead, and Triton kernels JIT-compiled *during* inference (`jit_monitor.py: Triton kernel JIT compilation during inference … latency spike`). **Fix:** `vllm_enforce_eager: false`. Now booting reliably, so the safety net comes off; CUDA-graph warmup also pre-compiles the Triton shapes, removing the spike.

2. **Tensor-parallel across 2× GeForce without NVLink / with PCIe P2P disabled.** TP does an **all-reduce every layer**; with no P2P that traffic is staged through host RAM over PCIe and dominates wall-clock (`GPU KV cache usage: 0.0%` → not memory-bound, it's comms/latency-bound). **Fix (user-chosen): pipeline parallel** — `tensor_parallel_size=1`, `pipeline_parallel_size=2`. Layers are split across the cards; only activations cross PCIe (once per token), not an all-reduce per layer. Better single-stream latency on this hardware; throughput still scales with concurrency via microbatching. Also added `NCCL_P2P_DISABLE=1` to avoid the known Blackwell P2P deadlocks.

Trade-offs considered: a **14B AWQ on one GPU (TP=1)** would be the fastest single-request option (no cross-GPU traffic at all) if 32B quality isn't required; **TP=2** only wins when saturated with many concurrent requests. Pipeline-parallel 32B was chosen as the balance.

---

## 3c. Tool / function calling (observed 2026-06-04)

Tool calling produced no structured `tool_calls`. Root cause: the start script used
`--tool-call-parser qwen3_coder`, but the served model is the **dense `Qwen3-32B-AWQ`**,
which emits **Hermes-style** `<tool_call>{json}</tool_call>`. The `qwen3_coder` parser
expects the Qwen3-**Coder** XML format, so it can't extract the calls and they fall
through as plain `content`.

**Fix:** parser must match the model. Now parameterized:
```yaml
vllm_enable_tool_calling: true
vllm_tool_call_parser: "hermes"   # Qwen3 dense/instruct, Qwen2.5, QwQ
                                   # → "qwen3_coder" ONLY for Qwen3-Coder models
vllm_reasoning_parser: "qwen3"     # <think> → reasoning_content
```
Emits `--enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3`.
Caveat: hermes + **streaming** tool-call parsing has version-dependent edge cases
(vLLM #31871); use non-streaming for tool calls if a streamed call returns as text.

---

## 3d. Extending context to 64K (observed 2026-06-04)

The Hermes agent needs 64K but vLLM served only 26592. Two independent limits:

1. **KV cache VRAM.** Max context is bounded by KV cache that fits in VRAM, and the
   active sequence's KV *must* be resident in VRAM during attention — it cannot be
   swapped to RAM/HDD to enlarge one request's window. Qwen3-32B KV/token =
   **0.25 MiB (fp16)**. 64K → 16 GiB KV, which won't fit beside ~19 GB weights.
   **Fix: `--kv-cache-dtype fp8`** → 0.125 MiB/token → 8 GiB total (~4 GiB/GPU at
   PP=2). Native on Blackwell, ~nil quality loss. This is the main lever.
2. **Position range.** Qwen3-32B native context is 32768; serving 65536 requires
   **YaRN rope scaling**: `--hf-overrides '{"rope_scaling":{"rope_type":"yarn",
   "factor":2.0,"original_max_position_embeddings":32768}}'` (factor = 65536/32768).
   `--rope-scaling` was superseded by `--hf-overrides` in current vLLM.

**RAM/HDD — what it can and can't do (the user's question):**
- `--cpu-offload-gb N` offloads **N GiB of weights/GPU to CPU RAM**, freeing VRAM for
  KV. This is the legitimate way to use the 64 GB RAM, but weights stream over PCIe
  each forward pass → slow, especially with no-P2P pipeline parallel. Fallback only.
- `--swap-space` / LMCache offload **inactive/preempted** KV (reuse, multi-turn,
  scheduling) — they do **not** raise a single request's max-len. HDD is far too slow
  for KV; NVMe only helps KV *reuse*, not the live window.

Config now: `max_model_len 65536`, `kv_cache_dtype fp8`, YaRN factor 2.0,
`cpu_offload_gb 0` (raise only if fp8+YaRN still OOMs). Per-GPU budget at 64K:
~9.5 GB weights + ~4 GB fp8 KV + overhead ≈ 15 GB of 16 GB — tight; if it OOMs,
set `vllm_cpu_offload_gb: 4`–`8` or `vllm_gpu_memory_utilization: 0.95`.

---

## 4. Model sizing — are the models right?

**Ollama** (`group_vars/all.yml`): `qwen3:32b` ✅ real model. Q4_K_M ≈ **~20 GB** → does not fit one 16 GB card, Ollama auto-splits across both → fits in 32 GB. Good.

**vLLM** (`group_vars/all.yml`): `Qwen/Qwen3-32B-AWQ` ✅ real model. AWQ INT4 weights ≈ **~19 GB**, sharded TP=2 → ~9.5 GB/card, leaving ~4–6 GB/card for KV cache at 32k ctx. **Fits, tight but correct** — and matches your nvidia-smi.

**Wrong / hallucinated references to delete:**
- `Qwen/Qwen3.6-27B-Instruct-AWQ`, `Qwen/Qwen3.6-35B-A3B-Instruct-AWQ`, `qwen3.6:27b` — **no such models** (no "Qwen3.6" line exists). All over `vllm/defaults/main.yml` comments and the README.
- Size claims "~15 GB AWQ" for a 27B — wrong; a 32B AWQ is ~19 GB.

**Recommendation:** standardize on what actually exists and fits 32 GB:
- General: `Qwen/Qwen3-32B-AWQ` (current) — keep.
- Coding alt: `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ` (~19 GB) — real, fits TP=2.
- Smaller/faster: `Qwen/Qwen3-14B-AWQ` or an FP8 14B if you want single-card headroom.

Fix every comment/README line that references the fictional models.

---

## 5. Ollama config fixes

`ollama/tasks/main.yml` override sets two non-existent variables:

```ini
Environment="OLLAMA_GPU_MEMORY_FRACTION=0.90"   # ❌ not a real Ollama var → ignored
Environment="OLLAMA_NUM_CTX=32768"              # ❌ wrong name → context stays 4096
```

**Fix:**
- Drop `OLLAMA_GPU_MEMORY_FRACTION` (Ollama auto-manages VRAM; use `OLLAMA_GPU_OVERHEAD=<bytes>` only if reserving).
- Replace `OLLAMA_NUM_CTX` with **`OLLAMA_CONTEXT_LENGTH`** (the real var). Note even this is reported flaky via `/etc/environment`; setting it in the systemd drop-in (as done here) is the reliable place.
- Keep `OLLAMA_HOST`, `OLLAMA_NUM_PARALLEL`, `OLLAMA_KEEP_ALIVE` — those are valid.

---

## 6. Security (do this regardless of the rest)

1. **Rotate the leaked HF token now** — `hf_VsJ…` in `group_vars/all.yml:53` is in git history. Revoke it at huggingface.co/settings/tokens, then:
   - Move it out of the repo: `ansible-vault` or an env/`--extra-vars @secrets.yml` that is git-ignored.
   - For public Qwen models you don't even need a token — leave it empty.
2. **`ansible_password=tom`** in `inventory.ini` — move to vault or use SSH keys + `--ask-become-pass`.
3. Scrub git history (`git filter-repo`) for both secrets, since the repo already has them committed.

---

## 7. Other correctness fixes

- **`ansible_user_id` → root under `become`.** Replace with the connection user (`ansible_user`, = `tom`) for `docker_users` and `vllm_user`, or define an explicit `service_user: tom`. Otherwise root joins the docker group and vLLM runs as root.
- **NVIDIA driver floor.** Set `nvidia_driver_min_version: 570` and assert the installed driver supports Blackwell; on a fresh box, install an explicit `nvidia-driver-570`+ rather than relying on `ubuntu-drivers install` picking a recent enough branch. (Your box already has 595 — fine — but the role would mis-provision a clean install.)
- **vLLM version probe.** Change `python -m vllm --version` → `vllm --version` (or `{{ vllm_venv_dir }}/bin/vllm --version`).
- **Open WebUI + vLLM.** When `inference_engine == vllm`, also pass an OpenAI connection so the UI sees vLLM:
  ```
  -e OPENAI_API_BASE_URL=http://host.docker.internal:{{ vllm_port }}/v1
  -e OPENAI_API_KEY=dummy
  ```
- **Re-encode `vllm_start.sh.j2` as UTF-8** to remove the `�` mojibake.
- **De-duplicate vLLM vars** between `defaults/main.yml` and `group_vars/all.yml` so there's one source of truth.
- **Summary task template error (observed 2026-06-04).** `site.yml`'s final "Zusammenfassung" task built `msg:` as a YAML **list** with `{% if vllm_enabled %}` in one element and `{% endif %}` in another. Ansible templates each list element independently, so the `if` never finds its `endif` → *"Unexpected end of template."* Fixed by making `msg` a single block-scalar template (balanced control flow) and making it engine-aware (`inference_engine`) instead of always printing "Ollama active".
- **vLLM startup OOM `Free memory … 1.46/15.48 GiB` (observed 2026-06-04).** Not a config bug — ~14 GiB/GPU was already held by a **leftover vLLM** (a still-running manual `vllm serve`, or leaked workers from a crashed start — log showed `destroy_process_group() was not called … leak resources`). vLLM's preflight needs `gpu_memory_utilization × total` free, so it aborts. Fix is operational: stop the service, `pkill -9 -f 'vllm serve'`/`VLLM`, confirm Ollama is inactive, verify `nvidia-smi` shows GPUs near 0, then start — or reboot. Operating rule: never run a manual `vllm serve` *and* the systemd service simultaneously; they compete for the same VRAM.
- **Open WebUI container lifecycle (observed 2026-06-04).** The role gated the `docker run` on `docker inspect open-webui` returning non-zero — but `inspect` succeeds for a container in *any* state (created/exited/dead), so a previously-exited container caused the start task to be skipped forever (`docker ps` empty, WebUI never up). Fixed by always `docker rm -f` + `docker run` (data persists in the mounted volume), and removed the per-run random `WEBUI_SECRET_KEY` (Open WebUI persists its own key in the volume → no logout on redeploy). Also decoupled the systemd unit from `ollama.service` (`Requires=docker.service` only) since Ollama is stopped+disabled when vLLM is the active engine.

---

## 8. Proposed concrete changes (by file)

**`group_vars/all.yml`**
- Add `inference_engine: "vllm"`.
- Remove the HF token (empty or vault).
- `vllm_quantization: "awq_marlin"`, `vllm_attention_backend: "TRITON_ATTN"`, `vllm_max_model_len: 32768`, `vllm_gpu_memory_utilization: "0.92"`.
- Drop `ollama_gpu_memory_fraction`; keep `ollama_num_ctx` but wire it to `OLLAMA_CONTEXT_LENGTH`.
- Fix the fictional-model comments.

**`roles/vllm/defaults/main.yml`**
- `vllm_service_restart_sec: 10` (remove ` I`).
- Add `vllm_attention_backend`, `vllm_enforce_eager`.
- `vllm_user: "{{ ansible_user }}"` (or explicit `tom`).
- Fix hallucinated model comments.

**`roles/vllm/templates/vllm_start.sh.j2`**
- `VLLM_ATTENTION_BACKEND={{ vllm_attention_backend }}` (default TRITON_ATTN); remove FLASH_ATTN + FlashInfer-sampler lines.
- Always emit `--quantization` unless `none`; gate `--enforce-eager` on the var.
- Re-save as UTF-8.

**`roles/vllm/tasks/main.yml`**
- Wrap enable/start in `when: inference_engine == 'vllm'`; add a task to **stop+disable Ollama** in that case.
- `vllm --version` instead of `python -m vllm`.
- Optional: `pip install nvidia-cuda-runtime-cu12` fallback note.

**`roles/ollama/tasks/main.yml`**
- Drop `OLLAMA_GPU_MEMORY_FRACTION`; rename to `OLLAMA_CONTEXT_LENGTH`.
- Gate enable/start on `inference_engine == 'ollama'`; stop+disable vLLM otherwise.

**`roles/docker/tasks/main.yml`** — `docker_users` based on `ansible_user`, not `ansible_user_id`.

**`roles/nvidia/tasks/main.yml`** + `all.yml` — driver floor 570; assert Blackwell-capable.

**`roles/open_webui/tasks/main.yml`** — add vLLM OpenAI connection when vLLM active.

**`inventory.ini`** — remove plaintext password (vault / SSH key).

**`README.md`** — rewrite the model table and the "share 32 GB" narrative to "either/or, full 32 GB each."

---

## 9. Suggested rollout order

1. Rotate HF token + scrub secrets (independent, do first).
2. Fix `RestartSec` typo + attention backend + quantization → get vLLM actually booting.
3. Add `inference_engine` toggle + mutual-exclusion → stop the VRAM fight.
4. Fix Ollama env vars + `ansible_user` + driver floor.
5. Docs/README cleanup.
6. Validate with `ansible-playbook test.yml` (after adding a `vllm --version`/backend check).

### Quick manual smoke test for the vLLM box (before re-running Ansible)
```bash
source /opt/vllm/env/bin/activate
VLLM_ATTENTION_BACKEND=TRITON_ATTN \
VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm serve Qwen/Qwen3-32B-AWQ \
  --quantization awq_marlin \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 \
  --enforce-eager \
  --port 8001
# then: curl http://localhost:8001/v1/models
```
If that serves, the Ansible changes above just encode it.

---

## Sources
- [Field report: AWQ on RTX 5060 Ti (SM_120) — awq_marlin + TRITON_ATTN working](https://discuss.vllm.ai/t/field-report-awq-on-rtx-5060-ti-sm-120-blackwell-awq-marlin-triton-attn-working/2463)
- [vLLM #37714 — Blackwell SM120 + CUDA pip install failures](https://github.com/vllm-project/vllm/issues/37714)
- [vLLM #40677 — FLASHINFER head_size unsupported on SM120](https://github.com/vllm-project/vllm/issues/40677)
- [blackwell-linux-infra-optimizer (SM_120 vLLM deployment notes)](https://github.com/informatico-madrid/blackwell-linux-infra-optimizer)
- [Ollama FAQ — environment variables / OLLAMA_CONTEXT_LENGTH](https://docs.ollama.com/faq)
- [Ollama envconfig package (authoritative var list)](https://pkg.go.dev/github.com/ollama/ollama/envconfig)
- [vLLM — Tool Calling (parser per model: hermes vs qwen3_coder)](https://docs.vllm.ai/en/stable/features/tool_calling/)
- [vLLM #31871 — hermes streaming tool_calls returns raw text](https://github.com/vllm-project/vllm/issues/31871)
- [vLLM — Context Extension (YaRN via hf_overrides)](https://docs.vllm.ai/en/v0.10.2/examples/offline_inference/context_extension.html)
- [vLLM Engine Args — `--cpu-offload-gb` (weights) vs `--swap-space` (KV)](https://docs.vllm.ai/en/v0.8.3/serving/engine_args.html)
- [Qwen/Qwen3-32B — native 32768, YaRN to 131072](https://huggingface.co/Qwen/Qwen3-32B)
