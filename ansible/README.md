# 🤖 Local LLM Stack — Ansible

Automated installation of **Ollama + vLLM + Open WebUI** on Ubuntu 24.04 LTS.
Optimized for: **2× RTX 5060 Ti 16 GB (= 32 GB VRAM, Blackwell sm_120)** · 64 GB RAM · AMD Ryzen 9 9950X

> **Important:** Only **one** inference engine (Ollama **or** vLLM) runs at a time,
> controlled via `inference_engine` in `group_vars/all.yml`. The other one is
> stopped + disabled, so the active engine uses the **full 32 GB VRAM**.

---

## Architecture

Only **one** engine runs at a time — controlled via `inference_engine`. The other
is stopped + disabled, so the active engine gets the full GPU.

```
        Clients (curl / Python / Open WebUI / LangChain / n8n …)
                              │
            inference_engine: │ vllm                  ollama │
                              ▼                              ▼
                        vLLM :8001                     Ollama :11434
                  ──────────────────────         ──────────────────────
                  Parallel Agents               Single requests
                  PagedAttention / Batching      Model switching, Web UI
                  AWQ (awq_marlin), PP=2          GGUF / Q4_K_M, Auto-Split
                              │                              │
                              └──────────────┬───────────────┘
                                             ▼
                     2× RTX 5060 Ti — 32 GB VRAM (Blackwell sm_120)
                     → the ACTIVE engine uses the full 32 GB
```

Open WebUI (:8080) automatically points to the active engine.

---

## Quick Start

```bash
# 1. Install Ansible
sudo apt install -y ansible

# 2. Change to the Ansible directory
cd ansible

# 3. Choose active engine in group_vars/all.yml (inference_engine: vllm | ollama)
#    and install the stack
ansible-playbook -i inventory.ini site.yml --ask-pass --ask-become-pass

# 4. Test after completion
ansible-playbook -i inventory.ini test.yml
```

Total duration: **30–90 minutes** (mainly model downloads).

> A detailed analysis and rationale for all design decisions
> (Blackwell support, VRAM, engine switching) can be found in
> [`../doc/analysis-and-fix-plan.md`](../doc/analysis-and-fix-plan.md).

---

## Project Structure

```
ansible/
├── site.yml                          ← Main playbook
├── update.yml                        ← Update all components
├── test.yml                          ← Smoke test (active engine + WebUI)
├── inventory.ini                     ← Target host
├── group_vars/
│   └── all.yml                       ← Configuration ← ADJUST here
└── roles/
    ├── nvidia/                       ← NVIDIA drivers, CUDA, container toolkit
    ├── docker/                       ← Docker CE + NVIDIA Container Runtime
    ├── ollama/                       ← Ollama + models (tasks/pull_models.yml)
    ├── open_webui/                   ← Web interface (Docker), points to active engine
    └── vllm/                         ← vLLM (TP=2, Blackwell-compatible)
        ├── defaults/main.yml         ← All vLLM parameters
        ├── handlers/main.yml         ← Restart handler (only when active)
        ├── tasks/
        │   ├── main.yml              ← Installation + engine switching
        │   ├── update.yml            ← pip upgrade
        │   └── swap_model.yml        ← Switch model without reinstallation
        └── templates/
            ├── vllm_start.sh.j2      ← Start script (generated from variables)
            └── vllm.service.j2       ← systemd unit

../doc/analysis-and-fix-plan.md       ← Analysis & rationale for the configuration
```

---

## Configuration (group_vars/all.yml)

### Engine Selection
```yaml
inference_engine: "vllm"    # "vllm" | "ollama" — only ONE runs at a time
```

### Ollama
```yaml
ollama_models:
  - "qwen3:32b"            # Primary model (~20 GB, Q4_K_M, splits across 2 GPUs)
  # - "qwen2.5-coder:32b"  # Coding (optional)
ollama_num_ctx: 32768      # → set as OLLAMA_CONTEXT_LENGTH
ollama_port: 11434
```

### vLLM
```yaml
vllm_enabled: true                  # false = don't install vLLM at all

vllm_model: "Qwen/Qwen3-32B-AWQ"    # AWQ INT4, ~19 GB, distributed across 2 GPUs
vllm_quantization: "awq_marlin"     # awq_marlin | gptq_marlin | fp8 | none
vllm_attention_backend: "TRITON_ATTN"  # Blackwell-compatible (FLASH_ATTN is broken)
vllm_enforce_eager: false           # CUDA graphs on (faster); true = fallback
vllm_port: 8001
vllm_max_num_seqs: 32               # Parallel sequences — important for agents
vllm_gpu_memory_utilization: "0.92" # Full GPU (no co-tenant)
# No NVLink/P2P → Pipeline instead of Tensor Parallelism (faster, less comms)
vllm_tensor_parallel_size: 1
vllm_pipeline_parallel_size: 2
```

> **VRAM note:** Since only one engine runs at a time, the full 32 GB is available
> to it. `Qwen3-32B-AWQ` (~19 GB weights) is distributed across both RTX 5060 Ti
> via **Pipeline Parallelism** (layer split); KV cache for 32k context fits in
> the remaining space. Pipeline instead of Tensor Parallel because GeForce cards
> lack NVLink and PCIe P2P is disabled — Tensor Parallelism would be
> communication-bound.

### Context window (64K)

The max context is bounded by **KV cache that fits in VRAM** — and the active
sequence's KV cache *must* be in VRAM during attention, so it cannot be moved to
RAM/HDD to enlarge a single request's window. Two VRAM-side levers get a 32B model
to 64K on 2×16 GB:

```yaml
vllm_max_model_len: 65536
vllm_kv_cache_dtype: "fp8"   # halves KV VRAM (0.25→0.125 MiB/token) — main lever
vllm_rope_scaling: '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}'
                             # required: Qwen3 native context is 32768
vllm_cpu_offload_gb: 0       # fallback: spill weights to the 64 GB RAM (slow)
```

KV-cache math (Qwen3-32B): **fp16 = 0.25 MiB/token, fp8 = 0.125 MiB/token**.
64K context → fp16 ≈ **16 GiB** KV (won't fit with 19 GB weights), fp8 ≈ **8 GiB**
(~4 GiB/GPU under PP=2) → fits.

**Using the 64 GB RAM / disk — what actually works:**
- `vllm_cpu_offload_gb: N` moves **N GiB of weights per GPU** into RAM, freeing VRAM
  for more KV. This is the way to "use the RAM", but weights then stream over PCIe
  every forward pass → noticeably slower (worse with no-P2P pipeline parallel). Use
  only if fp8 + YaRN still won't fit, and raise it in small steps (e.g. 4–8).
- `--swap-space` (CPU RAM) and LMCache (RAM/NVMe) offload **inactive/preempted** KV
  for *reuse and scheduling* — they do **not** extend one request's max window.
- **Never** put KV on a spinning HDD; even NVMe is only worthwhile for KV *reuse*,
  not for the live attention window.

> If you run many concurrent 64K requests, lower `vllm_max_num_seqs` (e.g. 4–8) to
> avoid constant preemption — a single 64K sequence already needs ~8 GiB of fp8 KV.

---

## Running Individual Tags

```bash
# NVIDIA drivers only
ansible-playbook -i inventory.ini site.yml --tags nvidia --ask-become-pass

# Ollama only (without model download)
ansible-playbook -i inventory.ini site.yml --tags ollama --ask-become-pass

# Open WebUI only
ansible-playbook -i inventory.ini site.yml --tags webui --ask-become-pass

# vLLM only
ansible-playbook -i inventory.ini site.yml --tags vllm --ask-become-pass

# Download models only
ansible-playbook -i inventory.ini site.yml --tags models --ask-become-pass

# Switch vLLM to a different model without reinstalling
ansible-playbook -i inventory.ini site.yml --tags vllm_swap \
  -e "vllm_model=Qwen/Qwen2.5-Coder-32B-Instruct-AWQ" --ask-become-pass

# After reboot (driver install): skip NVIDIA
ansible-playbook -i inventory.ini site.yml --skip-tags nvidia --ask-become-pass
```

### Switching Engines (Ollama ↔ vLLM)

The active engine is controlled via `inference_engine`. When switching, the other
one is automatically stopped + disabled, so the active engine has the full GPU.

```bash
# Switch to Ollama (stops vLLM, starts Ollama)
ansible-playbook -i inventory.ini site.yml \
  -e "inference_engine=ollama" --ask-become-pass

# Switch back to vLLM
ansible-playbook -i inventory.ini site.yml \
  -e "inference_engine=vllm" --ask-become-pass
```

> **Note:** Ollama models are only pulled (`--tags models`) when
> `inference_engine=ollama` — the pull requires a running Ollama service.
> `vllm_enabled=false` completely skips the vLLM **installation** (independent
> of `inference_engine`).

---

## After Installation

Depending on `inference_engine`, **either** the Ollama **or** the vLLM endpoint
is active (not both at the same time). Open WebUI always runs.

| Service | URL | Active when | Purpose |
|--------|-----|-----------|-------|
| Open WebUI | http://localhost:8080 | always | Chat interface |
| Ollama API | http://localhost:11434/v1 | `inference_engine: ollama` | Single requests, UI |
| vLLM API | http://localhost:8001/v1 | `inference_engine: vllm` | Agents, parallel calls |

### Ollama — API Test
```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3:32b","messages":[{"role":"user","content":"Hello!"}]}'
```

### vLLM — API Test
```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-32B-AWQ","messages":[{"role":"user","content":"Hello!"}]}'
```

### Python (OpenAI SDK) — Active Engine Endpoint
```python
from openai import OpenAI

# For inference_engine: ollama
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# For inference_engine: vllm
client = OpenAI(base_url="http://localhost:8001/v1",  api_key="vllm")
```

---

## Tool / Function Calling (vLLM)

vLLM is started with OpenAI-compatible tool calling enabled:

```yaml
vllm_enable_tool_calling: true
vllm_tool_call_parser: "hermes"   # Qwen3 dense/instruct, Qwen2.5, QwQ
vllm_reasoning_parser: "qwen3"     # splits <think>…</think> into reasoning_content
```

> **The parser must match the model's tool-call format.** `Qwen/Qwen3-32B-AWQ`
> (dense) emits Hermes-style `<tool_call>{…}</tool_call>` → use **`hermes`**.
> Only the **Qwen3-Coder** models use `qwen3_coder`. A mismatched parser is the
> usual reason tool calls are returned as plain text instead of structured
> `tool_calls`.

Test it:

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-32B-AWQ",
    "messages": [{"role":"user","content":"What is the weather in Berlin?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "auto"
  }'
```

A working response contains `choices[0].message.tool_calls` (with the thinking
text separated into `reasoning_content`), not a tool call embedded in `content`.

> **Note (streaming + reasoning):** with the `hermes` parser, tool-call parsing in
> **streaming** mode has had edge cases across vLLM versions. If a streamed
> `tool_calls` arrives as raw text, use non-streaming for tool-calling requests or
> upgrade vLLM (`update.yml --tags vllm`).

---

## Reboot After Driver Installation

If Ansible reports that a reboot is needed after NVIDIA driver installation:

```bash
sudo reboot

# Then continue the playbook (skip NVIDIA role)
ansible-playbook -i inventory.ini site.yml --skip-tags nvidia --ask-become-pass
```

---

## Logs & Diagnostics

```bash
# Ollama
journalctl -u ollama -f

# vLLM
journalctl -u vllm -f
# or directly:
tail -f /var/log/vllm/vllm.log

# Open WebUI
docker logs -f open-webui

# Live GPU utilization
nvtop
# or
watch -n1 nvidia-smi
```

---

## More vLLM Models (real, fit in 32 GB with TP=2)

```yaml
# In group_vars/all.yml or via -e flag:
vllm_model: "Qwen/Qwen3-32B-AWQ"                   # ~19 GB — all-round (default)
vllm_model: "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"  # ~19 GB — coding
vllm_model: "Qwen/Qwen3-14B-AWQ"                   # ~10 GB — faster
vllm_model: "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"  # ~28 GB BF16 — reasoning
```
