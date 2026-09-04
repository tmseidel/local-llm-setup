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
    ├── vllm/                         ← vLLM (TP=2, Blackwell-compatible)
    │   ├── defaults/main.yml         ← All vLLM parameters
    │   ├── handlers/main.yml         ← Restart handler (only when active)
    │   ├── tasks/
    │   │   ├── main.yml              ← Installation + engine switching
    │   │   ├── update.yml            ← pip upgrade
    │   │   └── swap_model.yml        ← Switch model without reinstallation
    │   └── templates/
    │       ├── vllm_start.sh.j2      ← Start script (generated from variables)
    │       └── vllm.service.j2       ← systemd unit
    └── resilience/                   ← vLLM + VPN watchdogs (auto-restart on failure)
        ├── defaults/main.yml         ← Watchdog configuration
        ├── tasks/main.yml            ← Deploy watchdog scripts + systemd units
        └── templates/                ← vllm-watchdog.sh, vpn-watchdog.sh

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
  - "qwen3.8:27b"          # Primary model (dense 27B VLM, splits across 2 GPUs)
  # - "qwen3.6:35b-a3b"   # MoE predecessor (optional)
ollama_num_ctx: 32768      # → set as OLLAMA_CONTEXT_LENGTH
ollama_port: 11434
```

### vLLM
```yaml
vllm_enabled: true                  # false = don't install vLLM at all

vllm_model: "philbert440/Qwen3.8-27B-W4A16-AWQ"  # INT4 W4A16, ~19.5 GB, 2 GPUs
vllm_quantization: "none"           # auto-detect from checkpoint (compressed-tensors)
vllm_attention_backend: "TRITON_ATTN"  # Blackwell-compatible (FLASH_ATTN is broken)
vllm_enforce_eager: true            # CUDA graphs off (deadlock fix); MTP needs none
vllm_port: 8001
vllm_max_num_seqs: 4                # full 262K sequences in parallel (more for short)
vllm_gpu_memory_utilization: "0.95" # Full GPU (no co-tenant)
vllm_tensor_parallel_size: 2        # TP over both cards (no NVLink/P2P)
vllm_pipeline_parallel_size: 1
```

> **VRAM note:** Since only one engine runs at a time, the full 32 GB is available
> to it. `Qwen3.8-27B-W4A16` (~19.5 GB weights) is sharded across both RTX 5060 Ti
> via **Tensor Parallelism**; KV cache for 262K context fits comfortably in the
> remaining space thanks to the hybrid architecture (only 16 of 64 layers use full
> attention + GQA → fp8 KV ≈ 8 KiB/token/GPU).

### Context window (262K)

Qwen3.8-27B's **native** context is 262,144 tokens — no YaRN/RoPE scaling needed,
no quality loss. The hybrid architecture makes this cheap: only 16 full-attention
layers × 4 GQA KV-heads → fp8 KV ≈ **16 KiB/token total** (8 KiB/GPU under TP=2).
A full 262K sequence costs ~2.1 GiB KV total, vs ~8 GiB for Qwen3-32B at 64K.

```yaml
vllm_max_model_len: 262144   # native max; >262K (up to 1M) via YaRN override
vllm_kv_cache_dtype: "fp8"   # halves KV VRAM — the main lever
vllm_rope_scaling: ""        # NOT needed for ≤262K (native)
vllm_cpu_offload_gb: 0       # fallback: spill weights to the 64 GB RAM (slow)
```

Going beyond 262K (up to 1M) requires a nested YaRN override (see the comment at
`vllm_rope_scaling` in `group_vars/all.yml`) — but static YaRN slightly hurts
short-context quality, so only enable it when actually needed.

**Using the 64 GB RAM / disk — what actually works:**
- `vllm_cpu_offload_gb: N` moves **N GiB of weights per GPU** into RAM, freeing VRAM
  for more KV. This is the way to "use the RAM", but weights then stream over PCIe
  every forward pass → noticeably slower (worse with no-P2P pipeline parallel). Use
  only if fp8 + YaRN still won't fit, and raise it in small steps (e.g. 4–8).
- `--swap-space` (CPU RAM) and LMCache (RAM/NVMe) offload **inactive/preempted** KV
  for *reuse and scheduling* — they do **not** extend one request's max window.
- **Never** put KV on a spinning HDD; even NVMe is only worthwhile for KV *reuse*,
  not for the live attention window.

> If you run many concurrent 262K requests, lower `vllm_max_num_seqs` (e.g. 2–4) to
> avoid constant preemption — a single 262K sequence needs ~2.1 GiB of fp8 KV.

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

## Resilience (vLLM + VPN Watchdogs)

The `resilience` role deploys two lightweight systemd watchdogs that keep the
stack reachable even after crashes, hangs, or network changes.

| Watchdog | What it checks | Action on failure |
|----------|---------------|-------------------|
| `vllm-watchdog.service` | `http://localhost:8001/health` + `/v1/models` | Restarts `vllm.service` |
| `vpn-watchdog.service` | VPN interface/service + optional remote probe | Reconnects/restarts the detected VPN |

### Supported VPN types

Set `resilience_vpn_type` in `group_vars/all.yml`:

```yaml
resilience_vpn_type: "auto"   # detect WireGuard / OpenVPN / Tailscale / ZeroTier
resilience_vpn_type: "wireguard"
resilience_vpn_type: "openvpn"
resilience_vpn_type: "tailscale"
resilience_vpn_type: "zerotier"
resilience_vpn_type: "custom" # supply custom check/reconnect commands
```

Optional remote probe (only reachable through the VPN):

```yaml
resilience_vpn_remote_probe_host: "my-tailscale-host"
```

### Deploy / reconfigure only the watchdogs

```bash
ansible-playbook -i inventory.ini site.yml --tags resilience --ask-become-pass
```

### Tune thresholds

```yaml
resilience_vllm_check_interval: 30      # seconds between health checks
resilience_vllm_fail_threshold: 10      # consecutive failures before restart (covers model load)
resilience_vllm_max_restarts: 5         # max restarts within restart window
resilience_vllm_restart_window: 3600    # sliding window in seconds

resilience_vpn_check_interval: 60
resilience_vpn_fail_threshold: 2
```

### Manual checks

```bash
# Watchdog status
sudo systemctl status vllm-watchdog vpn-watchdog

# Watch logs
sudo journalctl -u vllm-watchdog -f
sudo journalctl -u vpn-watchdog -f

# Simulate vLLM failure and verify the watchdog restarts it
sudo systemctl stop vllm
# wait resilience_vllm_fail_threshold * resilience_vllm_check_interval seconds
sudo systemctl status vllm
```

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
  -d '{"model":"Qwen3.8-27B-W4A16-AWQ","messages":[{"role":"user","content":"Hello!"}]}'
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

vllm is started with OpenAI-compatible tool calling enabled:

```yaml
vllm_enable_tool_calling: true
vllm_tool_call_parser: "qwen3_coder"  # Qwen3.5/3.6/3.8 (all variants) — official
vllm_reasoning_parser: "qwen3"         # splits <think>…</think> into reasoning_content
```

> **The parser must match the model's tool-call format.** Qwen3.8-27B (like all
> Qwen3.5/3.6 models) uses the `qwen3_coder` format. `hermes` is only correct for
> the older dense Qwen3 models (`Qwen/Qwen3-32B-AWQ`) and Qwen2.5. A mismatched
> parser is the usual reason tool calls are returned as plain text instead of
> structured `tool_calls`. `--reasoning-parser qwen3` is mandatory in practice:
> the chat template opens every assistant turn with `<think>`, so without it the
> reasoning block lands in `content` and eats the token budget.

Test it:

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.8-27B-W4A16-AWQ",
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
vllm_model: "philbert440/Qwen3.8-27B-W4A16-AWQ"  # ~19.5 GB — all-round (default), quantization: "none"
vllm_model: "Qwen/Qwen3.8-27B-FP8"               # ~28 GB — official FP8, quantization: "none"
vllm_model: "Qwen/Qwen3-14B-AWQ"                 # ~10 GB — faster, quantization: "awq_marlin"
vllm_model: "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"  # ~20 GB — MoE predecessor, quantization: "awq_marlin"
```

> **Quantization flag:** compressed-tensors / FP8 checkpoints auto-detect →
> `vllm_quantization: "none"`. Only classic AWQ repos (`Qwen/Qwen3-*-AWQ`) need
> `"awq_marlin"`. A wrong flag breaks loading.

### Hermes harness config hint

Point Hermes at the vLLM endpoint as an OpenAI-compatible provider (example for
`~/.hermes/config.yaml`; key name is `api_base` or `base_url` depending on your
Hermes version):

```yaml
provider: openai-api
model: Qwen3.8-27B-W4A16-AWQ        # = --served-model-name (repo basename)
api_base: http://localhost:8001/v1  # or the LAN host, e.g. http://172.30.0.15:8001/v1
api_key: EMPTY                      # vllm_api_key is empty (no auth on LAN)
```

Recommended request-side settings for this model (official Qwen sampling):
thinking mode `temperature 1.0, top_p 0.95, top_k 20`; non-thinking
`temperature 0.7, top_p 0.80, presence_penalty 1.5`. Thinking is ON by default —
per request `"chat_template_kwargs": {"enable_thinking": false}` disables it,
`{"reasoning_effort": "low"}` makes it adaptive. Give agentic tasks generous
`max_tokens` — the model expects room to reason before answering.
