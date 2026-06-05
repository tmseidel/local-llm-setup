# local-llm-setup

Ansible automation for a local LLM stack (**Ollama** or **vLLM** + **Open WebUI**)
on Ubuntu 24.04 LTS.

**Target hardware:** 2× NVIDIA RTX 5060 Ti 16 GB (= **32 GB VRAM**, Blackwell / sm_120) ·
64 GB RAM · AMD Ryzen 9 9950X.

## Design Principle

Only **one** inference engine runs at a time. Controlled via `inference_engine`
(`vllm` | `ollama`) in `ansible/group_vars/all.yml` — the other one is
stopped + disabled, so the active engine uses the **full 32 GB VRAM**.

- **vLLM** — parallel agent workloads, tensor parallelism across both GPUs,
  AWQ (`awq_marlin`), Blackwell-compatible (`VLLM_ATTENTION_BACKEND=TRITON_ATTN`).
- **Ollama** — single requests, easy model switching, Open WebUI backend.

## Quick Start

```bash
cd ansible
# Choose engine in group_vars/all.yml, then:
ansible-playbook -i inventory.ini site.yml --ask-become-pass
ansible-playbook -i inventory.ini test.yml
```

## Documentation

| File | Content |
|-------|--------|
| [`ansible/README.md`](ansible/README.md) | Installation, configuration, tags, engine switching |
| [`doc/analysis-and-fix-plan.md`](doc/analysis-and-fix-plan.md) | Analysis, Blackwell support, VRAM, design decisions |

## Security

- `ansible/group_vars/all.yml` no longer contains a HuggingFace token (not needed
  for the public Qwen models). For gated models, pass via Ansible Vault or
  `-e`, **never** commit.
- `ansible/inventory.ini` currently uses a plaintext password — switch to SSH keys
  or Ansible Vault.
