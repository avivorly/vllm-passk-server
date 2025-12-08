# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **GPU**: NVIDIA RTX 4090 (24GB VRAM, ~1 TB/s memory bandwidth)
- **Python**: 3.12
- **Key packages**: vLLM 0.12.0, PyTorch with CUDA 12.8

## Quick Start: Running the Server

```bash
# 1. Set CUDA library path (required)
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cusparselt/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"

# 2. Start the vLLM server
python3 vllm_server.py

# 3. In another terminal, expose via SSH tunnel for remote access
bash tunnel.sh
# Tunnel URL appears in /tmp/tunnel.log
```

## Remote Batch Processing

From your local machine, use `server_script.py` to process questions:

```bash
python3 server_script.py \
  --input_file questions.jsonl \
  --output_file results.jsonl \
  --n 100 \
  --temperature 0.8 \
  --max_tokens 500 \
  --top_p 1.0 \
  --stop "\ndef" "\nclass" "\n\n\n" \
  --server_url https://<tunnel-url>/generate
```

**Input format** (questions.jsonl):
```json
{"question": "def add(a, b):\n    "}
```

**Output format** (results.jsonl):
```json
{"question": "...", "completions": ["return a + b", ...], "total_tokens": 1500}
```

## API Endpoints

- `POST /generate` - Generate n completions (params: prompt, n, temperature, max_tokens, top_p, stop)
- `GET /health` - Server status
- `GET /` - Hello world

## Throughput

| n (completions) | TPS |
|-----------------|-----|
| 100 | ~8K |
| 500 | ~22K |
| 1000 | ~27K |
| 2000 | ~30K |

Higher n = better throughput due to prefix caching (all completions share the same prompt KV cache).

## vLLM Server Configuration

The server uses Qwen/Qwen2.5-0.5B-Instruct with these optimizations:

```python
LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    max_model_len=2048,
    gpu_memory_utilization=0.95,
    dtype="half",
    enforce_eager=False,
    max_num_seqs=2500,
    max_num_batched_tokens=150000,
)
```

## Important Notes

**max_model_len must accommodate your prompts + max_tokens:**
```python
# Ensure: max_model_len >= prompt_length + max_tokens
# Server default: max_model_len=2048, plenty of room for max_tokens=500
```

**Stop sequences** for clean code completions: `["\ndef", "\nclass", "\n\n\n"]`

## Other Scripts (for local benchmarking)

- `vllm_final.py` - Pure throughput benchmark
- `evaluate_passk.py` - Correctness evaluation (94% pass@2500)
