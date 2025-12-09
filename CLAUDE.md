# vLLM Pass@K Server

High-throughput code generation and evaluation server for HumanEval benchmarks using vLLM.

## Purpose

The main purpose of this project is to run `benchmark_server.py` in a loop to evaluate code generation models on HumanEval problems. The server keeps the model loaded in memory, eliminating model loading overhead between benchmark runs.

**Core workflow:**
1. Start `vllm_server.py` (loads model once)
2. Run `benchmark_server.py` in a loop for different questions/temperatures/n values
3. Each run generates completions, tests them, and saves results to JSON

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

## Throughput Benchmarks

### Raw Generation TPS (max_tokens=100, 3 stop sequences)
| n (completions) | TPS | Avg tokens/completion |
|-----------------|-----|----------------------|
| 10 | ~1.9K | 62 |
| 100 | ~10.5K | 59 |
| 1000 | ~35K | 59 |
| 10000 | ~35.8K | 59 |

### Evalplus-style Generation (max_tokens=768, 11 stop sequences)
| n (completions) | TPS | Avg tokens/completion |
|-----------------|-----|----------------------|
| 100 | ~4.8K | 76 |
| 1000 | ~16.5K | 76 |

Higher n = better throughput due to prefix caching (all completions share the same prompt KV cache).

**Note**: TPS varies with completion length. Shorter completions (max_tokens=100) achieve higher TPS than longer ones (max_tokens=768).

### Full Pipeline: Generation + Testing (n=1000, 164 HumanEval+ problems)
| Phase | Time | TPS |
|-------|------|-----|
| Generation | 4.69s | 16,536 TPS |
| Testing (1 worker) | 77.6s | 999 Test TPS |
| Testing (16 workers) | 11.3s | 6,886 Test TPS |

**Parallel Testing Sweet Spot**: 16 workers provides 6.9x speedup on 32-core AMD Ryzen 9 9950X.

Test TPS = total generated tokens / testing time (measures tokens verified per second).

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

## Evalplus Integration

For HumanEval+ evaluation with the optimized vLLM config:

```python
from evalplus.data import get_human_eval_plus
from evalplus.evaluate import check_correctness, get_groundtruth, PASS
from evalplus.gen.util import trusted_exec
from evalplus.gen.util.api_request import extra_eos_for_direct_completion
from evalplus.data.utils import EOS
from vllm import LLM, SamplingParams
from concurrent.futures import ProcessPoolExecutor

# Get evalplus stop sequences
STOP = list(EOS) + extra_eos_for_direct_completion('humaneval')

# Generate with optimized config
params = SamplingParams(n=1000, temperature=0.8, max_tokens=768, top_p=0.95, stop=STOP)
outputs = llm.generate([prompt], params)

# Parallel testing (16 workers optimal)
with ProcessPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(check_correctness, ...) for ...]
```

## Single Question Benchmark (Main Script)

Use `benchmark_server.py` to benchmark a single HumanEval question using the running server:

```bash
# Start the server first (one time)
python3 vllm_server.py &

# Run benchmarks in a loop
for q in {0..163}; do
  python3 benchmark_server.py --question $q --temperature 0.5 --n 1000
done
```

**Arguments:**
- `--question, -q` - Question index (0-163)
- `--temperature, -t` - Sampling temperature (default: 0.8)
- `--n` - Number of completions (default: 1000)
- `--output, -o` - Output file (default: `results_q{q}_t{t}_n{n}.json`)
- `--server, -s` - Server URL (default: `http://localhost:8000`)

**Output JSON structure:**
```json
{
  "task_id": "HumanEval/0",
  "question_idx": 0,
  "temperature": 0.5,
  "n": 1000,
  "total_tokens": 43798,
  "gen_time": 2.16,
  "test_time": 4.09,
  "pipeline_time": 6.25,
  "total_time": 9.08,
  "gen_tps": 20316,
  "test_tps": 10712,
  "pipeline_tps": 7014,
  "passed": 309,
  "pass_rate": 30.9,
  "completions": [
    {"text": "...", "tokens": 45, "passed": true},
    ...
  ]
}
```

**TPS Metrics:**
- **Gen TPS**: Tokens generated per second (generation phase only)
- **Test TPS**: Tokens verified per second (testing phase only)
- **Pipeline TPS**: Tokens / (gen_time + test_time) - the practical throughput

### Benchmark Results (HumanEval/0, n=1000, T=0.5)

| Metric | Value |
|--------|-------|
| Gen TPS | 20,316 |
| Test TPS | 10,712 |
| Pipeline TPS | 7,014 |
| Pass Rate | 30.9% |
| Total Time | 9.08s |

### Temperature Effect on Pass Rate

| Temperature | Gen TPS | Test TPS | Pass Rate |
|-------------|---------|----------|-----------|
| 0.5 | ~20K | ~10K | ~31% |
| 1.0 | ~20K | ~7.5K | ~8.5% |

Lower temperature = higher pass rate (more deterministic outputs).

## Other Scripts

- `benchmark_question.py` - Standalone benchmark (loads model each time, slower)
- `bigcode_benchmark.py` - Full HumanEval benchmark (all 164 problems)
- `vllm_final.py` - Pure throughput benchmark
- `evaluate_passk.py` - Correctness evaluation
