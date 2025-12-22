# LiveCodeBench Temperature Sweep Benchmark

High-throughput pass@k evaluation on LiveCodeBench v6 using vLLM.

---

## 🚀 CURRENT RUN STATUS

**Benchmark is RUNNING in background.** Check progress:
```bash
tail -20 /workspace/orkspace/benchmark_fast_mt500.log
```

**If benchmark stopped/crashed, resume from where it left off:**
```bash
# Find last completed question
ls /workspace/orkspace/results_fast_mt500/ | grep "^q" | sed 's/q//' | sort -n | tail -1

# Resume from next question (replace START with last+1)
cd /workspace/orkspace
HF_HOME=/workspace/.cache/huggingface nohup python3 run_full_benchmark.py \
    --start START --end 175 --n 3000 \
    --output_dir results_fast_mt500 \
    > benchmark_fast_mt500.log 2>&1 &
```

**Key files:**
- `run_full_benchmark.py` - Main orchestrator (spawns GPU workers)
- `gpu_worker.py` - Single GPU worker (generation + evaluation)
- `results_fast_mt500/q{N}/` - Results per question

**GPU Status:**
- GPUs 2-7: Working (6 GPUs)
- GPUs 0-1: Zombie memory stuck (~22GB each) - cannot clear without pod restart
- Config in `run_full_benchmark.py`: `NUM_GPUS=6`, `GPU_OFFSET=2`

---

## ⚠️ CRITICAL: LOAD MODEL ONCE PER GPU FOR ENTIRE BENCHMARK ⚠️

**NEVER reload the model between questions or temperatures!** Model loading wastes ~15-20 seconds each time.

| ❌ WRONG | ✅ CORRECT |
|----------|-----------|
| Reload model per temperature | Model loaded ONCE per GPU |
| Reload model per question | Same model processes ALL questions |
| 175 questions × 16 temps × 8 GPUs = 22,400 loads | 8 loads total (one per GPU) |

**Correct approach:** Use `run_benchmark_optimal.py` which launches 8 persistent workers. Each GPU loads the model ONCE and processes ALL 175 questions × 16 temperatures.

```bash
# OPTIMAL (model loaded ONCE per GPU for entire run)
python3 run_benchmark_optimal.py --start 0 --end 175 --n 3000 --output_dir results

# Monitor progress
tail -f results/gpu*.log
nvidia-smi
```

**Scripts hierarchy (fastest to slowest):**
1. `run_benchmark_optimal.py` + `gpu_worker_full.py` - ONE model load per GPU (USE THIS)
2. `run_benchmark_fast.py` + `gpu_worker_multi_temp.py` - One load per question (slower)
3. `run_full_benchmark.py` + `gpu_worker.py` - One load per temperature (DON'T USE)

---

## ⚠️ CRITICAL: DO NOT USE TENSOR PARALLELISM ⚠️

**NEVER use `tensor_parallel_size` in vLLM on this system!**

| ❌ WRONG | ✅ CORRECT |
|----------|-----------|
| `LLM(model=..., tensor_parallel_size=8)` | Run 8 separate `gpu_worker.py` instances |
| ~2,000 TPS (10x slower!) | ~60,000+ TPS |

**Why:** RTX 4090s have NO NVLink. Tensor parallelism forces inter-GPU communication over slow PCIe, killing performance.

**Correct approach:** Data parallelism - run `run_full_benchmark.py` which spawns independent workers per GPU.

---

## ⚠️ CRITICAL: USE CORRECT STORAGE MOUNT ⚠️

**ALWAYS use `/workspace` for all data storage!**

| ❌ WRONG | ✅ CORRECT |
|----------|-----------|
| Default HF cache (`~/.cache/huggingface`) | `HF_HOME=/workspace/.cache/huggingface` |
| Writing to `/root`, `/home`, `/tmp` | Writing to `/workspace/...` |
| ~50GB overlay filesystem | ~137TB mounted storage |

**Before running ANY command, set:**
```bash
export HF_HOME=/workspace/.cache/huggingface
```

The root filesystem is a tiny overlay. `/workspace` is the large persistent storage mount.

---

## Quick Start: Run the Temperature Sweep

```bash
# Set CUDA library path (required)
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cusparselt/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"

# Run full benchmark (175 questions x 16 temperatures x 2500 completions)
python3 benchmark_lcb.py --start 0 --end 175 --n 2500 --output_dir results_temps
```

This runs in the background and saves results to `results_temps/q{idx}/summary.json`.

## Current Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen3-0.6B |
| Dataset | LiveCodeBench v6 (175 questions) |
| Completions per temp | 3000 |
| Max tokens | 500 |
| max_model_len | 2182 (500 + longest prompt 1682) |
| Prompt mode | Zero-shot |
| Stop words | `["```\n```", "```\n\n", "\n\n\n"]` |
| num_process_evaluate | 24 (CPU workers for test evaluation) |
| Output folder | `results_fast_mt500/` |

**Temperatures:** `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.5, 1.7, 2.0]`

## Environment

- **GPU**: 8x NVIDIA RTX 4090 (24GB VRAM each) - RunPod
- **Python**: 3.11+
- **Key packages**: vLLM 0.13.0, PyTorch with CUDA 12.4
- **Storage**: `/workspace` mount (~137TB)

## Multi-GPU Parallelization (8x RTX 4090)

### Why Data Parallelism, NOT Tensor Parallelism

For small models like Qwen3-0.6B, **use data parallelism** (4 separate instances) instead of tensor parallelism:

| Approach | Performance | Why |
|----------|-------------|-----|
| **Data Parallel (4 instances)** | ~19,000 TPS | Each GPU runs full model independently |
| Tensor Parallel (split model) | ~2,000 TPS | PCIe overhead kills performance |

**RTX 4090s lack NVLink** - all inter-GPU communication goes over PCIe, making tensor parallelism extremely inefficient for small models.

### Quick Start: 4-GPU Parallel Benchmark

```bash
# Run 4 workers in parallel (n=5000 total, 1250 per GPU)
for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu python3 gpu_worker.py $gpu 0 1250 0.8 /tmp/gpu$gpu.json &
    sleep 5  # Stagger starts to avoid init conflicts
done
wait

# Combine results
python3 -c "
import json
results = [json.load(open(f'/tmp/gpu{i}.json')) for i in range(4)]
total_tokens = sum(r['total_tokens'] for r in results)
max_time = max(r['gen_time'] for r in results)
print(f'Combined TPS: {total_tokens/max_time:,.0f}')
print(f'Per-GPU: ' + ', '.join(f\"GPU{r['gpu_id']}={r['gen_tps']:,.0f}\" for r in results))
"
```

### gpu_worker.py Usage

```bash
# Arguments: gpu_id question_idx n temperature output_file
CUDA_VISIBLE_DEVICES=0 python3 gpu_worker.py 0 5 1250 0.8 results/gpu0.json
```

### Performance Benchmarks

| Config | TPS | Notes |
|--------|-----|-------|
| 1x RTX 4090 | ~5,100 | Single GPU baseline |
| 4x RTX 4090 (data parallel) | ~19,300 | Near-linear scaling |
| 4x RTX 4090 (tensor parallel) | ~2,000 | Avoid - PCIe bottleneck |

### Optimal vLLM Config for Multi-GPU

```python
# Per-GPU instance (used in gpu_worker.py)
LLM(
    model="Qwen/Qwen3-0.6B",
    trust_remote_code=True,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    max_model_len=3730,           # Exact max needed (see below)
    gpu_memory_utilization=0.95,  # Max out each GPU
    dtype="half",
    max_num_seqs=512,
    max_num_batched_tokens=32768,
    disable_log_stats=True,
)
```

Key settings for throughput:
- `max_model_len=3730` - longest prompt (1682) + max output (2048) = 3730. Smaller = more KV cache = faster
- `gpu_memory_utilization=0.95` - maximize KV cache (small model leaves plenty of room)
- `max_num_seqs=512` - high concurrent sequences
- `max_num_batched_tokens=32768` - large batches for throughput (default 2048 is latency-optimized)

### Token Budget Analysis

| Metric | Value |
|--------|-------|
| Total questions | 175 |
| Longest prompt | Q85: 1,682 tokens |
| Shortest prompt | Q70: 210 tokens |
| Avg prompt | 561 tokens |
| Max output tokens | 2,048 |
| **max_model_len needed** | **3,730** (1682 + 2048) |

**Why 3730 matters:** Reducing from 4096 to 3730 increases KV cache by 12.7% and max concurrency by 23%:
- KV cache: 141K → 160K tokens (+12.7%)
- Max concurrency: 34.6x → 42.7x (+23%)

## Command Reference

```bash
# Single question, all temperatures
python3 benchmark_lcb.py --question 5 --n 2500 --output_dir results

# Range of questions
python3 benchmark_lcb.py --start 10 --end 50 --n 2500 --output_dir results

# Skip greedy-passing questions (faster)
python3 benchmark_lcb.py --start 0 --end 175 --n 2500 --output_dir results

# Run all temps even for easy questions
python3 benchmark_lcb.py --start 0 --end 175 --n 2500 --output_dir results --no-skip
```

## Output Structure

```
results_temps/
├── run_summary.json           # Global run info
├── q0/
│   ├── t0_0_n2500.json       # Results at T=0.1
│   ├── t0_1_n2500.json       # Results at T=0.1
│   ├── ...
│   └── summary.json          # Pass rates across all temps
├── q1/
│   └── skipped.json          # Easy question (passed greedy)
└── ...
```

**Temperature result JSON format** (e.g., `t0_5_n2500.json`):
```json
{
  "model": "Qwen/Qwen3-0.6B",
  "prompt": "You are an expert Python programmer...\n<full prompt text>",
  "sampling_params": {
    "temperature": 0.5,
    "max_tokens": 2048,
    "stop": ["```\n```", "```\n\n", "\n\n\n"]
  },
  "model_config": {
    "max_model_len": 3730,
    "gpu_memory_utilization": 0.95,
    "dtype": "half",
    "max_num_seqs": 512,
    "max_num_batched_tokens": 32768
  },
  "question_idx": 13,
  "question_title": "9x9",
  "temperature": 0.5,
  "n": 2500,
  "passed": 1110,
  "pass_rate": 44.4,
  "gen_tps": 6035,
  "eval_tps": 150,
  "pipeline_tps": 45,
  "completions": [
    {"raw": "<full model output>", "passed": true},
    {"raw": "<full model output>", "passed": false},
    ...
  ]
}
```

**Summary JSON format** (clean, no completions):
```json
{
  "question_idx": 13,
  "question_title": "9x9",
  "n": 2500,
  "temperatures": [0.1, 0.2, 0.3, ...],
  "total_time": 2104.5,
  "results": [
    {"temperature": 0.5, "n": 2500, "passed": 1110, "pass_rate": 44.4, "gen_tps": 6035},
    ...
  ]
}
```

## Visualizing Results with Plotext

Terminal-based plotting for quick analysis:

```bash
pip install plotext
```

```python
import plotext as plt
import json

# Load results
with open('results_temps/q13/summary.json') as f:
    data = json.load(f)

# Extract data
temps = [r['temperature'] for r in data['results']]
pass_rates = [r['pass_rate'] for r in data['results']]

# Plot pass rate vs temperature
plt.clear_figure()
plt.plot(temps, pass_rates, marker='braille')
plt.title(f"Q{data['question_idx']}: {data['question_title']}")
plt.xlabel("Temperature")
plt.ylabel("Pass Rate (%)")
plt.show()
```

**Example output:**
```
        Q13: 9x9
   45 ┤    ╭──╮
   40 ┤   ╭╯  ╰─╮
   35 ┤  ╭╯     ╰╮
   30 ┤ ╭╯       ╰─╮
   25 ┤╭╯          ╰╮
   20 ┤│            ╰╮
   15 ┤│             │
   10 ┼╯             ╰╮
    5 ┤               │
    0 ┤               ╰─
      └──────────────────
       0.0  0.5  1.0  1.5
         Temperature
```

## vLLM Configuration (Single GPU)

Optimized for single RTX 4090 with high batch throughput:

```python
LLM(
    model="Qwen/Qwen3-0.6B",
    trust_remote_code=True,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    max_model_len=3730,  # Exact max: longest prompt (1682) + max output (2048)
    gpu_memory_utilization=0.95,
    dtype="half",
    enforce_eager=False,
    max_num_seqs=512,
    max_num_batched_tokens=32768,
    disable_log_stats=True,
)
```

For multi-GPU, see [Multi-GPU Parallelization](#multi-gpu-parallelization-4x-rtx-4090) above.

## Reproducibility

Every result JSON contains full parameters for exact reproduction:
- Complete prompt text
- All sampling parameters
- Model configuration
- Stop words used
- Raw completions with pass/fail status

## Dependencies

The benchmark uses LiveCodeBench (cloned to `./LiveCodeBench`):

```bash
# Clone LiveCodeBench if not present
git clone https://github.com/LiveCodeBench/LiveCodeBench.git

# Install with older datasets version (required for loading script compatibility)
pip install -e ./LiveCodeBench
pip install 'datasets<3.0.0'  # Required for HuggingFace loading script
```

```python
from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
```
