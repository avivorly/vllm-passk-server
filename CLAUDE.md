# LiveCodeBench Temperature Sweep Benchmark

High-throughput pass@k evaluation on LiveCodeBench v6 using vLLM with 8x RTX 4090.

---

## Quick Start: Run the Full Benchmark

```bash
cd /workspace/orkspace && source /workspace/venv/bin/activate && \
export HF_HOME=/workspace/.cache/huggingface && \
python3 run_benchmark_optimal.py --start 0 --end 175 --n 10000 --output_dir results_n10k --num_evaluators 16
```

This runs **175 questions × 16 temperatures × 10,000 completions** = 28 million total generations.

---

## Architecture: Optimal Pipeline (ALWAYS USE THIS)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          TWO-STAGE PENDING QUEUE                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  GPU Generators (8x)         CPU Evaluators (16x)        Combiner Thread     │
│  ──────────────────          ──────────────────────      ────────────────    │
│  gpu_generator.py            cpu_evaluator.py            (in orchestrator)   │
│                                                                              │
│  [GPU 0] ───┐                ┌─── [EVAL 0]  ───┐         Watches             │
│  [GPU 1] ───┤                ├─── [EVAL 1]  ───┤         pending_combine/    │
│  [GPU 2] ───┤                ├─── [EVAL 2]  ───┤                             │
│  [GPU 3] ───┼──► pending_eval/ ──► ...       ──┼──► pending_combine/ ──►     │
│  [GPU 4] ───┤                ├─── [EVAL 13] ───┤         Combines when       │
│  [GPU 5] ───┤                ├─── [EVAL 14] ───┤         8 GPU files ready   │
│  [GPU 6] ───┤                └─── [EVAL 15] ───┘                             │
│  [GPU 7] ───┘                                            ▼                   │
│                                                                              │
│  Outputs:                    Outputs:                    Final Output:       │
│  q{idx}_t{temp}_gpu{id}.json q{idx}_t{temp}_gpu{id}.json q{idx}/             │
│  (1,250 completions each)    (1,250 + pass/fail)         ├── t0_1_combined   │
│                                                          ├── t0_2_combined   │
│                                                          ├── ...             │
│                                                          └── summary.json    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

| Principle | Implementation |
|-----------|----------------|
| **GPU never waits for CPU** | Separate processes; GPU writes to queue and moves on |
| **Model loaded ONCE per GPU** | Each GPU processes ALL 175×16 questions/temps |
| **16 evaluators for throughput** | CPU eval is slower than GPU gen; more parallelism needed |
| **Two-stage queue** | `pending_eval/` → `pending_combine/` prevents race conditions |
| **Atomic file writes** | Write to `.tmp`, rename to final - no partial reads |
| **File claiming via rename** | Evaluators claim work by renaming to `.processing{id}` |
| **Auto summary creation** | When all 16 temps complete, `summary.json` is created |

### Key Files

| File | Purpose |
|------|---------|
| `run_benchmark_optimal.py` | **Orchestrator** - launches all workers + combiner thread |
| `gpu_generator.py` | **GPU worker** - generates completions, writes to `pending_eval/` |
| `cpu_evaluator.py` | **CPU worker** - evaluates code, writes to `pending_combine/` |

---

## Monitor Progress

```bash
# GPU generation progress (8 logs)
tail -f results_n10k/gpu*.log

# CPU evaluation progress (16 logs)
tail -f results_n10k/eval*.log

# Main benchmark log (combiner output, completion messages)
tail -f benchmark.log

# Queue sizes (should stay low if evaluators keep up)
echo "Pending eval: $(ls results_n10k/pending_eval/ 2>/dev/null | wc -l)"
echo "Pending combine: $(ls results_n10k/pending_combine/ 2>/dev/null | wc -l)"

# GPU utilization (should be ~80-100% during generation)
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

# Completed questions
ls -d results_n10k/q*/summary.json 2>/dev/null | wc -l
```

---

## Output Structure

```
results_n10k/
├── pending_eval/              # Stage 1: GPU → CPU eval queue
│   └── (files claimed quickly by evaluators)
├── pending_combine/           # Stage 2: Evaluated → Combiner queue
│   └── (files combined when all 8 GPUs done for a temp)
├── q0/
│   ├── t0_1_combined.json     # 10,000 completions at T=0.1
│   ├── t0_2_combined.json     # 10,000 completions at T=0.2
│   ├── ...
│   ├── t2_0_combined.json     # 10,000 completions at T=2.0
│   └── summary.json           # Pass rates for all 16 temperatures
├── q1/
│   └── ...
├── gpu0.log ... gpu7.log      # Per-GPU generation logs
├── eval0.log ... eval15.log   # Per-evaluator logs
└── run_summary.json           # Global run metadata
```

### Summary JSON Format (created when all 16 temps complete)

```json
{
  "question_idx": 0,
  "question_title": "Problem Title",
  "n": 10000,
  "temperatures": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.5, 1.7, 2.0],
  "results": [
    {"temperature": 0.1, "n": 10000, "passed": 0, "pass_rate": 0.0, "gen_tps": 117655},
    {"temperature": 0.2, "n": 10000, "passed": 0, "pass_rate": 0.0, "gen_tps": 102792},
    ...
    {"temperature": 1.2, "n": 10000, "passed": 171, "pass_rate": 1.71, "gen_tps": 97450},
    ...
  ]
}
```

---

## Stopping the Benchmark Safely

**Zombie GPU memory = stuck until pod restart.** Follow this procedure EXACTLY.

### Correct: Kill GPU generators FIRST

```bash
# Step 1: Kill GPU generators (they hold GPU memory)
pkill -TERM -f gpu_generator
echo "Waiting for GPU memory to release..."

# Step 2: Poll until GPU memory is freed (MUST show ~2 MiB per GPU)
while true; do
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    # If all show "2 MiB", proceed. If not, keep waiting!
    sleep 5
done

# Step 3: Only AFTER GPU memory is clear, kill evaluators and orchestrator
pkill -TERM -f cpu_evaluator
pkill -TERM -f run_benchmark
```

### Why This Order Matters

| Action | Result |
|--------|--------|
| Kill generators first, wait for memory clear, then rest | Clean |
| Kill orchestrator first | **ZOMBIE** (orphans GPU processes) |
| `kill -9` on GPU generators | **ZOMBIE** (bypasses cleanup handlers) |
| Interrupt during model loading | **ZOMBIE** |
| `killall python` | **ZOMBIE** |

### If Zombies Exist

```bash
# Check for zombie memory (PIDs show [Not Found])
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

# If you see "[Not Found]" with memory allocated:
# ONLY OPTION: Restart the pod. No way to clear zombie GPU memory.
```

---

## Configuration Parameters

### Temperatures (16 total)
```python
TEMPERATURES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.5, 1.7, 2.0]
```

### vLLM Settings (in gpu_generator.py)
```python
LLM(
    model="Qwen/Qwen3-0.6B",
    trust_remote_code=True,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    max_model_len=2182,            # 500 (max_tokens) + 1682 (longest prompt)
    gpu_memory_utilization=0.95,
    dtype="half",
    max_num_seqs=512,
    max_num_batched_tokens=32768,
    disable_log_stats=True,
)
```

### Scaling Parameters (in run_benchmark_optimal.py)
```python
NUM_GPUS = 8                # Number of GPU generators
NUM_EVALUATORS = 16         # Number of CPU evaluators (increase if queue grows)
STAGGER_DELAY = 0.5         # Seconds between GPU launches (prevents vLLM init race)
MAX_RESTARTS = 3            # Auto-restart crashed workers
```

---

## Common Issues and Fixes

### 1. Pending Queue Growing (Files Piling Up)

**Symptom:** `ls pending_eval/ | wc -l` shows increasing numbers

**Cause:** CPU evaluators can't keep up with GPU generation

**Fix:** Increase `--num_evaluators`:
```bash
python3 run_benchmark_optimal.py ... --num_evaluators 24
```

### 2. Model Reloading Per Temperature

**Symptom:** ~15-20s delay every temperature

**Cause:** Using wrong script (old `benchmark_lcb.py`)

**Fix:** Use `run_benchmark_optimal.py` which loads model ONCE per GPU

### 3. Low GPU Utilization (~30%)

**Symptom:** `nvidia-smi` shows low GPU%

**Cause:** Likely using tensor parallelism

**Fix:** Use data parallelism (8 separate GPU processes). NEVER use `tensor_parallel_size` on RTX 4090s (no NVLink).

### 4. vLLM Init Race Condition

**Symptom:** Multiple GPUs crash during startup

**Cause:** vLLM model loading race when launched simultaneously

**Fix:** Already handled - `STAGGER_DELAY = 0.5` staggers GPU launches

### 5. Evaluator Claiming Same File

**Symptom:** Errors about file not found, duplicate processing

**Cause:** Race condition in file claiming

**Fix:** Already handled - files renamed to `.processing{id}.json` when claimed

---

## Performance Expectations

| Metric | Expected Value |
|--------|----------------|
| Generation TPS (per GPU) | ~95,000-105,000 |
| Total Generation TPS (8 GPUs) | ~750,000-800,000 |
| Questions per hour | ~3-4 (with 16 temps × 10k completions) |
| Full benchmark (175 questions) | ~50-60 hours |

---

## Environment

- **GPUs:** 8x NVIDIA RTX 4090 (24GB VRAM each)
- **Platform:** RunPod
- **Python:** 3.11+ with vLLM 0.13.0
- **Storage:** `/workspace` mount (~137TB) - ALWAYS use this, never `/root` or `~`

### Required Environment Setup
```bash
source /workspace/venv/bin/activate
export HF_HOME=/workspace/.cache/huggingface
```

---

## DO NOT

1. **DO NOT** use `tensor_parallel_size` - RTX 4090s have no NVLink
2. **DO NOT** reload model per temperature - use `run_benchmark_optimal.py`
3. **DO NOT** kill orchestrator first - kills GPU generators first instead
4. **DO NOT** use `kill -9` on GPU processes - bypasses cleanup handlers
5. **DO NOT** write to `/root` or `~` - use `/workspace` (137TB mount)
6. **DO NOT** use fewer than 16 evaluators for n=10000 - queue will grow

---

## File Naming Convention

### Pending Files
- `pending_eval/q{idx}_t{temp}_gpu{id}.json` - Generated, awaiting evaluation
- `pending_eval/q{idx}_t{temp}_gpu{id}.processing{eval_id}.json` - Being evaluated
- `pending_combine/q{idx}_t{major}_{minor}_gpu{id}.json` - Evaluated, awaiting combine

### Final Files
- `q{idx}/t{major}_{minor}_combined.json` - Combined 10k completions
- `q{idx}/summary.json` - All 16 temperatures aggregated

Temperature encoding: `t{major}_{minor}` where `0.1` → `t0_1`, `1.2` → `t1_2`, `2.0` → `t2_0`
