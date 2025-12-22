#!/usr/bin/env python3
"""GPU Generator: ONLY generates completions, writes to pending queue for CPU evaluation.
   Never waits for eval. GPU stays at 100% utilization."""
import os
import sys
import json
import time
import re
import signal
import atexit
import datetime

# Get args: gpu_id start_q end_q n_per_gpu output_dir temps(comma-separated)
gpu_id = int(sys.argv[1])
start_q = int(sys.argv[2])
end_q = int(sys.argv[3])
n = int(sys.argv[4])
output_dir = os.path.abspath(sys.argv[5])
temps = [float(t) for t in sys.argv[6].split(',')]

# Setup directories
pending_eval_dir = f"{output_dir}/pending_eval"
os.makedirs(pending_eval_dir, exist_ok=True)

# Setup LiveCodeBench
os.chdir('/workspace/orkspace/LiveCodeBench')
sys.path.insert(0, '/workspace/orkspace/LiveCodeBench')

from vllm import LLM, SamplingParams
from lcb_runner.benchmarks.code_generation import load_code_generation_dataset

# Global LLM for cleanup
llm = None

def cleanup():
    """Clean up LLM to release GPU memory"""
    global llm
    if llm is not None:
        print(f"[GPU {gpu_id}] Cleaning up LLM...", file=sys.stderr)
        del llm
        llm = None
        # Force CUDA cleanup
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print(f"[GPU {gpu_id}] Received signal {signum}, cleaning up...", file=sys.stderr)
    cleanup()
    sys.exit(0)

# Register cleanup handlers
atexit.register(cleanup)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def extract_code(text):
    pattern = r'```python\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    pattern = r'```\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text.strip()


def format_zero_shot_prompt(problem):
    prompt = "You are an expert Python programmer.\n"
    prompt += "You will be given a programming problem and must generate a correct\n"
    prompt += "Python solution that matches the specification and passes all\n"
    prompt += "tests.\n\n"
    prompt += problem.question_content
    prompt += "\n\n"
    prompt += "Format:\n"
    if problem.starter_code:
        prompt += "You will use the following starter code to write the solution\n"
        prompt += "and enclose your code within backticks.\n\n"
        prompt += "```python\n"
        prompt += problem.starter_code
        prompt += "\n```\n\n"
    else:
        prompt += "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.\n\n"
        prompt += "```python\n"
        prompt += "# YOUR CODE HERE\n"
        prompt += "```\n\n"
    prompt += "Answer:\n\n"
    return prompt


print(f"[GPU {gpu_id}] Loading dataset...", file=sys.stderr)
dataset = load_code_generation_dataset(release_version="v6")

print(f"[GPU {gpu_id}] Loading model ONCE...", file=sys.stderr)
model_load_start = time.time()
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    trust_remote_code=True,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    max_model_len=2182,
    gpu_memory_utilization=0.95,
    dtype="half",
    enforce_eager=False,
    max_num_seqs=512,
    max_num_batched_tokens=32768,
    disable_log_stats=True,
)
model_load_time = time.time() - model_load_start
print(f"[GPU {gpu_id}] Model loaded in {model_load_time:.1f}s - GENERATOR ONLY (no eval wait)!", file=sys.stderr)

total_generated = 0

# Process all questions
for question_idx in range(start_q, end_q):
    if question_idx >= len(dataset):
        continue

    problem = dataset[question_idx]
    prompt = format_zero_shot_prompt(problem)

    print(f"[GPU {gpu_id}] Q{question_idx}: {problem.question_title}", file=sys.stderr)

    for temp in temps:
        gen_start_time = datetime.datetime.now().isoformat()
        gen_start = time.time()

        sampling_params = SamplingParams(
            n=n,
            temperature=temp,
            max_tokens=500,
            stop=["```\n```", "```\n\n", "\n\n\n"],
        )

        outputs = llm.generate([prompt], sampling_params)
        gen_end = time.time()
        gen_time = gen_end - gen_start

        completions = []
        raw_outputs = []
        total_tokens = 0
        for out in outputs[0].outputs:
            raw_text = out.text
            code = extract_code(raw_text)
            completions.append(code)
            raw_outputs.append(raw_text)
            total_tokens += len(out.token_ids)

        gen_tps = total_tokens / gen_time

        # Write to pending_eval queue for CPU evaluator
        pending_file = f"{pending_eval_dir}/q{question_idx}_t{temp:.1f}_gpu{gpu_id}.json"
        pending_data = {
            "question_idx": question_idx,
            "question_title": problem.question_title,
            "temperature": temp,
            "gpu_id": gpu_id,
            "n": n,
            "prompt": prompt,
            "completions": completions,  # Extracted code
            "raw_outputs": raw_outputs,  # Full model output
            "total_tokens": total_tokens,
            "gen_time": gen_time,
            "gen_tps": gen_tps,
            "gen_start_time": gen_start_time,
            "gen_end_time": datetime.datetime.now().isoformat(),
            "model_load_time": model_load_time,
        }

        # Write atomically (write to temp, then rename)
        temp_file = pending_file + ".tmp"
        with open(temp_file, "w") as f:
            json.dump(pending_data, f)
        os.rename(temp_file, pending_file)

        total_generated += 1
        print(f"[GPU {gpu_id}] Q{question_idx} T={temp}: {gen_tps:,.0f} TPS, queued for eval", file=sys.stderr)

# Cleanup
print(f"[GPU {gpu_id}] DONE - generated {total_generated} batches, cleaning up...", file=sys.stderr)
cleanup()
print(f"[GPU {gpu_id}] Exiting cleanly", file=sys.stderr)
