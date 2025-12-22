#!/usr/bin/env python3
"""GPU worker that ONLY generates - no evaluation. Returns immediately."""
import os
import sys
import json
import time
import re

# Get args
gpu_id = int(sys.argv[1])
question_idx = int(sys.argv[2])
n = int(sys.argv[3])
temp = float(sys.argv[4])
output_file = sys.argv[5]

# Setup LiveCodeBench
os.chdir('/workspace/orkspace/LiveCodeBench')
sys.path.insert(0, '/workspace/orkspace/LiveCodeBench')

from vllm import LLM, SamplingParams
from lcb_runner.benchmarks.code_generation import load_code_generation_dataset


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
problem = dataset[question_idx]
prompt = format_zero_shot_prompt(problem)

print(f"[GPU {gpu_id}] Loading model...", file=sys.stderr)
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    trust_remote_code=True,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    max_model_len=2182,
    gpu_memory_utilization=0.90,
    dtype="half",
    enforce_eager=False,
    max_num_seqs=256,
    max_num_batched_tokens=16384,
    disable_log_stats=True,
)

import datetime
start_time = datetime.datetime.now().isoformat()

print(f"[GPU {gpu_id}] Generating n={n} at T={temp}...", file=sys.stderr)
sampling_params = SamplingParams(
    n=n,
    temperature=temp,
    max_tokens=500,
    stop=["```\n```", "```\n\n", "\n\n\n"],
)

gen_start = time.time()
outputs = llm.generate([prompt], sampling_params)
gen_time = time.time() - gen_start

# Extract completions - NO EVALUATION HERE
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
print(f"[GPU {gpu_id}] Generated {total_tokens:,} tokens in {gen_time:.1f}s = {gen_tps:,.0f} TPS", file=sys.stderr)

end_time = datetime.datetime.now().isoformat()

# Save generation results ONLY - evaluation happens separately
result = {
    "start_time": start_time,
    "end_time": end_time,
    "model": "Qwen/Qwen3-0.6B",
    "prompt": prompt,
    "sampling_params": {
        "temperature": temp,
        "max_tokens": 500,
        "stop": ["```\n```", "```\n\n", "\n\n\n"],
    },
    "model_config": {
        "max_model_len": 2182,
        "gpu_memory_utilization": 0.90,
        "dtype": "half",
        "max_num_seqs": 256,
        "max_num_batched_tokens": 16384,
    },
    "question_idx": question_idx,
    "question_title": problem.question_title,
    "gpu_id": gpu_id,
    "n": n,
    "temperature": temp,
    "total_tokens": total_tokens,
    "gen_time": gen_time,
    "gen_tps": gen_tps,
    "raw_outputs": raw_outputs,
    "completions": completions,
    "evaluated": False,  # Flag: needs evaluation
}

with open(output_file, "w") as f:
    json.dump(result, f)

print(f"[GPU {gpu_id}] Done! Saved to {output_file}", file=sys.stderr)
