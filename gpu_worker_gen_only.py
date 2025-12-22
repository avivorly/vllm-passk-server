#!/usr/bin/env python3
"""GPU worker: ONLY generates completions, NO evaluation (that's done by combiner)"""
import os
import sys
import json
import time
import datetime

# Args: gpu_id start_q end_q n_per_gpu output_dir temps
gpu_id = int(sys.argv[1])
start_q = int(sys.argv[2])
end_q = int(sys.argv[3])
n = int(sys.argv[4])
output_dir = os.path.abspath(sys.argv[5])
temps = [float(t) for t in sys.argv[6].split(',')]

# Setup
os.chdir('/workspace/orkspace/LiveCodeBench')
sys.path.insert(0, '/workspace/orkspace/LiveCodeBench')

from vllm import LLM, SamplingParams
from lcb_runner.benchmarks.code_generation import load_code_generation_dataset

def format_zero_shot_prompt(problem):
    prompt = "You are an expert Python programmer.\n"
    prompt += "You will be given a programming problem and must generate a correct\n"
    prompt += "Python solution that matches the specification and passes all\n"
    prompt += "tests.\n\n"
    prompt += problem.question_content + "\n\n"
    prompt += "Format:\n"
    if problem.starter_code:
        prompt += "You will use the following starter code to write the solution\n"
        prompt += "and enclose your code within backticks.\n\n"
        prompt += f"```python\n{problem.starter_code}\n```\n\n"
    else:
        prompt += "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.\n\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
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
    max_num_seqs=512,
    max_num_batched_tokens=32768,
    disable_log_stats=True,
)
model_load_time = time.time() - model_load_start
print(f"[GPU {gpu_id}] Model loaded in {model_load_time:.1f}s", file=sys.stderr)

for question_idx in range(start_q, end_q):
    if question_idx >= len(dataset):
        continue

    problem = dataset[question_idx]
    prompt = format_zero_shot_prompt(problem)

    q_dir = f"{output_dir}/q{question_idx}"
    os.makedirs(q_dir, exist_ok=True)

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
        gen_time = time.time() - gen_start
        gen_end_time = datetime.datetime.now().isoformat()

        # Save raw outputs (no code extraction, no evaluation)
        raw_outputs = []
        total_tokens = 0
        for out in outputs[0].outputs:
            raw_outputs.append(out.text)
            total_tokens += len(out.token_ids)

        gen_tps = total_tokens / gen_time
        print(f"[GPU {gpu_id}] Q{question_idx} T={temp}: {gen_tps:,.0f} TPS", file=sys.stderr)

        result = {
            "status": "generated",  # Mark as needing evaluation
            "gpu_id": gpu_id,
            "question_idx": question_idx,
            "question_title": problem.question_title,
            "temperature": temp,
            "n": n,
            "prompt": prompt,
            "total_tokens": total_tokens,
            "gen_time": gen_time,
            "gen_tps": gen_tps,
            "timing": {
                "gen_start": gen_start_time,
                "gen_end": gen_end_time,
            },
            "raw_outputs": raw_outputs,  # Raw model outputs, not processed
        }

        temp_str = f"{temp:.1f}".replace('.', '_')
        output_file = f"{q_dir}/gpu{gpu_id}_t{temp_str}_raw.json"
        with open(output_file, "w") as f:
            json.dump(result, f)

print(f"[GPU {gpu_id}] DONE generating all questions!", file=sys.stderr)
