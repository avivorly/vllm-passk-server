#!/usr/bin/env python3
"""Single GPU worker that processes ALL temperatures with ONE model load"""
import os
import sys
import json
import time
import re
import datetime

# Get args: gpu_id question_idx n output_dir temps(comma-separated)
gpu_id = int(sys.argv[1])
question_idx = int(sys.argv[2])
n = int(sys.argv[3])
output_dir = sys.argv[4]
temps = [float(t) for t in sys.argv[5].split(',')]

# Setup LiveCodeBench
os.chdir('/workspace/orkspace/LiveCodeBench')
sys.path.insert(0, '/workspace/orkspace/LiveCodeBench')

from vllm import LLM, SamplingParams
from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics


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
eval_sample = problem.get_evaluation_sample()

print(f"[GPU {gpu_id}] Loading model ONCE for all {len(temps)} temperatures...", file=sys.stderr)
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
print(f"[GPU {gpu_id}] Model loaded in {model_load_time:.1f}s", file=sys.stderr)

os.makedirs(output_dir, exist_ok=True)
all_results = []

for temp in temps:
    # Overall timing
    overall_start_time = datetime.datetime.now().isoformat()
    overall_start = time.time()

    # Generation timing
    gen_start_time = datetime.datetime.now().isoformat()
    gen_start = time.time()

    print(f"[GPU {gpu_id}] T={temp}: Generating n={n}...", file=sys.stderr)
    sampling_params = SamplingParams(
        n=n,
        temperature=temp,
        max_tokens=500,
        stop=["```\n```", "```\n\n", "\n\n\n"],
    )

    outputs = llm.generate([prompt], sampling_params)
    gen_end = time.time()
    gen_end_time = datetime.datetime.now().isoformat()
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
    print(f"[GPU {gpu_id}] T={temp}: {total_tokens:,} tokens in {gen_time:.1f}s = {gen_tps:,.0f} TPS", file=sys.stderr)

    # Evaluation timing
    eval_start_time = datetime.datetime.now().isoformat()
    eval_start = time.time()

    print(f"[GPU {gpu_id}] T={temp}: Evaluating...", file=sys.stderr)
    metrics, results, _ = codegen_metrics(
        samples_list=[eval_sample],
        generations_list=[completions],
        k_list=[1],
        num_process_evaluate=24,
        timeout=6,
        debug=False,
    )

    eval_end = time.time()
    eval_end_time = datetime.datetime.now().isoformat()
    eval_time = eval_end - eval_start
    eval_tps = n / eval_time

    passed_count = sum(1 for r in results[0] if all(x == True for x in r))
    pass_list = [all(x == True for x in r) for r in results[0]]
    print(f"[GPU {gpu_id}] T={temp}: Passed {passed_count}/{n} | Eval: {eval_tps:,.0f} tests/s", file=sys.stderr)

    # Overall timing
    overall_end = time.time()
    overall_end_time = datetime.datetime.now().isoformat()
    total_time = overall_end - overall_start
    pipeline_tps = n / total_time

    completions_with_status = [
        {"raw": raw, "passed": passed}
        for raw, passed in zip(raw_outputs, pass_list)
    ]

    result = {
        # Detailed timing info
        "timing": {
            "generation": {
                "start_time": gen_start_time,
                "end_time": gen_end_time,
                "duration_seconds": round(gen_time, 3),
            },
            "evaluation": {
                "start_time": eval_start_time,
                "end_time": eval_end_time,
                "duration_seconds": round(eval_time, 3),
            },
            "overall": {
                "start_time": overall_start_time,
                "end_time": overall_end_time,
                "duration_seconds": round(total_time, 3),
            },
            "model_load_seconds": round(model_load_time, 3),
        },
        # Reproducibility params
        "model": "Qwen/Qwen3-0.6B",
        "prompt": prompt,
        "sampling_params": {
            "temperature": temp,
            "max_tokens": 500,
            "stop": ["```\n```", "```\n\n", "\n\n\n"],
        },
        "model_config": {
            "max_model_len": 2182,
            "gpu_memory_utilization": 0.95,
            "dtype": "half",
            "max_num_seqs": 512,
            "max_num_batched_tokens": 32768,
        },
        # Question info
        "question_idx": question_idx,
        "question_title": problem.question_title,
        # Run info
        "gpu_id": gpu_id,
        "temperature": temp,
        "n": n,
        "total_tokens": total_tokens,
        "gen_time": gen_time,
        "gen_tps": gen_tps,
        "eval_time": eval_time,
        "eval_tps": eval_tps,
        "total_time": total_time,
        "pipeline_tps": pipeline_tps,
        "passed": passed_count,
        "completions": completions_with_status,
    }

    # Save each temp result
    temp_str = f"{temp:.1f}".replace('.', '_')
    output_file = f"{output_dir}/gpu{gpu_id}_t{temp_str}.json"
    with open(output_file, "w") as f:
        json.dump(result, f)

    all_results.append({
        "temperature": temp,
        "passed": passed_count,
        "gen_tps": gen_tps,
        "eval_tps": eval_tps,
    })

# Save summary for this GPU
summary_file = f"{output_dir}/gpu{gpu_id}_summary.json"
with open(summary_file, "w") as f:
    json.dump({"gpu_id": gpu_id, "question_idx": question_idx, "n": n, "results": all_results}, f)

print(f"[GPU {gpu_id}] All {len(temps)} temperatures done!", file=sys.stderr)
