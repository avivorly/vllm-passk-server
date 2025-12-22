#!/usr/bin/env python3
"""Run single question with single temperature"""
import os
import sys
import json
import time
import re

# Setup
os.chdir('/workspace/orkspace/LiveCodeBench')
sys.path.insert(0, '/workspace/orkspace/LiveCodeBench')

from vllm import LLM, SamplingParams
from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

# Config
question_idx = 0
n = 3000
temp = 0.5
MAX_TOKENS = 2048
STOP_WORDS = ["```\n```", "```\n\n", "\n\n\n"]
NUM_CPU_WORKERS = 16
TEST_TIMEOUT = 6

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

print("Loading dataset...")
dataset = load_code_generation_dataset(release_version="v6")
problem = dataset[question_idx]
prompt = format_zero_shot_prompt(problem)
eval_sample = problem.get_evaluation_sample()
print(f"Question {question_idx}: {problem.question_title}")

print("Loading model...")
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    trust_remote_code=True,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    max_model_len=3730,
    gpu_memory_utilization=0.90,
    dtype="half",
    enforce_eager=False,
    max_num_seqs=256,
    max_num_batched_tokens=16384,
    disable_log_stats=True,
)

print(f"Generating n={n} at T={temp}...")
sampling_params = SamplingParams(
    n=n,
    temperature=temp,
    max_tokens=MAX_TOKENS,
    stop=STOP_WORDS,
)

gen_start = time.time()
outputs = llm.generate([prompt], sampling_params)
gen_time = time.time() - gen_start

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
print(f"Generation: {gen_time:.1f}s | {total_tokens:,} tokens | {gen_tps:,.0f} TPS")

print(f"Evaluating {n} completions with {NUM_CPU_WORKERS} CPU workers...")
eval_start = time.time()
metrics, results, _ = codegen_metrics(
    samples_list=[eval_sample],
    generations_list=[completions],
    k_list=[1, 10, 100, 1000],
    num_process_evaluate=NUM_CPU_WORKERS,
    timeout=TEST_TIMEOUT,
    debug=False,
)
eval_time = time.time() - eval_start
eval_tps = n / eval_time

passed_count = sum(1 for r in results[0] if all(x == True for x in r))
pass_rate = passed_count / n * 100

print(f"Evaluation: {eval_time:.1f}s | {eval_tps:,.0f} tests/s")
print(f"Passed: {passed_count}/{n} ({pass_rate:.1f}%)")
print(f"pass@1={metrics['pass@1']*100:.2f}%, pass@10={metrics.get('pass@10', 0)*100:.2f}%, pass@100={metrics.get('pass@100', 0)*100:.2f}%")

# Save results
os.makedirs('/workspace/orkspace/results_test', exist_ok=True)
result = {
    'question_idx': question_idx,
    'question_title': problem.question_title,
    'temperature': temp,
    'n': n,
    'max_tokens': MAX_TOKENS,
    'num_cpu_workers': NUM_CPU_WORKERS,
    'test_timeout': TEST_TIMEOUT,
    'stop_words': STOP_WORDS,
    'total_tokens': total_tokens,
    'gen_time': round(gen_time, 2),
    'gen_tps': round(gen_tps),
    'eval_time': round(eval_time, 2),
    'eval_tps': round(eval_tps),
    'passed': passed_count,
    'pass_rate': round(pass_rate, 2),
    'metrics': metrics,
    'completions': [{'raw': r, 'passed': all(x == True for x in results[0][i])} for i, r in enumerate(raw_outputs)]
}

output_file = '/workspace/orkspace/results_test/q0_t0.5_n3000.json'
with open(output_file, 'w') as f:
    json.dump(result, f, indent=2)
print(f"Saved: {output_file}")
