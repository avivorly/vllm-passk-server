"""Parallel benchmark using all GPUs independently (data parallelism at process level)"""
import os
import sys
import json
import time
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_on_gpu(gpu_id, question_idx, n_per_gpu, temp, output_file):
    """Run benchmark on a specific GPU"""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # Isolate multiprocessing for each subprocess
    env['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    # Use different ports for each GPU to avoid conflicts
    env['VLLM_PORT'] = str(29500 + gpu_id)

    cmd = [
        'python3', '-c', f'''
import os
import sys
import json
import time

os.chdir('/workspace/LiveCodeBench')
sys.path.insert(0, '/workspace/LiveCodeBench')

from vllm import LLM, SamplingParams
from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
import re

def extract_code(text):
    pattern = r"```python\\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    pattern = r"```\\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text.strip()

def format_zero_shot_prompt(problem):
    prompt = "You are an expert Python programmer.\\n"
    prompt += "You will be given a programming problem and must generate a correct\\n"
    prompt += "Python solution that matches the specification and passes all\\n"
    prompt += "tests.\\n\\n"
    prompt += problem.question_content
    prompt += "\\n\\n"
    prompt += "Format:\\n"
    if problem.starter_code:
        prompt += "You will use the following starter code to write the solution\\n"
        prompt += "and enclose your code within backticks.\\n\\n"
        prompt += "```python\\n"
        prompt += problem.starter_code
        prompt += "\\n```\\n\\n"
    else:
        prompt += "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.\\n\\n"
        prompt += "```python\\n"
        prompt += "# YOUR CODE HERE\\n"
        prompt += "```\\n\\n"
    prompt += "Answer:\\n\\n"
    return prompt

# Load dataset
dataset = load_code_generation_dataset(release_version="v6")
problem = dataset[{question_idx}]
prompt = format_zero_shot_prompt(problem)
eval_sample = problem.get_evaluation_sample()

# Load model
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    trust_remote_code=True,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    max_model_len=4096,
    gpu_memory_utilization=0.95,
    dtype="half",
    enforce_eager=False,
    max_num_seqs=512,
    max_num_batched_tokens=32768,
    disable_log_stats=True,
)

# Generate
sampling_params = SamplingParams(
    n={n_per_gpu},
    temperature={temp},
    max_tokens=2048,
    stop=["```\\n```", "```\\n\\n", "\\n\\n\\n"],
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

# Evaluate
eval_start = time.time()
metrics, results, _ = codegen_metrics(
    samples_list=[eval_sample],
    generations_list=[completions],
    k_list=[1],
    num_process_evaluate=16,
    timeout=6,
    debug=False,
)
eval_time = time.time() - eval_start

passed_count = sum(1 for r in results[0] if all(x == True for x in r))
pass_list = [all(x == True for x in r) for r in results[0]]

result = {{
    "gpu_id": {gpu_id},
    "n": {n_per_gpu},
    "total_tokens": total_tokens,
    "gen_time": gen_time,
    "gen_tps": gen_tps,
    "eval_time": eval_time,
    "passed": passed_count,
    "pass_list": pass_list,
    "raw_outputs": raw_outputs,
}}

with open("{output_file}", "w") as f:
    json.dump(result, f)
'''
    ]

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"GPU {gpu_id} error: {result.stderr[-500:]}")
        return None
    return output_file


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', '-q', type=int, default=0)
    parser.add_argument('--n', type=int, default=5000)
    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--gpus', type=int, default=4)
    args = parser.parse_args()

    n_per_gpu = args.n // args.gpus
    remainder = args.n % args.gpus

    print(f"Running {args.n} completions across {args.gpus} GPUs ({n_per_gpu} each)")

    # Create temp files for results
    output_files = [tempfile.mktemp(suffix='.json') for _ in range(args.gpus)]

    # Launch parallel processes with staggered start to avoid init conflicts
    start = time.time()
    processes = []
    for gpu_id in range(args.gpus):
        n_this_gpu = n_per_gpu + (1 if gpu_id < remainder else 0)
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

        cmd = ['python3', '/workspace/gpu_worker.py',
               str(gpu_id), str(args.question), str(n_this_gpu), str(args.temp), output_files[gpu_id]]
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(p)
        time.sleep(2)  # Stagger starts

    # Wait for all to complete
    for i, p in enumerate(processes):
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            print(f"GPU {i} error: {stderr.decode()[-500:]}")

    total_time = time.time() - start

    # Combine results
    all_results = []
    total_tokens = 0
    total_gen_time = 0
    total_passed = 0

    for output_file in output_files:
        if os.path.exists(output_file):
            with open(output_file) as f:
                data = json.load(f)
            all_results.append(data)
            total_tokens += data['total_tokens']
            total_gen_time = max(total_gen_time, data['gen_time'])  # parallel, so take max
            total_passed += data['passed']
            os.remove(output_file)

    combined_tps = total_tokens / total_gen_time
    pass_rate = total_passed / args.n * 100

    print(f"\n{'='*60}")
    print(f"COMBINED RESULTS (4 GPUs parallel)")
    print(f"{'='*60}")
    print(f"Total completions: {args.n}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Generation time (wall): {total_gen_time:.1f}s")
    print(f"Combined TPS: {combined_tps:,.0f}")
    print(f"Passed: {total_passed}/{args.n} ({pass_rate:.1f}%)")
    print(f"Total wall time: {total_time:.1f}s")

    # Per-GPU stats
    print(f"\nPer-GPU breakdown:")
    for r in all_results:
        print(f"  GPU {r['gpu_id']}: {r['gen_tps']:,.0f} TPS, {r['passed']}/{r['n']} passed")


if __name__ == "__main__":
    main()
