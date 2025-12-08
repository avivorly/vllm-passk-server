"""vLLM final push - all optimizations"""
import time

if __name__ == '__main__':
    from vllm import LLM, SamplingParams

    PROMPT = '''from typing import List


def filter_by_substring(strings: List[str], substring: str) -> List[str]:
    """ Filter an input list of strings only for ones that contain given substring
    >>> filter_by_substring([], 'a')
    []
    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')
    ['abc', 'bacd', 'array']
    """
'''

    print("Loading model with all optimizations...")
    llm = LLM(
        model="Qwen/Qwen2.5-0.5B",
        trust_remote_code=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,    # Explicit chunked prefill
        max_model_len=600,  # prompt(87) + max_tokens(500) + buffer
        gpu_memory_utilization=0.95,
        dtype="half",
        enforce_eager=False,
        max_num_seqs=2500,
        max_num_batched_tokens=150000,
        disable_log_stats=True,
    )

    print("Running 3 tests with n=2500...\n")

    results = []
    for i in range(3):
        sampling_params = SamplingParams(
            n=2500,
            temperature=0.8,
            max_tokens=500,
            stop=["\ndef", "\nclass", "\n\n\n"],
        )

        start = time.time()
        outputs = llm.generate([PROMPT], sampling_params)
        elapsed = time.time() - start

        completions = outputs[0].outputs
        total_tokens = sum(len(c.token_ids) for c in completions)

        tps = total_tokens / elapsed
        crs = len(completions) / elapsed
        results.append(tps)

        print(f"Run {i+1}: {crs:.0f} CRS | {tps:.0f} TPS")

    print(f"\n>>> AVERAGE: {sum(results)/len(results):.0f} TPS <<<")
    print(f">>> MAX: {max(results):.0f} TPS <<<")
