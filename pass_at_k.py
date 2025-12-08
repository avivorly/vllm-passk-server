"""
Pass@k experiment: Generate 1000 completions for the same prompt using vLLM.
Optimized for maximum throughput with:
- n=1000 for single prompt (leverages prefix caching automatically)
- Large batch processing
- Prefix caching enabled
"""

import time
from vllm import LLM, SamplingParams

# The prompt to ask 1000 times
PROMPT = '''from typing import List


def filter_by_substring(strings: List[str], substring: str) -> List[str]:
    """ Filter an input list of strings only for ones that contain given substring
    >>> filter_by_substring([], 'a')
    []
    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')
    ['abc', 'bacd', 'array']
    """
'''

def main():
    print("=" * 60)
    print("Pass@k Experiment: 1000 completions for code generation")
    print("Model: Qwen/Qwen2.5-0.5B")
    print("=" * 60)

    # Initialize vLLM with optimizations
    print("\nLoading model...")
    load_start = time.time()

    llm = LLM(
        model="Qwen/Qwen2.5-0.5B",
        trust_remote_code=True,
        enable_prefix_caching=True,  # Reuse KV cache for identical prefix
        max_model_len=2200,          # Full context for 2000 tokens
        gpu_memory_utilization=0.95, # Safe GPU memory
        dtype="half",                # FP16 for speed
        enforce_eager=False,         # Use CUDA graphs
        max_num_seqs=1000,           # High concurrency
        max_num_batched_tokens=32768, # Large batch token budget
        disable_log_stats=True,      # Reduce overhead
    )

    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    # Sampling parameters optimized for pass@k
    # Using n=1000 generates 1000 different completions for the SAME prompt
    # This is the most efficient way as prefix KV cache is shared
    sampling_params = SamplingParams(
        n=1000,              # Generate 1000 completions
        temperature=2.0,     # High temperature for diversity (pass@k)
        max_tokens=2000,     # Full 2000 tokens as required
        stop=["\ndef", "\nclass", "\n\n\n"],  # Stop at next function/class
    )

    print(f"\nGenerating 1000 completions with n={sampling_params.n}...")
    print(f"Max tokens per completion: {sampling_params.max_tokens}")
    print(f"Temperature: {sampling_params.temperature}")

    # Generate - single prompt with n=1000
    gen_start = time.time()
    outputs = llm.generate([PROMPT], sampling_params)
    gen_time = time.time() - gen_start

    # Collect results
    completions = []
    total_tokens = 0

    for output in outputs:
        for completion in output.outputs:
            completions.append(completion.text)
            total_tokens += len(completion.token_ids)

    # Statistics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total completions generated: {len(completions)}")
    print(f"Total output tokens: {total_tokens:,}")
    print(f"Generation time: {gen_time:.2f}s")
    print(f"Tokens per second: {total_tokens / gen_time:,.2f}")
    print(f"Completions per second: {len(completions) / gen_time:.2f}")

    # Show a few sample completions
    print("\n" + "=" * 60)
    print("SAMPLE COMPLETIONS (first 3):")
    print("=" * 60)
    for i, completion in enumerate(completions[:3]):
        print(f"\n--- Completion {i+1} ---")
        print(completion[:500] + "..." if len(completion) > 500 else completion)

    # Return stats for further analysis
    return {
        "total_completions": len(completions),
        "total_tokens": total_tokens,
        "generation_time_seconds": gen_time,
        "tokens_per_second": total_tokens / gen_time,
        "completions_per_second": len(completions) / gen_time,
        "completions": completions,
    }

if __name__ == "__main__":
    results = main()
