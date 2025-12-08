"""Show full model completions"""
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

    llm = LLM(
        model="Qwen/Qwen2.5-0.5B",
        max_model_len=700,  # prompt (~87) + max_tokens (500) + buffer
        gpu_memory_utilization=0.9,
        dtype="half",
        disable_log_stats=True,
    )

    sampling_params = SamplingParams(
        n=5,
        temperature=0.8,
        max_tokens=500,
        stop=["\ndef", "\nclass", "\n\n\n"],
    )

    outputs = llm.generate([PROMPT], sampling_params)

    print("="*70)
    print("PROMPT:")
    print(PROMPT)
    print("="*70)

    for i, comp in enumerate(outputs[0].outputs):
        print(f"\n--- Completion {i+1} ({len(comp.token_ids)} tokens) ---")
        print(repr(comp.text))
        print()
        print(comp.text)
