"""Evaluate pass@k - run 100 completions and check correctness"""
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

    # Test cases
    TEST_CASES = [
        ([], 'a', []),
        (['abc', 'bacd', 'cde', 'array'], 'a', ['abc', 'bacd', 'array']),
        (['hello', 'world', 'help'], 'el', ['hello', 'help']),
        (['foo', 'bar', 'baz'], 'x', []),
        (['test'], 'test', ['test']),
    ]

    print("Loading model with max optimizations...")
    llm = LLM(
        model="Qwen/Qwen2.5-0.5B",
        trust_remote_code=True,
        enable_prefix_caching=True,
        max_model_len=150,
        gpu_memory_utilization=0.95,
        dtype="half",
        enforce_eager=False,
        max_num_seqs=3000,
        max_num_batched_tokens=200000,
        disable_log_stats=True,
    )

    print("Generating 2500 completions with temperature=0.8...\n")

    sampling_params = SamplingParams(
        n=2500,
        temperature=0.8,
        max_tokens=100,
        stop=["\ndef", "\nclass", "\n\n\n"],
    )

    import time
    start = time.time()
    outputs = llm.generate([PROMPT], sampling_params)
    elapsed = time.time() - start
    completions = outputs[0].outputs

    total_tokens = sum(len(c.token_ids) for c in completions)
    tps = total_tokens / elapsed
    print(f"Generated {len(completions)} completions in {elapsed:.2f}s")
    print(f">>> {tps:.0f} TPS | {len(completions)/elapsed:.0f} CRS <<<\n")

    correct = 0
    incorrect = 0
    errors = []

    for i, comp in enumerate(completions):
        code = comp.text
        full_code = PROMPT + code

        try:
            # Execute the code
            local_ns = {}
            exec(full_code, {"List": list}, local_ns)
            func = local_ns.get('filter_by_substring')

            if func is None:
                errors.append((i+1, code.strip()[:80], "Function not defined"))
                incorrect += 1
                continue

            # Test all cases
            all_passed = True
            fail_reason = None
            for inputs, substring, expected in TEST_CASES:
                try:
                    result = func(inputs, substring)
                    if result != expected:
                        all_passed = False
                        fail_reason = f"Expected {expected}, got {result}"
                        break
                except Exception as e:
                    all_passed = False
                    fail_reason = f"Runtime error: {e}"
                    break

            if all_passed:
                correct += 1
            else:
                incorrect += 1
                errors.append((i+1, code.strip()[:80], fail_reason))

        except SyntaxError as e:
            incorrect += 1
            errors.append((i+1, code.strip()[:80], f"Syntax error: {e}"))
        except Exception as e:
            incorrect += 1
            errors.append((i+1, code.strip()[:80], f"Error: {e}"))

    print("=" * 70)
    total = correct + incorrect
    pct = correct * 100 / total
    print(f"RESULTS: {correct}/{total} correct ({pct:.1f}% pass@{total})")
    print("=" * 70)

    print(f"\n❌ FAILURES ({incorrect} total):\n")
    for idx, code, reason in errors[:20]:  # Show first 20 failures
        print(f"#{idx}: {code}")
        print(f"     → {reason}\n")

    if len(errors) > 20:
        print(f"... and {len(errors) - 20} more failures")
