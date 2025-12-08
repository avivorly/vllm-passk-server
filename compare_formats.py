"""Compare raw vs chat template outputs"""
import requests
import time

SERVER = "http://localhost:8000"

# The code completion prompt
CODE_PROMPT = '''from typing import List


def filter_by_substring(strings: List[str], substring: str) -> List[str]:
    """ Filter an input list of strings only for ones that contain given substring
    >>> filter_by_substring([], 'a')
    []
    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')
    ['abc', 'bacd', 'array']
    """
'''

# Chat template version
CHAT_PROMPT = '''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Complete the following Python function:

from typing import List


def filter_by_substring(strings: List[str], substring: str) -> List[str]:
    """ Filter an input list of strings only for ones that contain given substring
    >>> filter_by_substring([], 'a')
    []
    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')
    ['abc', 'bacd', 'array']
    """
<|im_end|>
<|im_start|>assistant
'''

def test_format(name, prompt, stop, n=5):
    print(f"\n{'='*60}")
    print(f"{name}")
    print('='*60)

    start = time.time()
    resp = requests.post(f"{SERVER}/generate", json={
        "prompt": prompt,
        "n": n,
        "temperature": 0.8,
        "max_tokens": 500,
        "stop": stop
    }, timeout=60)
    elapsed = time.time() - start

    data = resp.json()

    total_tokens = 0
    for i, c in enumerate(data['completions'][:3]):  # Show first 3
        total_tokens += c['tokens']
        print(f"\n--- Completion {i+1}: {c['tokens']} tokens ---")
        # Show first 150 chars
        text = c['text']
        if len(text) > 200:
            print(text[:200] + "...")
        else:
            print(text)

    avg = sum(c['tokens'] for c in data['completions']) / len(data['completions'])
    print(f"\n>>> AVERAGE: {avg:.0f} tokens per completion")
    print(f">>> Generated {n} completions in {elapsed:.2f}s")

if __name__ == "__main__":
    # Test 1: Raw code completion
    test_format(
        "1. RAW CODE COMPLETION (no template)",
        CODE_PROMPT,
        stop=["\ndef", "\nclass", "\n\n\n"]
    )

    # Test 2: Chat template
    test_format(
        "2. CHAT TEMPLATE (Qwen instruct format)",
        CHAT_PROMPT,
        stop=["<|im_end|>"]
    )

    print("\n" + "="*60)
    print("CONCLUSION:")
    print("- Raw format: ~15-30 tokens (just the function body)")
    print("- Chat format: ~200-300 tokens (explanation + code block)")
    print("="*60)
