"""Client to call the vLLM server from any machine"""
import requests

SERVER_URL = "http://YOUR_SERVER_IP:8000"  # Change this!

def generate(prompt: str, n: int = 100, temperature: float = 0.8, max_tokens: int = 500):
    """Call the server and get completions"""
    response = requests.post(
        f"{SERVER_URL}/generate",
        json={
            "prompt": prompt,
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=300,  # 5 min timeout for large batches
    )
    response.raise_for_status()
    return response.json()

# Example usage
if __name__ == "__main__":
    PROMPT = '''from typing import List


def filter_by_substring(strings: List[str], substring: str) -> List[str]:
    """ Filter an input list of strings only for ones that contain given substring
    >>> filter_by_substring([], 'a')
    []
    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')
    ['abc', 'bacd', 'array']
    """
'''

    print("Calling server for 100 completions...")
    result = generate(PROMPT, n=100)

    print(f"Got {len(result['completions'])} completions")
    print(f"Total tokens: {result['total_tokens']}")

    print("\nFirst 3 completions:")
    for i, comp in enumerate(result['completions'][:3]):
        print(f"\n--- {i+1} ({comp['tokens']} tokens) ---")
        print(comp['text'])
