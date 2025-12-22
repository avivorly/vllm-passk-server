import sys
sys.path.insert(0, '/workspace/orkspace/LiveCodeBench')
from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from transformers import AutoTokenizer

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

# Load dataset
dataset = load_code_generation_dataset(release_version="v6")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)

# Get token lengths
prompt_data = []
for idx, problem in enumerate(dataset):
    prompt = format_zero_shot_prompt(problem)
    tokens = tokenizer.encode(prompt)
    token_length = len(tokens)
    title = problem.question_title if hasattr(problem, 'question_title') else f"Question {idx}"
    prompt_data.append((idx, token_length, title))

# Sort by length
prompt_data_sorted = sorted(prompt_data, key=lambda x: x[1], reverse=True)

print("\n" + "="*80)
print("TOP 5 LONGEST PROMPTS (Full Details)")
print("="*80)
for rank, (idx, length, title) in enumerate(prompt_data_sorted[:5], 1):
    print(f"\n{rank}. Index: {idx}")
    print(f"   Token Length: {length}")
    print(f"   Title: {title}")
print("="*80)
