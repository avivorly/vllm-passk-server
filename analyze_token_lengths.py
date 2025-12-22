import sys
sys.path.insert(0, '/workspace/orkspace/LiveCodeBench')
from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from transformers import AutoTokenizer
import numpy as np

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
print("Loading dataset...")
dataset = load_code_generation_dataset(release_version="v6")
print(f"Loaded {len(dataset)} questions")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
print("Tokenizer loaded")

# Tokenize all prompts
print("\nTokenizing prompts...")
token_lengths = []
prompt_data = []

for idx, problem in enumerate(dataset):
    prompt = format_zero_shot_prompt(problem)
    tokens = tokenizer.encode(prompt)
    token_length = len(tokens)
    token_lengths.append(token_length)
    prompt_data.append((idx, token_length, problem.question_title if hasattr(problem, 'question_title') else f"Question {idx}"))

    if (idx + 1) % 25 == 0:
        print(f"  Processed {idx + 1}/{len(dataset)} questions...")

print(f"  Processed {len(dataset)}/{len(dataset)} questions...")

# Calculate statistics
token_lengths_array = np.array(token_lengths)
percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
percentile_values = np.percentile(token_lengths_array, percentiles)

mean_length = np.mean(token_lengths_array)
std_length = np.std(token_lengths_array)

# Find top 5 longest prompts
prompt_data_sorted = sorted(prompt_data, key=lambda x: x[1], reverse=True)
top_5 = prompt_data_sorted[:5]

# Print results
print("\n" + "="*80)
print("TOKEN LENGTH ANALYSIS - LiveCodeBench v6 Dataset")
print("="*80)
print(f"Total Questions: {len(dataset)}")
print(f"Tokenizer: Qwen/Qwen2.5-0.5B")
print("="*80)

print("\nSTATISTICAL SUMMARY")
print("-"*80)
print(f"{'Statistic':<20} {'Token Length':>15}")
print("-"*80)
print(f"{'Mean':<20} {mean_length:>15.2f}")
print(f"{'Std Dev':<20} {std_length:>15.2f}")
print("-"*80)

print("\nPERCENTILE DISTRIBUTION")
print("-"*80)
print(f"{'Percentile':<20} {'Token Length':>15}")
print("-"*80)
for pct, val in zip(percentiles, percentile_values):
    pct_label = "Min" if pct == 0 else ("Max" if pct == 100 else f"{pct}th")
    print(f"{pct_label:<20} {int(val):>15}")
print("-"*80)

print("\nTOP 5 LONGEST PROMPTS")
print("-"*80)
print(f"{'Index':<10} {'Token Length':<15} {'Question Title'}")
print("-"*80)
for idx, length, title in top_5:
    title_short = title[:50] + "..." if len(title) > 50 else title
    print(f"{idx:<10} {length:<15} {title_short}")
print("-"*80)

print("\nTOKEN LENGTH DISTRIBUTION (Histogram)")
print("-"*80)
bins = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, float('inf')]
bin_labels = ["0-200", "200-400", "400-600", "600-800", "800-1000",
              "1000-1200", "1200-1400", "1400-1600", "1600-1800", "1800-2000", "2000+"]

hist, _ = np.histogram(token_lengths_array, bins=bins)
print(f"{'Range':<15} {'Count':>10} {'Percentage':>12}")
print("-"*80)
for label, count in zip(bin_labels, hist):
    percentage = (count / len(dataset)) * 100
    bar = "#" * int(percentage / 2)
    print(f"{label:<15} {count:>10} {percentage:>11.1f}%  {bar}")
print("-"*80)
