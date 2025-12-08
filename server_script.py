"""Server script for batch processing questions through vLLM server."""
import argparse
import json
import requests


def main():
    parser = argparse.ArgumentParser(description="Generate code samples for pass@k evaluation")
    parser.add_argument("--input_file", required=True, help="Path to input JSONL file with questions")
    parser.add_argument("--output_file", required=True, help="Path to output JSONL file for results")
    parser.add_argument("--n", type=int, default=100, help="Number of samples to generate per question")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=500, help="Maximum tokens per completion")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling")
    parser.add_argument("--stop", type=str, nargs="*", default=["\ndef", "\nclass", "\n\n\n"], help="Stop sequences")
    parser.add_argument("--server_url", type=str, default="http://localhost:8000/generate", help="vLLM server URL")
    args = parser.parse_args()

    with open(args.input_file, "r") as f_in, open(args.output_file, "w") as f_out:
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            question = data["question"]

            response = requests.post(
                args.server_url,
                json={
                    "prompt": question,
                    "n": args.n,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "top_p": args.top_p,
                    "stop": args.stop,
                },
                timeout=300,
            )
            response.raise_for_status()
            result = response.json()

            output = {
                "question": question,
                "completions": [c["text"] for c in result["completions"]],
                "total_tokens": result["total_tokens"],
            }
            f_out.write(json.dumps(output) + "\n")
            print(f"Processed question {line_num}: {result['total_tokens']} tokens generated")

    print(f"Done. Results written to {args.output_file}")


if __name__ == "__main__":
    main()
