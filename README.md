# vLLM Open Server

A FastAPI server for generating code completions using vLLM with Qwen2.5-0.5B-Instruct model. Designed for pass@k evaluation experiments.

## Features

- High-throughput inference with vLLM optimizations
- Prefix caching and chunked prefill enabled
- SSH tunnel support for remote access via localhost.run
- No authentication required - open API

## Files

| File | Description |
|------|-------------|
| `vllm_server.py` | FastAPI server with `/generate` endpoint |
| `vllm_client.py` | Client example for calling the API |
| `tunnel.sh` | Robust SSH tunnel with auto-restart |
| `compare_formats.py` | Compare raw vs chat template outputs |

## Quick Start

### Start the server
```bash
python vllm_server.py
```

### Expose via SSH tunnel (for remote access)
```bash
bash tunnel.sh
```
Tunnel URL will appear in `/tmp/tunnel.log`.

## API

### POST /generate
Generate multiple completions for a prompt.

**Request:**
```json
{
    "prompt": "def hello():\n    ",
    "n": 100,
    "temperature": 0.8,
    "max_tokens": 500,
    "stop": ["\ndef", "\nclass"]
}
```

**Response:**
```json
{
    "completions": [
        {"text": "print('Hello')", "tokens": 5},
        ...
    ],
    "total_tokens": 1500
}
```

### GET /health
Check server status.

### GET /
Hello world endpoint.

## Client Example

```python
import requests

resp = requests.post("http://localhost:8000/generate", json={
    "prompt": "def add(a, b):\n    ",
    "n": 10,
    "temperature": 0.8,
    "max_tokens": 100
})
print(resp.json())
```

## Model Configuration

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- max_model_len: 2048
- GPU utilization: 95%
- dtype: half (FP16)
- Prefix caching: enabled
- Chunked prefill: enabled

## Notes

- Raw code completion: ~19 tokens avg (just function body)
- Chat template: ~200-335 tokens avg (verbose with explanations)
- For evalplus compatibility, use chat template format with stop token `\n```\n`
