"""Open vLLM server - no auth, anyone can use"""
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(title="vLLM Open Server")

# Global model - loaded once at startup
llm = None

class GenerateRequest(BaseModel):
    prompt: str
    n: int = 100                    # Number of completions
    temperature: float = 0.8
    max_tokens: int = 500
    stop: Optional[List[str]] = None

class Completion(BaseModel):
    text: str
    tokens: int

class GenerateResponse(BaseModel):
    completions: List[Completion]
    total_tokens: int

@app.on_event("startup")
async def load_model():
    global llm
    from vllm import LLM
    print("Loading model...")
    llm = LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_model_len=2048,  # Instruct model needs more for verbose outputs
        gpu_memory_utilization=0.95,
        dtype="half",
        enforce_eager=False,
        max_num_seqs=2500,
        max_num_batched_tokens=150000,
        disable_log_stats=True,
    )
    print("Model loaded!")

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    from vllm import SamplingParams

    stop = req.stop or ["\ndef", "\nclass", "\n\n\n"]

    sampling_params = SamplingParams(
        n=req.n,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        stop=stop,
    )

    outputs = llm.generate([req.prompt], sampling_params)

    completions = [
        Completion(text=c.text, tokens=len(c.token_ids))
        for c in outputs[0].outputs
    ]

    total_tokens = sum(c.tokens for c in completions)

    return GenerateResponse(completions=completions, total_tokens=total_tokens)

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": llm is not None}

@app.get("/")
async def hello():
    return {"message": "Hello World! vLLM server is running."}

@app.get("/download")
async def download_project():
    return FileResponse(
        "/home/claude-user/vllm_project.zip",
        filename="vllm_project.zip",
        media_type="application/zip"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
