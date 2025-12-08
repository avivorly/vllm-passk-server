"""Open vLLM server - no auth, anyone can use"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import traceback
import gc
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/vllm_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

class ErrorResponse(BaseModel):
    error: str
    detail: str

def clear_gpu_memory():
    """Clear GPU memory after an error"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        logger.info("GPU memory cleared")
    except Exception as e:
        logger.error(f"Failed to clear GPU memory: {e}")

@app.on_event("startup")
async def load_model():
    global llm
    try:
        from vllm import LLM
        logger.info("Loading model...")
        llm = LLM(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            trust_remote_code=True,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            max_model_len=2048,
            gpu_memory_utilization=0.95,
            dtype="half",
            enforce_eager=False,
            max_num_seqs=2500,
            max_num_batched_tokens=150000,
            disable_log_stats=True,
        )
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}\n{traceback.format_exc()}")
        clear_gpu_memory()
        raise

@app.post("/generate")
async def generate(req: GenerateRequest):
    global llm

    if llm is None:
        logger.error("Generate called but model not loaded")
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded", "detail": "Server is starting up or model failed to load"}
        )

    try:
        from vllm import SamplingParams

        stop = req.stop or ["\ndef", "\nclass", "\n\n\n"]

        # temperature=0 (greedy) requires n=1, so we run n separate requests
        if req.temperature == 0:
            sampling_params = SamplingParams(
                n=1,
                temperature=0,
                max_tokens=req.max_tokens,
                stop=stop,
            )
            outputs = llm.generate([req.prompt] * req.n, sampling_params)
            completions = [
                Completion(text=out.outputs[0].text, tokens=len(out.outputs[0].token_ids))
                for out in outputs
            ]
        else:
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

        logger.info(f"Generated {len(completions)} completions, {total_tokens} total tokens")
        return GenerateResponse(completions=completions, total_tokens=total_tokens)

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Generation failed: {error_msg}\n{error_trace}")

        # Clear GPU memory on error
        clear_gpu_memory()

        return JSONResponse(
            status_code=500,
            content={"error": "Generation failed", "detail": error_msg}
        )

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

@app.get("/logs")
async def get_logs():
    """Get recent server logs"""
    try:
        with open('/tmp/vllm_server.log', 'r') as f:
            lines = f.readlines()
            return {"logs": "".join(lines[-100:])}  # Last 100 lines
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
