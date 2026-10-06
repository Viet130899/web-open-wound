"""Block 3 chatbot inference server — HuggingFace transformers + PEFT, CUDA (Jetson Orin).

Loads a base causal LM plus a LoRA adapter and serves it over HTTP with the
tiny /health + /chat contract expected by app.py's `_call_model()`.

Usage:
    python model_server.py --base <hf_repo_or_path> --adapter <adapter_dir> --name <display_name> --port <port>
"""
import argparse
import threading

import torch
import uvicorn
from fastapi import FastAPI
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_CACHE_DIR = str((__import__("pathlib").Path(__file__).parent / ".hf_cache"))

app = FastAPI()
_lock = threading.Lock()
_state: dict = {}


class ChatRequest(BaseModel):
    message: str
    system: str = ""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


@app.get("/health")
def health():
    return {"status": "ok", "name": _state.get("name"), "device": _state.get("device")}


@app.post("/chat")
def chat(req: ChatRequest):
    tokenizer = _state["tokenizer"]
    model = _state["model"]

    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.append({"role": "user", "content": req.message})

    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)

    with _lock, torch.inference_mode():
        do_sample = req.temperature > 0
        output = model.generate(
            **encoded,
            max_new_tokens=max(1, min(req.max_tokens, 2048)),
            do_sample=do_sample,
            temperature=req.temperature if do_sample else None,
            top_p=req.top_p if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = encoded["input_ids"].shape[1]
    reply = tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True)
    return {"reply": reply.strip()}


def load(base: str, adapter: str, name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base, cache_dir=HF_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, cache_dir=HF_CACHE_DIR
    ).to(device)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    _state["tokenizer"] = tokenizer
    _state["model"] = model
    _state["name"] = name
    _state["device"] = device
    print(f"[block3] '{name}' loaded on {device}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="chatbot")
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()

    load(args.base, args.adapter, args.name)
    uvicorn.run(app, host="127.0.0.1", port=args.port)
