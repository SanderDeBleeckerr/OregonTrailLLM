from __future__ import annotations
import argparse
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-E4B-it"
DEFAULT_CTX = 16384


def load_model(model_id: str, four_bit: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    kwargs: dict = {}
    if device == "cuda":
        kwargs["device_map"] = "cuda"
        if four_bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            kwargs["dtype"] = torch.bfloat16
    else:
        kwargs["dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return tokenizer, model


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # quieter console
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"status": "ready", "model": self.server.model_id,
                             "ctx": self.server.ctx})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/generate":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "malformed JSON body"})
            return

        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            self._json(400, {"error": "missing prompt"})
            return
        temperature = float(payload.get("temperature") or 0.0)
        seed = payload.get("seed")
        max_tokens = int(payload.get("max_tokens") or 1024)

        try:
            text = self._generate(prompt, temperature, seed, max_tokens)
            self._json(200, {"response": text})
        except Exception as e:  # noqa: BLE001 - surfaced to the caller, never crashes the server
            traceback.print_exc()
            self._json(502, {"error": str(e) or repr(e)})

    def _generate(self, prompt: str, temperature: float, seed,
                  max_tokens: int) -> str:
        tokenizer = self.server.tokenizer
        model = self.server.model
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        ids = encoded["input_ids"]
        budget = self.server.ctx - max_tokens
        if ids.shape[1] > budget:
            ids = ids[:, -budget:]
        ids = ids.to(model.device)

        with self.server.model_lock:
            if isinstance(seed, int):
                transformers.set_seed(seed)
            do_sample = temperature > 0
            with torch.inference_mode():
                out = model.generate(
                    ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=0.95 if do_sample else None,
                    top_k=None,
                    pad_token_id=tokenizer.eos_token_id,
                )
        return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--ctx", type=int, default=DEFAULT_CTX,
                    help="token budget for prompt + completion (bounds the KV cache)")
    ap.add_argument("--no-4bit", action="store_true",
                    help="load bf16 instead of NF4-quantized (needs ~2.5x the VRAM)")
    args = ap.parse_args()

    print(f"Loading {args.model}...", flush=True)
    tokenizer, model = load_model(args.model, four_bit=not args.no_4bit)
    print(f"Ready on {model.device}.", flush=True)

    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    server.model_id = args.model
    server.tokenizer = tokenizer
    server.model = model
    server.model_lock = threading.Lock()  # one generation at a time; the GPU is shared with images
    server.ctx = args.ctx

    print(f"Text server running at http://{args.bind}:{args.port}/generate", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
