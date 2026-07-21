"""Local scene-image server for Westward Trail.

    python image_server.py

Runs alongside serve.py, on its own port, as a plain diffusers + CUDA
pipeline. Text generation (narrator/scorer/quiz) lives in its own sidecar,
text_server.py, on yet another port -- one resident model per process.

The checkpoint (SD-Turbo) is loaded once at startup and kept resident, since
reloading it per request would dwarf the ~1-step inference time. Deliberately
stdlib-only for the HTTP layer, matching serve.py.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from diffusers import AutoPipelineForText2Image

MODEL_ID = "stabilityai/sd-turbo"


def load_pipeline(model_id: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to(device)
    return pipe


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
            self._json(200, {"status": "ready"})
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

        try:
            with self.server.pipe_lock:
                image = self.server.pipe(
                    prompt=prompt,
                    num_inference_steps=self.server.steps,
                    guidance_scale=0.0,  # SD-Turbo is distilled for CFG-free sampling
                    width=self.server.size,
                    height=self.server.size,
                ).images[0]
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            self._json(200, {"image": base64.b64encode(buf.getvalue()).decode()})
        except Exception as e:  # noqa: BLE001 - surfaced to the caller, never crashes the server
            self._json(502, {"error": str(e)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--steps", type=int, default=1, help="SD-Turbo is tuned for 1-4 steps")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    print(f"Loading {args.model}...", flush=True)
    pipe = load_pipeline(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Ready on {device}.", flush=True)

    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    server.pipe = pipe
    server.pipe_lock = threading.Lock()  # one image at a time; GPU memory isn't free
    server.steps = args.steps
    server.size = args.size

    print(f"Scene-image server running at http://{args.bind}:{args.port}/generate", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
