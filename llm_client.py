"""Thin client for a local Ollama server.

Usage:
    client = OllamaClient()                       # DEFAULT_MODEL
    client = OllamaClient(model="qwen2.5:32b")    # any pulled Ollama tag
    text = client.generate("Classify: ...", temperature=0.0)

Reasoning models (qwen3, deepseek-r1, ...) put their chain-of-thought in a
separate `thinking` field and leave `response` empty, which reads to the rest
of the harness as an unparseable turn. We turn thinking off at the API level
so the whole token budget goes to the JSON we actually asked for.

Ollama also defaults to a 4096-token context regardless of what the model
supports, which silently truncates the oldest history first -- exactly where
the party intro the memory quizzes probe lives. We ask the server what the
loaded model supports and use that.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

FALLBACK_NUM_CTX = 32768
DEFAULT_MODEL = "hf.co/bartowski/Qwen2.5-14B-Instruct-GGUF:IQ4_XS"


class OllamaClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = "http://localhost:11434",
        think: bool = False,
        num_ctx: int | None = None,
    ):
        self.model = model
        self.host = host
        self.url = f"{host}/api/generate"
        self.think = think
        # Ollama rejects `think` for models with no reasoning mode; probed once.
        self._send_think = True
        self._num_ctx = num_ctx

    def _post(self, payload: dict, timeout: int, path: str = "/api/generate") -> dict:
        req = urllib.request.Request(
            f"{self.host}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def context_limit(self) -> int:
        if self._num_ctx is not None:
            return self._num_ctx
        try:
            info = self._post({"model": self.model}, timeout=60, path="/api/show")
            for key, val in (info.get("model_info") or {}).items():
                if key.endswith(".context_length") and isinstance(val, int) and val > 0:
                    self._num_ctx = val
                    return val
        except Exception:  # noqa: BLE001 - detection is best-effort
            pass
        self._num_ctx = FALLBACK_NUM_CTX
        return self._num_ctx

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 1024,
        retries: int = 3,
    ) -> str:
        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": self.context_limit(),
        }
        if seed is not None:
            options["seed"] = seed
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }

        last_err: Exception | None = None
        for attempt in range(retries):
            body = dict(payload)
            if self._send_think:
                body["think"] = self.think
            try:
                resp = self._post(body, timeout=300)
            except urllib.error.HTTPError as e:  # noqa: PERF203
                detail = e.read().decode(errors="replace")
                if self._send_think and "think" in detail.lower():
                    self._send_think = False
                    continue
                last_err = RuntimeError(f"HTTP {e.code}: {detail[:200]}")
                time.sleep(2**attempt)
                continue
            except Exception as e:  # noqa: BLE001 - log and retry
                last_err = e
                time.sleep(2**attempt)
                continue

            text = (resp.get("response") or "").strip()
            if text:
                return text
            last_err = RuntimeError(
                f"empty response (done_reason={resp.get('done_reason')}, "
                f"thinking={len(resp.get('thinking') or '')} chars)"
            )
            time.sleep(2**attempt)
        raise RuntimeError(f"Ollama request failed after {retries} retries: {last_err}")
