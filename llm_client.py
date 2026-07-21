from __future__ import annotations

import base64
import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_MODEL = "google/gemma-4-E4B-it"
DEFAULT_TEXT_HOST = "http://localhost:8091"

THINK_RE = re.compile(
    r"<think(?:ing)?>.*?(?:</think(?:ing)?>|\Z)", re.DOTALL | re.IGNORECASE
)


def strip_think(text: str) -> str:
    return THINK_RE.sub("", text).strip()


def _get_json(url: str, timeout: float) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


def ensure_text_server(host: str, model: str) -> subprocess.Popen | None:
    """Start text_server.py as a child process so `python serve.py` (or
    run_experiments) alone is enough -- nobody wants a second terminal.

    Only spawns if nothing is already answering on that host. An already
    running server with a different model is used as-is with a warning:
    loading a second multi-gigabyte model beats out a mismatch note on no
    dimension that matters here.
    """
    health = _get_json(f"{host}/health", timeout=1.5)
    if health is not None:
        loaded = health.get("model", "?")
        note = "" if loaded == model else f" -- NOT the requested {model}, using it anyway"
        print(f"  game master : {host} (already running: {loaded}{note})")
        return None
    port = urllib.parse.urlparse(host).port or 8091
    root = pathlib.Path(__file__).parent
    proc = subprocess.Popen(
        [sys.executable, str(root / "text_server.py"),
         "--port", str(port), "--model", model],
        cwd=root,
    )
    print(f"  game master : {model} via {host} (starting in the background, "
          f"pid {proc.pid}; first run downloads the model, can take a while)")
    return proc


class TextClient:
    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_TEXT_HOST):
        self.model = model
        self.host = host
        self.last_think_chars = 0
        self._ready = False

    def _wait_ready(self, timeout_s: float = 900) -> None:
        """Block until the server answers /health. A freshly spawned
        text_server spends anywhere from a minute (cached weights) to much
        longer (first-run download) loading before it starts listening."""
        if self._ready:
            return
        deadline = time.time() + timeout_s
        announced = False
        while time.time() < deadline:
            if _get_json(f"{self.host}/health", timeout=2) is not None:
                self._ready = True
                return
            if not announced:
                print(f"(waiting for the game master at {self.host} to load...)")
                announced = True
            time.sleep(2)
        raise RuntimeError(
            f"text_server at {self.host} did not become ready within {int(timeout_s)}s"
        )

    def _post(self, payload: dict, timeout: int) -> dict:
        req = urllib.request.Request(
            f"{self.host}/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 1024,
        retries: int = 3,
    ) -> str:
        self._wait_ready()
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed

        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                resp = self._post(payload, timeout=600)
            except urllib.error.HTTPError as e:  # noqa: PERF203
                detail = e.read().decode(errors="replace")
                last_err = RuntimeError(f"HTTP {e.code}: {detail[:200]}")
                time.sleep(2**attempt)
                continue
            except Exception as e:  # noqa: BLE001 - log and retry
                last_err = e
                time.sleep(2**attempt)
                continue

            raw = (resp.get("response") or "").strip()
            text = strip_think(raw)
            self.last_think_chars = len(raw) - len(text)
            if text:
                return text
            last_err = RuntimeError("empty response")
            time.sleep(2**attempt)
        raise RuntimeError(f"text_server request failed after {retries} retries: {last_err}")


class SceneImageClient:
    """Talks to image_server.py.

    Scene images run through a plain diffusers + CUDA process, which has real
    Windows wheels, on its own port -- same pattern as the text server.
    """

    def __init__(self, host: str = "http://localhost:8090"):
        self.host = host

    def generate_image(self, prompt: str, timeout: int = 60) -> bytes:
        req = urllib.request.Request(
            f"{self.host}/generate",
            data=json.dumps({"prompt": prompt}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            raise RuntimeError(f"image_server HTTP {e.code}: {detail[:300]}") from e
        image = data.get("image")
        if not image:
            raise RuntimeError(f"image_server returned no image: {data}")
        return base64.b64decode(image)
