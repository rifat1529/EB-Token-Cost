"""Minimal Ollama /api/chat client. Uses Ollama's own token counters."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class OllamaError(RuntimeError):
    """Raised when Ollama cannot complete a request."""


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:3b",
        timeout: int = 300,
        fresh_state: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        # keep_alive=0 asks Ollama to unload the model after each request.
        # This is slower, but helps prevent reuse of an in-memory prompt/KV cache.
        self.fresh_state = fresh_state

    def chat(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0,
        seed: int = 42,
        max_tokens: int = 256,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "seed": seed,
                "num_predict": max_tokens,
            },
        }
        if self.fresh_state:
            payload["keep_alive"] = 0

        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"Could not reach Ollama at {self.base_url}. "
                "Check that Ollama is running."
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise OllamaError("Ollama returned an invalid JSON response.") from exc

        required = ("message", "prompt_eval_count", "eval_count")
        missing = [key for key in required if key not in result]
        if missing:
            raise OllamaError(f"Ollama response is missing fields: {missing}")
        return result
