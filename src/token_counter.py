"""Reusable wrapper that extracts Ollama token usage from a chat response."""
from __future__ import annotations

import time
from typing import Any, Callable

from cost_calculator import calculate_cost


def count_call(
    backend: Callable[..., dict[str, Any]],
    prompt: str,
    system_prompt: str,
    *,
    temperature: float = 0,
    seed: int = 42,
    max_tokens: int = 256,
) -> dict[str, Any]:
    """
    Run one model call and return output, Ollama token counts, cost and latency.

    `backend` must accept prompt, system_prompt, temperature, seed, max_tokens
    and return an Ollama-compatible response dictionary.
    """
    started = time.perf_counter()
    response = backend(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
    )
    latency_s = time.perf_counter() - started

    try:
        input_tokens = int(response["prompt_eval_count"])
        output_tokens = int(response["eval_count"])
        output_text = response["message"]["content"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Backend response does not contain valid Ollama usage fields.") from exc

    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Ollama returned a negative token count.")

    costs = calculate_cost(input_tokens, output_tokens)
    return {
        "output": output_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        **costs,
        "latency_s": latency_s,
        "done": response.get("done"),
        "done_reason": response.get("done_reason"),
    }


def measure_baseline(
    backend: Callable[..., dict[str, Any]],
    system_prompt: str,
    *,
    temperature: float = 0,
    seed: int = 42,
    max_tokens: int = 1,
) -> dict[str, int]:
    """
    Diagnostic call using a tiny user prompt. The count includes the fixed
    system prompt and chat-template overhead. Do not subtract it from results.
    """
    response = backend(
        prompt="Reply with OK.",
        system_prompt=system_prompt,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
    )
    return {
        "baseline_prompt_eval_count": int(response["prompt_eval_count"]),
        "baseline_eval_count": int(response["eval_count"]),
    }
