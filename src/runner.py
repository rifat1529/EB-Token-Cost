"""Run all aligned English/Bangla prompts against local Ollama and save results.csv."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ollama_client import OllamaClient
from token_counter import count_call, measure_baseline

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's request clearly, "
    "accurately, and concisely. Do not add unnecessary introductory text."
)

MODEL = "qwen2.5:3b"
TEMPERATURE = 0
SEED = 42
MAX_TOKENS = 256

RESULT_FIELDS = [
    "id", "language", "prompt", "output",
    "input_tokens", "output_tokens", "cost_usd", "latency_s",
    "input_cost_usd", "output_cost_usd", "model",
    "temperature", "seed", "max_tokens", "done_reason", "error",
]


def load_dataset(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        expected = {"id", "category", "english", "bangla"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f"Dataset must contain columns: {sorted(expected)}")
        rows = list(reader)
    if len(rows) != 50:
        raise ValueError(f"Expected exactly 50 aligned pairs; found {len(rows)}")
    return rows


def read_completed_ids(path: Path) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("error", "") == "" and row.get("input_tokens", "") != "":
                completed.add((row["id"], row["language"]))
    return completed


def append_result(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset" / "dataset.tsv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "results.csv")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--resume", action="store_true",
                        help="Skip successful (id, language) rows already in results.csv")
    args = parser.parse_args()

    pairs = load_dataset(args.dataset)
    client = OllamaClient(
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        fresh_state=True,
    )

    # Save a diagnostic baseline separately. It is not subtracted from measured counts.
    print("Measuring fixed-system/template baseline...")
    baseline = measure_baseline(
        client.chat, SYSTEM_PROMPT,
        temperature=args.temperature, seed=args.seed, max_tokens=1,
    )
    baseline_record = {
        **baseline,
        "model": args.model,
        "system_prompt": SYSTEM_PROMPT,
        "note": "Diagnostic only; not subtracted from prompt_eval_count.",
    }
    baseline_path = args.output.parent / "baseline.json"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(
        json.dumps(baseline_record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Baseline saved: {baseline_path}")
    print(f"Baseline prompt_eval_count: {baseline['baseline_prompt_eval_count']}")

    completed = read_completed_ids(args.output) if args.resume else set()
    total = 100
    done = len(completed)

    # Paired order: English then Bangla for each ID; both receive identical settings.
    for pair in pairs:
        pair_id = str(pair["id"])
        for language, prompt in (("English", pair["english"]), ("Bangla", pair["bangla"])):
            if (pair_id, language) in completed:
                continue

            print(f"[{done + 1}/{total}] ID {pair_id} — {language}")
            try:
                measured = count_call(
                    client.chat,
                    prompt,
                    SYSTEM_PROMPT,
                    temperature=args.temperature,
                    seed=args.seed,
                    max_tokens=args.max_tokens,
                )
                result = {
                    "id": pair_id,
                    "language": language,
                    "prompt": prompt,
                    "output": measured["output"],
                    "input_tokens": measured["input_tokens"],
                    "output_tokens": measured["output_tokens"],
                    "cost_usd": f"{measured['cost_usd']:.12f}",
                    "latency_s": f"{measured['latency_s']:.6f}",
                    "input_cost_usd": f"{measured['input_cost_usd']:.12f}",
                    "output_cost_usd": f"{measured['output_cost_usd']:.12f}",
                    "model": args.model,
                    "temperature": args.temperature,
                    "seed": args.seed,
                    "max_tokens": args.max_tokens,
                    "done_reason": measured.get("done_reason") or "",
                    "error": "",
                }
            except Exception as exc:
                # Keep a row for every attempted call; failed rows are clearly marked.
                result = {
                    "id": pair_id, "language": language, "prompt": prompt,
                    "output": "", "input_tokens": "", "output_tokens": "",
                    "cost_usd": "", "latency_s": "",
                    "input_cost_usd": "", "output_cost_usd": "",
                    "model": args.model, "temperature": args.temperature,
                    "seed": args.seed, "max_tokens": args.max_tokens,
                    "done_reason": "", "error": str(exc),
                }
                print(f"ERROR: {exc}", file=sys.stderr)

            append_result(args.output, result)
            done += 1
            if result["error"] == "":
                print(
                    f"  input={result['input_tokens']} output={result['output_tokens']} "
                    f"cost=${result['cost_usd']} latency={result['latency_s']}s"
                )

    print(f"\nFinished. Rows attempted this run: {done}.")
    print(f"Results file: {args.output}")
    print("Check error column before using the file for statistical comparison.")


if __name__ == "__main__":
    main()
