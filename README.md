# English vs Bangla Token Cost 

## Experiment configuration

* Model: `qwen2.5:3b`
* Ollama endpoint: `http://localhost:11434`
* API: `/api/chat`, non-streaming
* Temperature: `0`
* Seed: `42`
* Maximum generated tokens: `256` (`num\_predict`)
* Fixed system prompt: same text for both languages
* Cache mitigation: `keep\_alive=0` asks Ollama to unload the model after each call.
* Token usage: Ollama `prompt\_eval\_count` and `eval\_count` fields.

## Setup (Windows PowerShell)

Open PowerShell in this project folder:

```powershell
ollama pull qwen2.5:3b
ollama list
```

Ensure Ollama is running. Then run the fake-backend unit tests:

```powershell
python -m unittest discover -s tests -v
```

Run the real experiment:

```powershell
python main.py
```

This makes one baseline diagnostic call and then 100 prompt calls (50 English + 50 Bangla). Each result is appended immediately to `results/results.csv`, so completed work is retained if a later call fails.

If the run is interrupted, continue without repeating successful calls:

```powershell
python main.py --resume
```

To change settings explicitly:

```powershell
python main.py --model qwen2.5:3b --temperature 0 --seed 42 --max-tokens 256
```

## Output files

* `results/results.csv`: one row per language call; includes required columns and extra metadata.
* `results/baseline.json`: diagnostic baseline token counts for the fixed system prompt plus the tiny probe.

## Interpretation notes

* `prompt\_eval\_count` includes prompt formatting/chat-template tokens and the fixed system prompt. It is not the user-text-only token count.
* Baseline is reported separately and is not subtracted, because template tokenization can depend on message boundaries and content.
* `keep\_alive=0` is a cache mitigation request, not a mathematical guarantee about every possible cache layer. Note Ollama version and model digest in the final report.
* `done\_reason` is retained so answers stopped due to `length` can be identified. Check whether `max\_tokens=256` truncated any outputs.
* Latency includes the request round trip and model generation; the unload/reload policy may make it much higher than a warm-model run.
* A failed call is retained with an error and blank counts. Do not include failed rows in ratios or totals; rerun them using `--resume`.
* Course notional prices: $0.15 per million input tokens and $0.60 per million output tokens.

