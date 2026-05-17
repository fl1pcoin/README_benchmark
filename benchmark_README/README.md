# README Benchmark

This folder contains the orchestration layer for comparing README generation tools on the repositories listed in `data/repo_list.csv`.

## Layout

```text
benchmark_README/
  data/repo_list.csv
  results/<run_name>/
  src/
  README_benchmark.ipynb
```

Each external tool keeps its own dependencies in its own checkout:

- `../OSA/.venv`
- `../ReadMeReady_eval/.venv`
- `../larch_eval/.venv`

The benchmark environment only needs the lightweight dependencies from `requirements.txt`.

## Environment

Create `benchmark_README/.env` or export the variables before opening the notebook:

```env
OPENROUTER_API_KEY=...
OPENAI_API_KEY=...
OPENAI_API_BASE=https://openrouter.ai/api/v1
OPENAI_BASE_URL=https://openrouter.ai/api/v1
GIT_TOKEN=...
GITHUB_TOKEN=...
HF_TOKEN=...
```

`OPENAI_API_KEY` can contain the same value as `OPENROUTER_API_KEY`. Some tools expect the OpenAI variable name even when they are pointed at OpenRouter.

## Run

1. Install the benchmark dependencies in the environment used by Jupyter:

   ```powershell
   pip install -r requirements.txt
   ```

2. Open `README_benchmark.ipynb`.
3. Edit the first configuration cell: choose `run_name`, `models`, enabled tools, and reset flags.
4. Run the notebook cells from top to bottom.

The notebook first clones repositories to `results/<run_name>/repositories`, checks out `commit_sha`, saves original READMEs and repository structures, then runs selected tools and writes status rows to `tool_status.csv`.

Evaluation is performed with DeepEval `GEval` using the README-quality criteria and six evaluation steps from OSA PR #355. `ACTUAL_OUTPUT` is a JSON object with `readme` and `repo_structure`; `EXPECTED_OUTPUT` is the original README.

## LArch

LArch is configured for local mode, not endpoint mode. The benchmark calls LArch programmatically to avoid the interactive prompts in `larch.cli`; the equivalent manual command is:

```powershell
larch --local --model <model> --openai-api-key <key> --input <repo> --input-context <seed_prompt.md> --out <output.md>
```

For `gpt-4.1`, `anthropic/claude-sonnet-4`, and `google/gemma-3-27b-it`, the local LArch code calls OpenRouter directly.
