import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Type

import pandas as pd
import requests
from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from pydantic import BaseModel

from .notebook_utils import BenchmarkPaths, model_label


logger = logging.getLogger(__name__)


README_QUALITY_CRITERIA = """
Determine whether the AI-generated Readme file (ACTUAL_OUTPUT)
is better than the original one (EXPECTED_OUTPUT).
ACTUAL_OUTPUT contains two fields: 'readme', which contains the generated README itself,
and 'repo_structure' which is json with repository's structure.
Generated README's content must be consistent with the provided repository structure.
The ACTUAL_OUTPUT does not necessary have to be the same as EXPECTED_OUTPUT,
Your goal is to determine which text is better, using the provided Evaluations steps.
Readme structure does not matter much as long as it passes the evaluation steps.
"""


README_QUALITY_STEPS = [
    "Step 1: Does the provided structure of the repository address README content?",
    "Step 2: Does the README provide a clear and accurate overview of the repository's purpose?",
    "Step 3: Are installation and setup instructions included and easy to follow?",
    "Step 4: Are usage examples provided and do they clearly demonstrate functionality?",
    "Step 5: Are dependencies or requirements listed appropriately?",
    "Step 6: Is the README easy to read, well-structured, and free of confusing language?",
]


def _strip_markdown_json_fence(text: str) -> str:
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    cleaned = cleaned.removeprefix("```").strip()
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].lstrip()
    if "```" in cleaned:
        cleaned = cleaned.split("```", 1)[0].strip()
    return cleaned


class CustomLLM(DeepEvalBaseLLM):
    """OpenAI-compatible chat API wrapper for DeepEval GEval."""

    def __init__(
        self,
        api: str = "openrouter",
        model: str = "gpt-4.1",
        url: str = "https://openrouter.ai/api/v1",
        *,
        max_tokens: int = 16384,
        request_timeout: float = 180.0,
        use_json_object_mode: bool = True,
    ):
        self.api = api
        self.model_name = model
        self.url = url.rstrip("/")
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.use_json_object_mode = use_json_object_mode

    def load_model(self):
        return self

    def supports_json_mode(self) -> bool:
        return True

    def get_model_name(self) -> str:
        return self.model_name

    def _api_key(self) -> str:
        api = (self.api or "").lower().strip()
        url = (self.url or "").lower()
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        openai_key = os.getenv("OPENAI_API_KEY", "")
        service_key = os.getenv("LLM_SERVICE_KEY", "")
        if api == "openrouter" or "openrouter.ai" in url:
            return openrouter_key or openai_key or service_key
        if api == "openai":
            return openai_key or openrouter_key or service_key
        return openrouter_key or openai_key or service_key

    def _headers(self) -> dict[str, str]:
        key = self._api_key()
        if not key:
            return {}
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        if "openrouter.ai" in self.url.lower():
            headers["HTTP-Referer"] = "https://github.com/aimclub/OSA"
            headers["X-Title"] = "OSA README benchmark"
        return headers

    def _post_chat(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: Optional[dict[str, str]] = None,
    ) -> str:
        headers = self._headers()
        if not headers:
            raise RuntimeError(
                "Missing judge API key. Set OPENROUTER_API_KEY, OPENAI_API_KEY, or LLM_SERVICE_KEY."
            )
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }
        if response_format and self.use_json_object_mode:
            payload["response_format"] = response_format

        response = requests.post(
            f"{self.url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.request_timeout,
        )
        if response.status_code == 200:
            return (response.json()["choices"][0]["message"]["content"] or "").strip()

        if response_format and self.use_json_object_mode:
            payload.pop("response_format", None)
            response = requests.post(
                f"{self.url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.request_timeout,
            )
            if response.status_code == 200:
                return (response.json()["choices"][0]["message"]["content"] or "").strip()

        raise RuntimeError(f"Judge LLM HTTP {response.status_code}: {response.text[:500]}")

    def generate(self, prompt: str) -> str:
        return self._post_chat([{"role": "user", "content": prompt}])

    def generate_with_schema(self, prompt: str, schema: Optional[Type[BaseModel]] = None, **kwargs):
        if schema is None:
            return self.generate(prompt)
        raw = self._post_chat(
            [
                {
                    "role": "system",
                    "content": "You are a strict JSON generator. Reply with one JSON object only.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        cleaned = _strip_markdown_json_fence(raw)
        try:
            return schema.model_validate_json(cleaned)
        except Exception:
            return schema.model_validate(json.loads(cleaned))

    async def a_generate(self, prompt: str, schema=None):
        return await asyncio.to_thread(self.generate, prompt)

    async def a_generate_with_schema(self, prompt: str, schema=None, **kwargs):
        if schema is None:
            return await self.a_generate(prompt)
        return await asyncio.to_thread(self.generate_with_schema, prompt, schema)


def build_readme_quality_metric(judge: dict[str, Any]) -> GEval:
    assess_model = CustomLLM(
        api=judge.get("api", "openrouter"),
        model=judge.get("model", "gpt-4.1"),
        url=judge.get("base_url", "https://openrouter.ai/api/v1"),
    )
    return GEval(
        name="Readme quality",
        criteria=README_QUALITY_CRITERIA,
        evaluation_steps=README_QUALITY_STEPS,
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=assess_model,
        verbose_mode=False,
        async_mode=False,
    )


def _read_text(path: Path, *, limit: int | None = None) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:limit] if limit else text


def evaluate_outputs(paths: BenchmarkPaths, experiment: dict[str, Any], *, reset: bool = False) -> pd.DataFrame:
    status_path = paths.experiment_dir / "tool_status.csv"
    if not status_path.is_file():
        raise FileNotFoundError(f"Tool status file not found: {status_path}")

    eval_path = paths.evaluation_dir / "eval_rows.csv"
    old = pd.read_csv(eval_path) if eval_path.is_file() and not reset else pd.DataFrame()
    completed = set()
    if not old.empty:
        completed = set(zip(old["tool"], old["model"], old["repo_slug"]))

    metric = build_readme_quality_metric(experiment["judge"])
    statuses = pd.read_csv(status_path)
    rows = []

    for row in statuses.itertuples():
        key = (row.tool, row.model, row.repo_slug)
        if row.status != "done" or key in completed:
            continue

        generated_path = Path(row.output_path)
        original_candidates = list(paths.original_readmes_dir.glob(f"{row.repo_slug}_original_README*"))
        structure_path = paths.structures_dir / f"{row.repo_slug}_struct.json"
        if not generated_path.is_file() or not original_candidates or not structure_path.is_file():
            continue

        try:
            generated_readme = _read_text(generated_path)
            original_readme = _read_text(original_candidates[0])
            repo_structure = _read_text(structure_path)
            test_case = LLMTestCase(
                input="",
                actual_output=json.dumps(
                    {
                        "readme": generated_readme,
                        "repo_structure": repo_structure,
                    },
                    ensure_ascii=False,
                ),
                expected_output=original_readme,
            )
            metric.measure(test_case)
            rows.append(
                {
                    "tool": row.tool,
                    "model": row.model,
                    "model_label": model_label(row.model),
                    "repo_slug": row.repo_slug,
                    "score": metric.score,
                    "reason": getattr(metric, "reason", ""),
                    "success": getattr(metric, "success", None),
                    "output_path": row.output_path,
                }
            )
        except Exception as exc:
            logger.exception("GEval failed for %s/%s/%s", row.tool, row.model, row.repo_slug)
            rows.append(
                {
                    "tool": row.tool,
                    "model": row.model,
                    "model_label": model_label(row.model),
                    "repo_slug": row.repo_slug,
                    "score": None,
                    "reason": f"GEval failed: {exc}",
                    "success": False,
                    "output_path": row.output_path,
                }
            )

    new = pd.DataFrame(rows)
    all_rows = pd.concat([old, new], ignore_index=True) if not old.empty else new
    paths.evaluation_dir.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(eval_path, index=False)

    if not all_rows.empty:
        summary = (
            all_rows.groupby(["tool", "model_label"], dropna=False)
            .agg(
                mean_score=("score", "mean"),
                evaluated=("score", "count"),
                rows=("repo_slug", "count"),
            )
            .reset_index()
        )
        summary.to_csv(paths.evaluation_dir / "final_summary.csv", index=False)

    return all_rows

