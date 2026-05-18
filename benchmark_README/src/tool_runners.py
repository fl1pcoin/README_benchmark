import os
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .notebook_utils import BenchmarkPaths, model_label, utc_now_iso, write_osa_table


DEFAULT_TIMEOUT_SEC = 60 * 20


def _python_exe(root: Path) -> Path:
    win = root / ".venv" / "Scripts" / "python.exe"
    posix = root / ".venv" / "bin" / "python"
    if win.is_file():
        return win
    if posix.is_file():
        return posix
    return Path("python")


def _tool_model(model: str) -> str:
    return model.removeprefix("openai/")


def _join_pythonpath(*paths: Path) -> str:
    return os.pathsep.join(str(path) for path in paths)


def _env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if env.get("OPENROUTER_API_KEY"):
        env.setdefault("OPENAI_API_KEY", env["OPENROUTER_API_KEY"])
    env.setdefault("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
    env.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    if extra:
        env.update(extra)
    return env


def _redact(text: str, env: dict[str, str] | None = None) -> str:
    redacted = text
    candidates = [
        os.getenv("OPENROUTER_API_KEY", ""),
        os.getenv("OPENAI_API_KEY", ""),
        os.getenv("GIT_TOKEN", ""),
        os.getenv("GITHUB_TOKEN", ""),
        os.getenv("HF_TOKEN", ""),
    ]
    if env:
        candidates.extend(
            env.get(name, "")
            for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "GIT_TOKEN", "GITHUB_TOKEN", "HF_TOKEN")
        )
    for secret in {value for value in candidates if value}:
        redacted = redacted.replace(secret, "<REDACTED>")
    return redacted


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    env: dict[str, str] | None = None,
) -> tuple[str, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_env = env or _env()
    try:
        proc = subprocess.run(
            [str(part) for part in cmd],
            cwd=str(cwd),
            env=run_env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            encoding="utf-8",
            errors="replace",
        )
        log_path.write_text(
            "COMMAND:\n"
            + _redact(" ".join(str(part) for part in cmd), run_env)
            + "\n\nSTDOUT:\n"
            + _redact(proc.stdout, run_env)
            + "\n\nSTDERR:\n"
            + _redact(proc.stderr, run_env),
            encoding="utf-8",
        )
        if proc.returncode != 0:
            return "failed", _redact((proc.stderr or proc.stdout or f"exit {proc.returncode}").strip(), run_env)
        return "done", ""
    except subprocess.TimeoutExpired as exc:
        log_path.write_text(str(exc), encoding="utf-8")
        return "timeout", str(exc)
    except Exception as exc:
        log_path.write_text(str(exc), encoding="utf-8")
        return "failed", str(exc)


def _status_row(
    *,
    tool: str,
    model: str,
    repo_slug: str,
    status: str,
    started_at: str,
    output_path: Path,
    log_path: Path,
    error: str = "",
    duration_sec: float = 0.0,
) -> dict[str, Any]:
    return {
        "tool": tool,
        "model": model,
        "model_label": model_label(model),
        "repo_slug": repo_slug,
        "status": status,
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "duration_sec": round(duration_sec, 3),
        "output_path": str(output_path),
        "log_path": str(log_path),
        "error": error,
    }


def _append_status(paths: BenchmarkPaths, rows: list[dict[str, Any]]) -> pd.DataFrame:
    out = paths.experiment_dir / "tool_status.csv"
    old = pd.read_csv(out) if out.is_file() else pd.DataFrame()
    new = pd.DataFrame(rows)
    all_rows = pd.concat([old, new], ignore_index=True) if not old.empty else new
    if not all_rows.empty:
        all_rows = all_rows.drop_duplicates(subset=["tool", "model", "repo_slug"], keep="last")
    all_rows.to_csv(out, index=False)
    return all_rows


def run_osa(repo_df: pd.DataFrame, paths: BenchmarkPaths, model: str, *, reset: bool = False) -> pd.DataFrame:
    label = model_label(model)
    tool_dir = paths.tools_dir / "osa" / label
    raw_readmes_dir = tool_dir / "readmes"
    table_path = tool_dir / "repos_table.csv"
    log_path = paths.logs_dir / "osa" / f"{label}.log"
    tool_dir.mkdir(parents=True, exist_ok=True)
    if reset or not table_path.is_file():
        write_osa_table(repo_df, table_path)

    started = utc_now_iso()
    start = time.time()
    osa_root = paths.workspace_root / "OSA"
    cmd = [
        _python_exe(osa_root),
        "-m",
        "OSA.osa_tool.run_multi_process",
        "--table-path",
        table_path,
        "--readme",
        "--api",
        "openai",
        "--base-url",
        "https://openrouter.ai/api/v1",
        "--model",
        model,
    ]
    status, error = _run(
        cmd,
        cwd=paths.workspace_root,
        log_path=log_path,
        env=_env({"PYTHONPATH": _join_pythonpath(paths.workspace_root, osa_root / "osa_tool")}),
    )
    rows = []
    for repo in repo_df.itertuples():
        expected = raw_readmes_dir / f"{repo.repo_slug}_README.md"
        rows.append(
            _status_row(
                tool="osa",
                model=model,
                repo_slug=repo.repo_slug,
                status="done" if expected.is_file() else status,
                started_at=started,
                output_path=expected,
                log_path=log_path,
                error="" if expected.is_file() else error,
                duration_sec=time.time() - start,
            )
        )
    return _append_status(paths, rows)


def _readmeready_code() -> str:
    return textwrap.dedent(
        r"""
        import os
        import shutil
        import sys
        from pathlib import Path

        import importlib
        import sys
        import types
        from ReadMeReady_eval.readme_ready.types import AutodocReadmeConfig, AutodocRepoConfig, AutodocUserConfig, LLMModels

        # python-magic needs native libmagic on Windows. For this benchmark we only need
        # a conservative text-file detector, so install a tiny fallback before modules import it.
        def fallback_from_buffer(content, mime=True):
            try:
                content.decode("utf-8")
                return "text/plain" if mime else "text"
            except Exception:
                return "application/octet-stream" if mime else "binary"

        sys.modules.setdefault("magic", types.SimpleNamespace(from_buffer=fallback_from_buffer))

        repo_root = Path(sys.argv[1])
        repo_url = sys.argv[2]
        repo_slug = sys.argv[3]
        model_value = sys.argv[4]
        work_dir = Path(sys.argv[5])
        output_path = Path(sys.argv[6])

        model = next((m for m in LLMModels if m.value == model_value), None)
        if model is None:
            raise SystemExit(f"Unsupported ReadMeReady model: {model_value}")

        def stable_embeddings(_model, device):
            from langchain_huggingface import HuggingFaceEmbeddings
            if device == "auto":
                device = None
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )

        def file_only_traverse(params):
            from pathlib import Path
            from ReadMeReady_eval.readme_ready.types import ProcessFileParams
            import fnmatch

            root = Path(params.input_path)

            def should_ignore(name):
                return any(fnmatch.fnmatch(name, pattern) for pattern in params.ignore)

            for entry in root.rglob("*"):
                if any(should_ignore(part) for part in entry.relative_to(root).parts):
                    continue
                if not entry.is_file():
                    continue
                try:
                    data = entry.read_bytes()
                    if fallback_from_buffer(data, mime=True).startswith("text/") and params.process_file:
                        params.process_file(
                            ProcessFileParams(
                                entry.name,
                                str(entry),
                                params.project_name,
                                params.content_type,
                                params.file_prompt,
                                params.target_audience,
                                params.link_hosted,
                            )
                        )
                except Exception:
                    continue

        llm_utils = importlib.import_module("ReadMeReady_eval.readme_ready.utils.llm_utils")
        llm_utils.get_embeddings = stable_embeddings
        from ReadMeReady_eval.readme_ready.types import LLMModelDetails
        llm_utils.models[model] = LLMModelDetails(
            name=model,
            input_cost_per_1k_tokens=0.0,
            output_cost_per_1k_tokens=0.0,
            max_length=16000,
            llm=llm_utils.get_openai_chat_model(
                model.value,
                temperature=0.1,
                streaming=False,
                model_kwargs={"frequency_penalty": 0.0, "presence_penalty": 0.0},
            ),
            input_tokens=0,
            output_tokens=0,
            succeeded=0,
            failed=0,
            total=0,
        )
        traverse_mod = importlib.import_module("ReadMeReady_eval.readme_ready.utils.traverse_file_system")
        traverse_mod.traverse_file_system = file_only_traverse
        process_repo_mod = importlib.import_module("ReadMeReady_eval.readme_ready.index.process_repository")
        process_repo_mod.traverse_file_system = file_only_traverse
        process_repo_mod.models = llm_utils.models
        process_repo_mod.select_model = lambda prompts, llms, models, priority: models.get(llms[0])
        vector_store_mod = importlib.import_module("ReadMeReady_eval.readme_ready.index.create_vector_store")
        vector_store_mod.get_embeddings = stable_embeddings
        vector_store_mod.LLMModels = LLMModels
        convert_mod = importlib.import_module("ReadMeReady_eval.readme_ready.index.convert_json_to_markdown")
        convert_mod.traverse_file_system = file_only_traverse
        query_mod = importlib.import_module("ReadMeReady_eval.readme_ready.query.query")
        query_mod.get_embeddings = stable_embeddings
        query_mod.clear = lambda: None
        from ReadMeReady_eval.readme_ready.index import index
        from ReadMeReady_eval.readme_ready.query import query

        work_dir.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        config = AutodocRepoConfig(
            name=repo_slug,
            repository_url=repo_url,
            root=str(repo_root),
            output=str(work_dir),
            llms=[model],
            priority=None,
            max_concurrent_calls=8,
            add_questions=False,
            ignore=[".*", "*package-lock.json", "*package.json", "node_modules", "*dist*", "*build*", "*test*", "*.svg", "*.md", "*.mdx", "*.toml"],
            file_prompt="Write a concise technical explanation of this code file in markdown.",
            folder_prompt="Write a concise technical explanation of this folder in markdown.",
            chat_prompt="",
            content_type="code",
            target_audience="developer",
            link_hosted=True,
            peft_model_path=None,
            device="cpu",
        )
        user = AutodocUserConfig(llms=[model])
        readme = AutodocReadmeConfig("Description,Requirements,Installation,Usage,Contributing,License")
        index.index(config)
        query.generate_readme(config, user, readme)
        candidates = sorted((work_dir / "docs" / "data").glob("README_*.md"))
        if not candidates:
            raise SystemExit("ReadMeReady did not produce README_*.md")
        shutil.copy2(candidates[0], output_path)
        """
    )


def run_readmeready(repo_df: pd.DataFrame, paths: BenchmarkPaths, model: str, *, reset: bool = False) -> pd.DataFrame:
    label = model_label(model)
    tool_root = paths.workspace_root / "ReadMeReady_eval"
    py = _python_exe(tool_root)
    rows = []
    for repo in repo_df.itertuples():
        output = paths.tools_dir / "readmeready" / label / f"{repo.repo_slug}_README.md"
        log = paths.logs_dir / "readmeready" / label / f"{repo.repo_slug}.log"
        started = utc_now_iso()
        start = time.time()
        if output.is_file() and not reset:
            rows.append(_status_row(tool="readmeready", model=model, repo_slug=repo.repo_slug, status="done", started_at=started, output_path=output, log_path=log))
            continue
        work_dir = paths.tools_dir / "readmeready" / label / "_work" / repo.repo_slug
        cmd = [py, "-c", _readmeready_code(), paths.repositories_dir / repo.repo_slug, repo.repo_url, repo.repo_slug, _tool_model(model), work_dir, output]
        status, error = _run(
            cmd,
            cwd=paths.workspace_root,
            log_path=log,
            env=_env({"PYTHONPATH": str(paths.workspace_root)}),
        )
        rows.append(_status_row(tool="readmeready", model=model, repo_slug=repo.repo_slug, status=status if output.is_file() else status, started_at=started, output_path=output, log_path=log, error=error, duration_sec=time.time() - start))
    return _append_status(paths, rows)


def _larch_code() -> str:
    return textwrap.dedent(
        r"""
        import sys
        from pathlib import Path

        import larch_eval.larch as larch
        from larch_eval.larch.context_creator import ContextCreatorName, Directory
        from larch_eval.larch.postprocessor import postprocess

        repo_dir = Path(sys.argv[1])
        repo_slug = sys.argv[2]
        model = sys.argv[3]
        api_key = sys.argv[4]
        seed_path = Path(sys.argv[5])
        output_path = Path(sys.argv[6])

        max_generation_length = 3200
        slack_length = 4

        larch.generator.init_models([model], api_key, {})
        generator = larch.generator.AVAILABLE_MODELS[model]
        # The original "entrypoint" context creator uses signal.SIGALRM, which is
        # unavailable on Windows. Use file_names as a Windows-safe LArch context.
        context_creator = ContextCreatorName("file_names").get_module()
        if hasattr(context_creator, "init"):
            context_creator.init()
        dir_tree = Directory.from_directory(str(repo_dir))
        context = context_creator.create_context(
            dir_tree,
            use_prompt=not generator.is_pretrained,
            tokenizer=generator.tokenizer,
            max_tokens=generator.max_context_length - max_generation_length - slack_length,
            project_name=repo_slug,
        )
        prompt = seed_path.read_text(encoding="utf-8")
        generated, diff = generator.generate(context, prompt, max_length=max_generation_length)
        generated, _ = postprocess(generated, diff)
        if generated.startswith(prompt):
            generated = generated[len(prompt):].lstrip()
        generated = generated.strip()
        if generated.startswith("```markdown"):
            generated = generated.removeprefix("```markdown").strip()
        if generated.startswith("```"):
            generated = generated.removeprefix("```").strip()
        if generated.endswith("```"):
            generated = generated.removesuffix("```").strip()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(generated, encoding="utf-8")
        """
    )


def run_larch(repo_df: pd.DataFrame, paths: BenchmarkPaths, model: str, *, reset: bool = False) -> pd.DataFrame:
    label = model_label(model)
    tool_root = paths.workspace_root / "larch_eval"
    py = _python_exe(tool_root)
    seed = paths.tools_dir / "larch" / "seed_prompt.md"
    seed.parent.mkdir(parents=True, exist_ok=True)
    seed.write_text(
        "# README\n\nGenerate a complete README for this repository. Include overview, installation, usage, configuration, development, testing, and license sections.\n",
        encoding="utf-8",
    )
    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    rows = []
    for repo in repo_df.itertuples():
        output = paths.tools_dir / "larch" / label / f"{repo.repo_slug}_README.md"
        log = paths.logs_dir / "larch" / label / f"{repo.repo_slug}.log"
        started = utc_now_iso()
        start = time.time()
        if output.is_file() and not reset:
            rows.append(_status_row(tool="larch", model=model, repo_slug=repo.repo_slug, status="done", started_at=started, output_path=output, log_path=log))
            continue
        if not key:
            rows.append(_status_row(tool="larch", model=model, repo_slug=repo.repo_slug, status="skipped", started_at=started, output_path=output, log_path=log, error="OPENROUTER_API_KEY or OPENAI_API_KEY is required"))
            continue
        cmd = [py, "-c", _larch_code(), paths.repositories_dir / repo.repo_slug, repo.repo_slug, _tool_model(model), key, seed, output]
        status, error = _run(
            cmd,
            cwd=paths.workspace_root,
            log_path=log,
            env=_env({"PYTHONPATH": str(paths.workspace_root)}),
        )
        rows.append(_status_row(tool="larch", model=model, repo_slug=repo.repo_slug, status="done" if output.is_file() else status, started_at=started, output_path=output, log_path=log, error="" if output.is_file() else error, duration_sec=time.time() - start))
    return _append_status(paths, rows)


def run_selected_tools(repo_df: pd.DataFrame, paths: BenchmarkPaths, experiment: dict[str, Any]) -> pd.DataFrame:
    reset = bool(experiment.get("reset", {}).get("tool_outputs", False))
    tools = experiment.get("tools", {})
    status = pd.DataFrame()
    for model in experiment.get("models", []):
        if tools.get("osa", {}).get("enabled", False):
            status = run_osa(repo_df, paths, model, reset=reset)
        if tools.get("readmeready", {}).get("enabled", False):
            status = run_readmeready(repo_df, paths, model, reset=reset)
        if tools.get("larch", {}).get("enabled", False):
            status = run_larch(repo_df, paths, model, reset=reset)
    return status
