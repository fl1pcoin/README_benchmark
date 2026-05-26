import json
import os
import re
import shutil
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


def _opencode_model(model: str) -> str:
    model = model.strip()
    if model.startswith("openrouter/"):
        return model
    return f"openrouter/{model}"


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


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)


def _clean_markdown(text: str) -> str:
    cleaned = _strip_ansi(text).strip()
    if cleaned.startswith("```markdown"):
        cleaned = cleaned.removeprefix("```markdown").strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").strip()
    if cleaned.endswith("```"):
        cleaned = cleaned.removesuffix("```").strip()
    return cleaned


def _looks_like_readme(text: str) -> bool:
    cleaned = _clean_markdown(text)
    if not cleaned:
        return False
    first_line = next((line.strip() for line in cleaned.splitlines() if line.strip()), "")
    if not first_line.startswith("# "):
        return False
    narrative_starts = (
        "great!",
        "to generate",
        "i'll",
        "i will",
        "let's",
        "next,",
        "here's",
    )
    return not first_line.lower().startswith(narrative_starts)


def _opencode_exe() -> str:
    found = shutil.which("opencode") or shutil.which("opencode.cmd")
    return found or ("opencode.cmd" if os.name == "nt" else "opencode")


def _opencode_repo_context(repo_dir: Path, *, max_chars: int = 28000) -> str:
    skip_dirs = {
        ".git",
        ".idea",
        ".venv",
        "__pycache__",
        "node_modules",
        "dist",
        "build",
        ".pytest_cache",
    }
    preferred = {
        "README.md",
        "README.rst",
        "README.txt",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements.txt",
        "package.json",
        "Cargo.toml",
        "go.mod",
        "pom.xml",
        "Makefile",
        "Dockerfile",
        "LICENSE",
    }
    files = []
    for path in repo_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_dir)
        if any(part in skip_dirs for part in rel.parts):
            continue
        rel_text = rel.as_posix()
        if path.name in preferred or len(files) < 80:
            files.append(rel_text)
    files = sorted(dict.fromkeys(files))
    parts = ["Repository file tree excerpt:", *files[:120], "", "Key file excerpts:"]
    remaining = max_chars - sum(len(part) + 1 for part in parts)
    for rel_text in files:
        if remaining <= 0:
            break
        path = repo_dir / rel_text
        if path.name not in preferred and not rel_text.lower().endswith((".py", ".js", ".ts", ".md")):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        excerpt = text[:3500]
        block = f"\n--- {rel_text} ---\n{excerpt}\n"
        if len(block) > remaining:
            block = block[:remaining]
        parts.append(block)
        remaining -= len(block)
    return "\n".join(parts)[:max_chars]


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


def _kill_process_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        proc.kill()


def _run_live(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    env: dict[str, str] | None = None,
) -> tuple[str, str, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_env = env or _env()
    stdout_path = log_path.with_suffix(log_path.suffix + ".stdout.tmp")
    start = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file, stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_file:
        log_file.write("COMMAND:\n" + _redact(" ".join(str(part) for part in cmd), run_env) + "\n\nOUTPUT:\n")
        log_file.flush()
        try:
            proc = subprocess.Popen(
                [str(part) for part in cmd],
                cwd=str(cwd),
                env=run_env,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=log_file,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            while proc.poll() is None:
                if time.time() - start > timeout_sec:
                    _kill_process_tree(proc)
                    log_file.write(f"\n\nTIMEOUT after {timeout_sec} seconds\n")
                    log_file.flush()
                    return "timeout", f"timeout after {timeout_sec} seconds", stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
                time.sleep(1)
            if proc.returncode != 0:
                log_file.write(f"\n\nEXIT CODE: {proc.returncode}\n")
                log_file.flush()
                text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
                return "failed", f"exit {proc.returncode}", _redact(text, run_env)
            text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
            return "done", "", _redact(text, run_env)
        except Exception as exc:
            log_file.write(f"\n\nERROR: {exc}\n")
            log_file.flush()
            return "failed", str(exc), ""


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


def _readmeready_batch_code() -> str:
    return textwrap.dedent(
        r"""
        import importlib
        import json
        import shutil
        import sys
        import time
        import traceback
        import types
        from datetime import datetime, timezone
        from pathlib import Path

        from ReadMeReady_eval.readme_ready.types import AutodocReadmeConfig, AutodocRepoConfig, AutodocUserConfig, LLMModels

        def utc_now_iso():
            return datetime.now(timezone.utc).isoformat(timespec="seconds")

        model_value = sys.argv[1]
        manifest_path = Path(sys.argv[2])
        status_path = Path(sys.argv[3])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        model = next((m for m in LLMModels if m.value == model_value), None)
        if model is None:
            raise SystemExit(f"Unsupported ReadMeReady model: {model_value}")

        def fallback_from_buffer(content, mime=True):
            try:
                content.decode("utf-8")
                return "text/plain" if mime else "text"
            except Exception:
                return "application/octet-stream" if mime else "binary"

        sys.modules.setdefault("magic", types.SimpleNamespace(from_buffer=fallback_from_buffer))

        embeddings_cache = {}

        def stable_embeddings(_model, device):
            from langchain_huggingface import HuggingFaceEmbeddings
            if device == "auto":
                device = None
            key = (device or "cpu")
            if key not in embeddings_cache:
                embeddings_cache[key] = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-mpnet-base-v2",
                    model_kwargs={"device": device},
                    encode_kwargs={"normalize_embeddings": True},
                )
            return embeddings_cache[key]

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
        convert_mod = importlib.import_module("ReadMeReady_eval.readme_ready.index.convert_json_to_markdown")
        convert_mod.traverse_file_system = file_only_traverse
        query_mod = importlib.import_module("ReadMeReady_eval.readme_ready.query.query")
        query_mod.get_embeddings = stable_embeddings
        query_mod.clear = lambda: None
        from ReadMeReady_eval.readme_ready.index import index
        from ReadMeReady_eval.readme_ready.query import query

        rows = []

        def flush():
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

        for item in manifest:
            started = utc_now_iso()
            start = time.time()
            repo_slug = item["repo_slug"]
            output_path = Path(item["output_path"])
            try:
                repo_root = Path(item["repo_root"])
                work_dir = Path(item["work_dir"])
                work_dir.mkdir(parents=True, exist_ok=True)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                config = AutodocRepoConfig(
                    name=repo_slug,
                    repository_url=item["repo_url"],
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
                    raise RuntimeError("ReadMeReady did not produce README_*.md")
                shutil.copy2(candidates[0], output_path)
                status = "done"
                error = ""
            except Exception:
                status = "failed"
                error = traceback.format_exc()
            rows.append(
                {
                    "repo_slug": repo_slug,
                    "status": status,
                    "started_at": started,
                    "finished_at": utc_now_iso(),
                    "duration_sec": round(time.time() - start, 3),
                    "output_path": str(output_path),
                    "error": error,
                }
            )
            flush()
        """
    )


def run_readmeready(repo_df: pd.DataFrame, paths: BenchmarkPaths, model: str, *, reset: bool = False) -> pd.DataFrame:
    label = model_label(model)
    tool_root = paths.workspace_root / "ReadMeReady_eval"
    py = _python_exe(tool_root)
    rows = []
    pending = []
    model_log = paths.logs_dir / "readmeready" / label / "_batch.log"
    tool_dir = paths.tools_dir / "readmeready" / label
    for repo in repo_df.itertuples():
        output = paths.tools_dir / "readmeready" / label / f"{repo.repo_slug}_README.md"
        log = model_log
        started = utc_now_iso()
        if output.is_file() and not reset:
            rows.append(_status_row(tool="readmeready", model=model, repo_slug=repo.repo_slug, status="done", started_at=started, output_path=output, log_path=log))
            continue
        work_dir = paths.tools_dir / "readmeready" / label / "_work" / repo.repo_slug
        pending.append(
            {
                "repo_slug": repo.repo_slug,
                "repo_url": repo.repo_url,
                "repo_root": str(paths.repositories_dir / repo.repo_slug),
                "work_dir": str(work_dir),
                "output_path": str(output),
            }
        )
    if pending:
        tool_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = tool_dir / "_batch_manifest.json"
        status_path = tool_dir / "_batch_status.json"
        manifest_path.write_text(json.dumps(pending, indent=2, ensure_ascii=False), encoding="utf-8")
        if status_path.is_file():
            status_path.unlink()
        cmd = [py, "-c", _readmeready_batch_code(), _tool_model(model), manifest_path, status_path]
        status, error = _run(
            cmd,
            cwd=paths.workspace_root,
            log_path=model_log,
            env=_env({"PYTHONPATH": str(paths.workspace_root)}),
        )
        if status_path.is_file():
            batch_rows = json.loads(status_path.read_text(encoding="utf-8"))
            for item in batch_rows:
                rows.append(
                    {
                        "tool": "readmeready",
                        "model": model,
                        "model_label": label,
                        "repo_slug": item["repo_slug"],
                        "status": item["status"],
                        "started_at": item["started_at"],
                        "finished_at": item["finished_at"],
                        "duration_sec": item["duration_sec"],
                        "output_path": item["output_path"],
                        "log_path": str(model_log),
                        "error": item.get("error", ""),
                    }
                )
        else:
            for item in pending:
                rows.append(_status_row(tool="readmeready", model=model, repo_slug=item["repo_slug"], status=status, started_at=utc_now_iso(), output_path=Path(item["output_path"]), log_path=model_log, error=error))
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


def _opencode_config(model: str) -> str:
    model_id = model.removeprefix("openrouter/")
    return json.dumps(
        {
            "$schema": "https://opencode.ai/config.json",
            "autoupdate": False,
            "snapshot": False,
            "share": "disabled",
            "provider": {
                "openrouter": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "OpenRouter",
                    "options": {
                        "baseURL": "https://openrouter.ai/api/v1",
                        "apiKey": "{env:OPENROUTER_API_KEY}",
                    },
                    "models": {
                        model_id: {
                            "name": model_id,
                        }
                    },
                }
            },
            "agent": {
                "readme-benchmark": {
                    "description": "Generate README files for benchmark repositories without modifying files.",
                    "model": _opencode_model(model),
                    "tools": {
                        "read": True,
                        "glob": True,
                        "grep": True,
                        "bash": False,
                        "edit": True,
                        "write": True,
                        "webfetch": False,
                        "websearch": False,
                        "task": False,
                        "todowrite": False,
                        "lsp": False,
                        "skill": False,
                    },
                }
            },
        },
        ensure_ascii=False,
    )


def _opencode_prompt(repo_slug: str, *, output_mode: str = "file") -> str:
    if output_mode == "stdout":
        return textwrap.dedent(
            f"""
            Generate a complete README.md for the repository in the current directory.

            Repository slug: {repo_slug}

            Requirements:
            - Inspect the repository files that are available in this working tree.
            - Do not modify, create, delete, or rename any files.
            - Return only the final README markdown in your response.
            - The first non-empty line of your response must start with "# ".
            - Do not write planning notes, progress updates, tool summaries, or commentary.
            - Include sections that are relevant to the project, such as overview, features,
              installation, usage, configuration, development, testing, and license.
            - If information is not present in the repository, avoid inventing exact claims.
            """
        ).strip()
    return textwrap.dedent(
        f"""
        Generate a complete README.md for the repository in the current directory and save it
        exactly to this file in the current directory:

        .opencode_benchmark_README.md

        Repository slug: {repo_slug}

        Requirements:
        - Inspect the repository files that are available in this working tree.
        - You may create or overwrite only .opencode_benchmark_README.md.
        - Do not modify, create, delete, or rename any other files.
        - The file content must be only the final README markdown.
        - The first non-empty line of the file must start with "# ".
        - Do not write planning notes, progress updates, tool summaries, or commentary into the file.
        - Include sections that are relevant to the project, such as overview, features,
          installation, usage, configuration, development, testing, and license.
        - If information is not present in the repository, avoid inventing exact claims.
        - After writing the file, respond with only: DONE
        """
    ).strip()


def run_opencode(repo_df: pd.DataFrame, paths: BenchmarkPaths, model: str, *, reset: bool = False) -> pd.DataFrame:
    label = model_label(model)
    output_mode = "stdout" if "gemma" in model.lower() else "file"
    tool_dir = paths.tools_dir / "opencode" / label
    tool_dir.mkdir(parents=True, exist_ok=True)
    config_path = tool_dir / "opencode.config.json"
    config_path.write_text(_opencode_config(model), encoding="utf-8")
    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    rows = []
    for repo in repo_df.itertuples():
        repo_dir = paths.repositories_dir / repo.repo_slug
        output = tool_dir / f"{repo.repo_slug}_README.md"
        generated = repo_dir / ".opencode_benchmark_README.md"
        repo_readme = repo_dir / "README.md"
        log = paths.logs_dir / "opencode" / label / f"{repo.repo_slug}.log"
        started = utc_now_iso()
        start = time.time()
        if output.is_file() and not reset:
            existing = output.read_text(encoding="utf-8", errors="replace")
            if _looks_like_readme(existing):
                rows.append(_status_row(tool="opencode", model=model, repo_slug=repo.repo_slug, status="done", started_at=started, output_path=output, log_path=log))
                continue
            output.unlink()
        if not key:
            rows.append(_status_row(tool="opencode", model=model, repo_slug=repo.repo_slug, status="skipped", started_at=started, output_path=output, log_path=log, error="OPENROUTER_API_KEY or OPENAI_API_KEY is required"))
            continue
        if not repo_dir.is_dir():
            rows.append(_status_row(tool="opencode", model=model, repo_slug=repo.repo_slug, status="failed", started_at=started, output_path=output, log_path=log, error=f"Repository directory does not exist: {repo_dir}"))
            continue
        if generated.is_file():
            generated.unlink()
        original_readme = repo_readme.read_text(encoding="utf-8", errors="replace") if repo_readme.is_file() else None
        cmd = [
            _opencode_exe(),
            "run",
            "--print-logs",
            "--log-level",
            "DEBUG",
            "--pure",
            "--dangerously-skip-permissions",
            "--model",
            _opencode_model(model),
            "--agent",
            "readme-benchmark",
            "--dir",
            repo_dir,
            "--",
            _opencode_prompt(repo.repo_slug, output_mode=output_mode),
        ]
        status, error, stdout = _run_live(
            cmd,
            cwd=paths.workspace_root,
            log_path=log,
            env=_env(
                {
                    "OPENROUTER_API_KEY": key,
                    "OPENCODE_CONFIG": str(config_path),
                    "OPENCODE_CONFIG_CONTENT": _opencode_config(model),
                    "OPENCODE_DISABLE_AUTOUPDATE": "1",
                    "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1",
                }
            ),
        )
        if status == "done":
            if generated.is_file():
                readme = _clean_markdown(generated.read_text(encoding="utf-8", errors="replace"))
            elif repo_readme.is_file() and repo_readme.read_text(encoding="utf-8", errors="replace") != (original_readme or ""):
                readme = _clean_markdown(repo_readme.read_text(encoding="utf-8", errors="replace"))
            else:
                readme = _clean_markdown(stdout)
                status = "failed"
                error = "OpenCode did not create .opencode_benchmark_README.md or modify README.md"
            if readme and _looks_like_readme(readme):
                output.write_text(readme + "\n", encoding="utf-8")
                try:
                    generated.unlink()
                except FileNotFoundError:
                    pass
            else:
                status = "failed"
                error = error or "OpenCode output does not look like a README"
            if original_readme is None:
                if repo_readme.is_file():
                    repo_readme.unlink()
            else:
                repo_readme.write_text(original_readme, encoding="utf-8")
        final_done = output.is_file() and _looks_like_readme(output.read_text(encoding="utf-8", errors="replace"))
        rows.append(
            _status_row(
                tool="opencode",
                model=model,
                repo_slug=repo.repo_slug,
                status="done" if final_done else status,
                started_at=started,
                output_path=output,
                log_path=log,
                error="" if final_done else error,
                duration_sec=time.time() - start,
            )
        )
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
        if tools.get("opencode", {}).get("enabled", False):
            status = run_opencode(repo_df, paths, model, reset=reset)
    return status
