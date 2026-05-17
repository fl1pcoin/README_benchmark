import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd


REQUIRED_REPO_COLUMNS = {"repo_url", "repo_name", "commit_sha"}


@dataclass(frozen=True)
class BenchmarkPaths:
    benchmark_root: Path
    workspace_root: Path
    experiment_dir: Path
    repositories_dir: Path
    original_readmes_dir: Path
    structures_dir: Path
    logs_dir: Path
    tools_dir: Path
    evaluation_dir: Path


def benchmark_root() -> Path:
    return Path(__file__).resolve().parent.parent


def workspace_root() -> Path:
    return benchmark_root().parent


def model_label(model: str) -> str:
    label = model.strip().split("/")[-1]
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)


def repository_slug(repo_url: str, repo_name: str | None = None) -> str:
    if repo_name and "/" in repo_name:
        name = repo_name.split("/")[-1]
    elif repo_name:
        name = repo_name
    else:
        name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_paths(run_name: str) -> BenchmarkPaths:
    root = benchmark_root()
    exp = root / "results" / run_name
    return BenchmarkPaths(
        benchmark_root=root,
        workspace_root=root.parent,
        experiment_dir=exp,
        repositories_dir=exp / "repositories",
        original_readmes_dir=exp / "original_readmes",
        structures_dir=exp / "repo_structures",
        logs_dir=exp / "logs",
        tools_dir=exp / "tools",
        evaluation_dir=exp / "evaluation",
    )


def ensure_directories(paths: BenchmarkPaths) -> None:
    for directory in (
        paths.experiment_dir,
        paths.repositories_dir,
        paths.original_readmes_dir,
        paths.structures_dir,
        paths.logs_dir,
        paths.tools_dir,
        paths.evaluation_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def load_env_file(env_path: Path | None = None) -> None:
    path = env_path or benchmark_root() / ".env"
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
    if os.environ.get("OPENROUTER_API_KEY"):
        os.environ.setdefault("OPENAI_API_KEY", os.environ["OPENROUTER_API_KEY"])


def read_repo_table(data_csv: str | Path) -> pd.DataFrame:
    path = Path(data_csv)
    if not path.is_absolute():
        path = benchmark_root() / path
    df = pd.read_csv(path)
    missing = REQUIRED_REPO_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Repository table is missing columns: {sorted(missing)}")
    df = df.copy()
    df["repository"] = df["repo_url"]
    df["repo_slug"] = [
        repository_slug(row.repo_url, row.repo_name) for row in df.itertuples()
    ]
    return df


def _git(
    args: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
        check=check,
        encoding="utf-8",
        errors="replace",
    )


def _clone_url_with_token(repo_url: str) -> str:
    token = os.getenv("GIT_TOKEN") or os.getenv("GITHUB_TOKEN") or ""
    if not token or not repo_url.startswith("https://github.com/"):
        return repo_url
    return repo_url.replace("https://github.com/", f"https://oauth2:{token}@github.com/")


def _checkout_commit(repo_dir: Path, commit_sha: str) -> None:
    sha = str(commit_sha or "").strip()
    if not sha:
        return
    proc = _git(["checkout", "--force", sha], cwd=repo_dir, check=False)
    if proc.returncode == 0:
        return
    for args in (["fetch", "origin", sha], ["fetch", "--depth", "1", "origin", sha]):
        _git(args, cwd=repo_dir, check=False)
        proc = _git(["checkout", "--force", sha], cwd=repo_dir, check=False)
        if proc.returncode == 0:
            return
    if (repo_dir / ".git" / "shallow").is_file():
        _git(["fetch", "--unshallow"], cwd=repo_dir, check=False)
        _git(["fetch", "origin", sha], cwd=repo_dir, check=False)
        proc = _git(["checkout", "--force", sha], cwd=repo_dir, check=False)
        if proc.returncode == 0:
            return
    raise RuntimeError((proc.stderr or proc.stdout or f"checkout failed for {sha}").strip())


def ensure_repo_at_commit(row: pd.Series, repositories_dir: Path, *, reset: bool = False) -> Path:
    repo_dir = repositories_dir / row["repo_slug"]
    if reset and repo_dir.exists():
        shutil.rmtree(repo_dir)
    if repo_dir.exists() and (repo_dir / ".git").is_dir():
        _git(["fetch", "origin"], cwd=repo_dir, check=False)
    else:
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        _git(["clone", _clone_url_with_token(row["repo_url"]), str(repo_dir)])
    _checkout_commit(repo_dir, str(row.get("commit_sha", "") or ""))
    return repo_dir


def snapshot_original_readme(repo_slug: str, repo_dir: Path, original_readmes_dir: Path) -> Path | None:
    for name in ("README.md", "readme.md", "Readme.md", "README.rst", "README.txt"):
        src = repo_dir / name
        if src.is_file():
            original_readmes_dir.mkdir(parents=True, exist_ok=True)
            suffix = src.suffix or ".md"
            dst = original_readmes_dir / f"{repo_slug}_original_README{suffix}"
            shutil.copy2(src, dst)
            return dst
    return None


def _tracked_paths(repo_dir: Path) -> list[str]:
    proc = _git(["ls-tree", "-r", "--name-only", "HEAD"], cwd=repo_dir, check=False)
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def build_tree(paths: Iterable[str]) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for path in paths:
        cur = tree
        for part in path.split("/"):
            cur = cur.setdefault(part, {})
    return tree


def tree_to_dict(tree: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for name, subtree in sorted(tree.items()):
        if subtree:
            result.append({"name": name, "type": "dir", "children": tree_to_dict(subtree)})
        else:
            result.append({"name": name, "type": "file"})
    return result


def write_repo_structure_json(
    repo_slug: str,
    repo_dir: Path,
    structures_dir: Path,
    *,
    stop_words: Optional[list[str]] = None,
) -> Path:
    stop_words = stop_words or ["assets", "results", "sources", "packages", "images", "data"]
    filtered = []
    for entry in _tracked_paths(repo_dir):
        wrapped = f"/{entry}/"
        if any(f"/{stop}/" in wrapped or entry.startswith(f"{stop}/") for stop in stop_words):
            continue
        filtered.append(entry)
    structures_dir.mkdir(parents=True, exist_ok=True)
    out = structures_dir / f"{repo_slug}_struct.json"
    out.write_text(
        json.dumps(tree_to_dict(build_tree(filtered)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out


def prepare_repositories(
    repo_df: pd.DataFrame,
    paths: BenchmarkPaths,
    *,
    reset: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    rows = []
    work_df = repo_df.head(limit) if limit else repo_df
    for row in work_df.to_dict(orient="records"):
        started = utc_now_iso()
        status = "done"
        error = ""
        repo_dir = ""
        original_readme = ""
        structure_json = ""
        try:
            series = pd.Series(row)
            cloned = ensure_repo_at_commit(series, paths.repositories_dir, reset=reset)
            repo_dir = str(cloned)
            readme = snapshot_original_readme(row["repo_slug"], cloned, paths.original_readmes_dir)
            if readme:
                original_readme = str(readme)
            structure_json = str(write_repo_structure_json(row["repo_slug"], cloned, paths.structures_dir))
        except Exception as exc:
            status = "failed"
            error = str(exc)
        rows.append(
            {
                "repo_url": row["repo_url"],
                "repo_name": row["repo_name"],
                "repo_slug": row["repo_slug"],
                "commit_sha": row["commit_sha"],
                "status": status,
                "started_at": started,
                "finished_at": utc_now_iso(),
                "repo_dir": repo_dir,
                "original_readme": original_readme,
                "structure_json": structure_json,
                "error": error,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(paths.experiment_dir / "preflight_status.csv", index=False)
    return out


def write_osa_table(repo_df: pd.DataFrame, table_path: Path) -> Path:
    table_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["repository", "repo_url", "repo_name", "repo_slug", "commit_sha"]
    repo_df[[c for c in cols if c in repo_df.columns]].to_csv(table_path, index=False)
    return table_path

