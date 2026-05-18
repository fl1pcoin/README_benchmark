# VKR README Evaluation

Репозиторий содержит окружение для экспериментальной оценки качества README, сгенерированных разными инструментами:

- `OSA`
- `LArch`
- `ReadMeReady`

Главная точка запуска находится в `benchmark_README/README_benchmark.ipynb`. Ноутбук подготавливает датасет, клонирует тестовые репозитории, сохраняет оригинальные README и структуру проектов, запускает выбранные инструменты и оценивает полученные README через `DeepEval GEval`.

## Структура проекта

```text
VKR_README_EVAL/
  benchmark_README/
    data/repo_list.csv
    results/
    src/
    README_benchmark.ipynb
    requirements.txt
  OSA/
  larch_eval/
  ReadMeReady_eval/
  .github/
  README.md
```

`benchmark_README` содержит только orchestration-код эксперимента. Сами инструменты лежат в отдельных директориях и используют собственные виртуальные окружения, чтобы зависимости не конфликтовали.

## Что нужно установить

Рекомендуется использовать Windows + PowerShell, так как текущие пути и виртуальные окружения настроены под Windows.

### 1. Клонировать репозиторий

```powershell
git clone https://github.com/fl1pcoin/README_benchmark
```

### 1. Корневое окружение для ноутбука

Из корня проекта:

```powershell
cd README_benchmark
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\benchmark_README\requirements.txt
python -m ipykernel install --user --name vkr-readme-eval --display-name "README_benchmark"
```

В Jupyter/IDE выберите kernel `README_benchmark`.

### 2. Окружения инструментов

Каждый инструмент должен иметь свое `.venv`:

```text
OSA/.venv/
larch_eval/.venv/
ReadMeReady_eval/.venv/
```

Если окружения уже есть, переустанавливать их не нужно. Если создаете заново:

```powershell
cd README_benchmark\OSA
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
deactivate

cd README_benchmark\larch_eval
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
deactivate

cd README_benchmark\ReadMeReady_eval
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
deactivate
```

## Настройка ключей

Создайте файл `benchmark_README/.env`:

```env
OPENROUTER_API_KEY=...
OPENAI_API_KEY=...
OPENAI_API_BASE=https://openrouter.ai/api/v1
OPENAI_BASE_URL=https://openrouter.ai/api/v1
GIT_TOKEN=...
GITHUB_TOKEN=...
HF_TOKEN=...
```

Пояснения:

- `OPENROUTER_API_KEY` нужен для запуска моделей через OpenRouter.
- `OPENAI_API_KEY` можно заполнить тем же значением, что и `OPENROUTER_API_KEY`, потому что часть инструментов ожидает именно это имя переменной.
- `OPENAI_API_BASE` и `OPENAI_BASE_URL` указывают OpenAI-compatible клиентам на OpenRouter.
- `GIT_TOKEN` или `GITHUB_TOKEN` нужны для GitHub API и снижения риска rate limit.
- `HF_TOKEN` нужен для HuggingFace-зависимостей ReadMeReady и LArch.

## Датасет

Список репозиториев лежит в:

```text
benchmark_README/data/repo_list.csv
```

Ожидаемые колонки:

- `repo_url`
- `repo_name`
- `commit_sha`
- `commit_date`

Эксперимент клонирует каждый репозиторий, делает checkout на `commit_sha`, сохраняет исходный README и JSON-структуру репозитория.

## Как запустить эксперимент

Откройте:

```text
benchmark_README/README_benchmark.ipynb
```

В первой ячейке находится конфигурация:

```python
EXPERIMENT = {
    "run_name": "default_run",
    "data_csv": "data/repo_list.csv",
    "tools": {
        "osa": {"enabled": True},
        "readmeready": {"enabled": True},
        "larch": {"enabled": True, "mode": "local"},
    },
    "models": [
        "openai/gpt-4.1",
        "anthropic/claude-sonnet-4",
        "google/gemma-3-27b-it",
    ],
    "judge": {
        "api": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "gpt-4.1",
    },
    "reset": {
        "repos": False,
        "tool_outputs": False,
        "evaluation": False,
    },
    "limit_repos": None,
}
```

Для полного запуска оставьте `limit_repos = None`. Для дешевого теста:

```python
"limit_repos": 1
```

Для перегенерации результатов инструментов:

```python
"reset": {
    "repos": False,
    "tool_outputs": True,
    "evaluation": False,
}
```

После настройки выполните все ячейки ноутбука

## Где лежат результаты

Все артефакты пишутся в:

```text
benchmark_README/results/<run_name>/
```

Основные файлы:

```text
preflight_status.csv
tool_status.csv
evaluation/eval_rows.csv
evaluation/final_summary.csv
```

Сгенерированные README:

```text
tools/osa/<model_label>/readmes/<repo>_README.md
tools/larch/<model_label>/<repo>_README.md
tools/readmeready/<model_label>/<repo>_README.md
```

Логи:

```text
logs/osa/<model_label>.log
logs/larch/<model_label>/<repo>.log
logs/readmeready/<model_label>/_batch.log
```

## Как работает оценка README

Оценка выполняется через `DeepEval GEval`..

`ACTUAL_OUTPUT` передается как JSON:

```json
{
  "readme": "сгенерированный README",
  "repo_structure": "JSON-структура репозитория"
}
```

`EXPECTED_OUTPUT` — оригинальный README из репозитория.

GEval проверяет:

1. Соответствует ли README структуре репозитория.
2. Дает ли README ясное описание назначения проекта.
3. Есть ли инструкции установки и настройки.
4. Есть ли понятные примеры использования.
5. Указаны ли зависимости или требования.
6. Насколько README читаемый, структурированный и понятный.

Результаты оценки сохраняются в:

```text
benchmark_README/results/<run_name>/evaluation/eval_rows.csv
benchmark_README/results/<run_name>/evaluation/final_summary.csv
```


## Особенности инструментов

### OSA

Runner запускает OSA через:

```text
OSA/.venv/Scripts/python.exe -m OSA.osa_tool.run_multi_process
```

Для OSA задается `PYTHONPATH`, включающий корень проекта и `OSA/osa_tool`, потому что в текущем коде есть смешанные imports вида `OSA.osa_tool...` и `config.settings`.

### LArch

LArch запускается без endpoint, в local mode через внутренний Python API. На Windows используется context creator `file_names`, потому что стандартный `entrypoint` context creator использует `signal.SIGALRM`, которого нет в Windows.

### ReadMeReady

ReadMeReady запускается batch-режимом: один subprocess на модель обрабатывает все выбранные репозитории. Это нужно, чтобы HuggingFace embeddings загружались в память один раз на model run, а не заново для каждого репозитория.

На Windows runner также обходит две практические проблемы:

- отсутствие нативного `libmagic`;
- ошибку `prompt_toolkit.clear()` в subprocess без Windows console buffer.
