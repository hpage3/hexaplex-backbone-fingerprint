# Reproducibility

This repository contains exploratory research work plus an additive publication track. The publication track is the clean entry point for manuscript-supporting computational results.

## Python Environment

The current workflow has been run with the repository virtual environment:

```powershell
.\.venv\Scripts\python.exe
```

Use the Python version already captured by the local virtual environment unless rebuilding from scratch is required.

## Create Or Use The Virtual Environment

If `.venv` already exists:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

If a fresh environment is needed:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

Install any additional development requirements documented in `pyproject.toml` or current project notes before rerunning the full test suite.

## Run Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Run The Manuscript Reproducibility Pipeline

```powershell
.\.venv\Scripts\python.exe scripts\run_manuscript_reproducibility_pipeline.py
```

Expected manuscript-track outputs:

- `outputs/manuscript/metrics/manuscript_item_status.csv`
- `outputs/manuscript/metrics/manuscript_surviving_family_summary.csv`
- `outputs/manuscript/reports/manuscript_reproducibility_summary.md`

The pipeline also copies selected stable reports and metrics into `outputs/manuscript/`.

## What Publication Track Means

The publication track is a documented, additive path through the current repo that identifies which code/data/outputs support manuscript-facing computational claims. It does not delete, move, rename, or hide exploratory scripts and outputs.

## What Remains Outside The Publication Track

Exploratory scans, prototype builders, intermediate debugging outputs, and speculative analyses remain in the repo. They may be scientifically useful but are not automatically manuscript-facing.

## Status Terms

- `reused_existing_output`: the pipeline found and copied a stable existing output.
- `generated_and_copied`: the pipeline reran a lightweight summary script and copied its outputs.
- `missing_data`: an expected script or output was absent; no value was invented.

Interpret `missing_data` as a reproducibility gap, not as negative scientific evidence.
