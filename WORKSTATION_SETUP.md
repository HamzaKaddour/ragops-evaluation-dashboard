# Workstation Setup Checklist

Use this checklist on the Ubuntu workstation.

## A. System check

```bash
nvidia-smi
python3 --version
git --version
```

## B. Extract repo

```bash
unzip ragops-evaluation-dashboard.zip
cd ragops-evaluation-dashboard
```

## C. Virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## D. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

## E. Verify PyTorch GPU support

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If CUDA prints `False`, the project still runs on CPU. For the A6000 workstation, install a CUDA-enabled PyTorch wheel if the default installation does not detect the GPU; follow the current PyTorch installation selector for your CUDA environment.

## F. Retrieval smoke test

```bash
python scripts/run_real_evaluation.py
```

## G. Qwen smoke test

```bash
python scripts/test_local_llm.py
```

## H. Full interactive RAG

```bash
python scripts/query_cli.py
```

## I. API

```bash
uvicorn api.app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## J. Tests

```bash
pytest -q
```

## K. Normal future startup

After the one-time setup, your normal workflow is only:

```bash
cd /path/to/ragops-evaluation-dashboard
source .venv/bin/activate
uvicorn api.app:app --reload
```

When finished, stop with `Ctrl+C` and optionally run:

```bash
deactivate
```
