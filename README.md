# AUPIMO

AUPIMO stands for **A**rea **U**nder the **P**er-**IM**age **O**verlap curve (pronounced a-u-pee-mo).

Official implementation of the paper **AUPIMO: Redefining Visual Anomaly Detection Benchmarks with High Speed and Low Tolerance**.

Arxiv: COMING UP

## Installation

```bash
git clone git@github.com:jpcbertoldo/aupimo.git
cd aupimo
pip install .
```

## Tutorials

COMING UP

## Development

For development, install the requirements in `requirements/dev.txt` and install pre-commit hooks:

```bash
git clone git@github.com:jpcbertoldo/aupimo.git
cd aupimo
pip install -e .  # `-e` is for 'editable' mode
pip install -r requirements/dev.txt
pre-commit install
```

then run the tests in `tests/` with `pytest`

```bash
pytest tests/
```
