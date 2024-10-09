# AUPIMO

AUPIMO stands for **A**rea **U**nder the **P**er-**IM**age **O**verlap curve (pronounced _a-u-pee-mo_).

Official implementation of the paper [**AUPIMO: Redefining Visual Anomaly Detection Benchmarks with High Speed and Low Tolerance**](https://arxiv.org/abs/2401.01984) (accepted to BMVC 2024, coming up in November 2024).

Integration in [`anomalib`](https://github.com/openvinotoolkit/anomalib) available! 

Tutorials in [`anomalib/notebooks/700_metrics`](https://github.com/openvinotoolkit/anomalib/tree/main/notebooks/700_metrics).

> Warning: `anomalib`'s version does not include the following features available here:
> - `numpy`-only API (only `torch` and `torchmetrics`-based API)
> - `numba` accelaration (makes it considerably slower)

## Installation

If you want to use AUPIMO, you can clone the git repository and install it with pip:

```bash
git clone git@github.com:jpcbertoldo/aupimo.git
cd aupimo
pip install .
```

You can add it to your `requirements.txt` file as `aupimo @ git+https://github.com/jpcbertoldo/aupimo`.

> PYPI package COMING UP

## Reproducing and extending paper results

If you want to reproduce or extend the results of the paper, install the requirements in `requirements/aupimo-paper.txt` as well:

```bash
git clone git@github.com:jpcbertoldo/aupimo.git
cd aupimo
pip install -e .  # `-e` is for 'editable' mode
pip install -r requirements/aupimo-paper.txt
```

> **Important:** it is recommended to use a virtual environment to install the dependencies and run the tests.
> We recommend using `conda`, and an enviroment file `dev-env.yml` is provided at the root of the repository.
> Install it with `conda env create -f dev-env.yml` and activate it with `conda activate aupimo-dev`.

### Data setup

In order to recompute the metrics reported in the paper you can use the script `scripts/eval.py`.

You will need to first setup the data, the anomaly score maps and images with their masks from the public datasets.

#### Anomaly score maps (`asmaps`)

Download them by running `data/experiments/download_asmaps.sh` then unzip the downloaded zip file exactly where it is (it will match the folder structure of in `data/experiments/benchmark`).

#### Images and masks

You can download MVTec AD and VisA from their respective original sources:

- MVTec AD: <https://www.mvtec.com/company/research/datasets/mvtec-ad/>
- VisA: <https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar>

You should unpack the data in folders respecitvely named `MVTec` and `VisA`, and the paths to these folders will be passed to the `eval.py` script.

## Development

If you want to modify the package and eventually open a Pull Request, install the requirements in `requirements/dev.txt` and install pre-commit hooks:

```bash
git clone git@github.com:jpcbertoldo/aupimo.git
cd aupimo
pip install -e .  # `-e` is for 'editable' mode
pip install -r requirements/dev.txt
pre-commit install
```

Run the tests in `tests/` locally with `pytest` before opening a Pull Request:

```bash
pytest tests/
```

> **Important:** it is recommended to use a virtual environment to install the dependencies and run the tests.
> We recommend using `conda`, and an enviroment file `dev-env.yml` is provided at the root of the repository.
> Install it with `conda env create -f dev-env.yml` and activate it with `conda activate aupimo-dev`.

## Cite Us

```tex

@misc{aupimo,
  title = {{AUPIMO}: Redefining Visual Anomaly Detection Benchmarks with High Speed and Low Tolerance},
  shorttitle = {{AUPIMO}},
  author = {Bertoldo, Joao P. C. and
            Ameln, Dick and
            Vaidya, Ashwin and
            Akçay, Samet},
  year = {2024},
  eprint = {2401.01984},
  eprinttype = {arxiv},
  primaryClass={cs.CV}
}
```

arXiv: <https://arxiv.org/abs/2401.01984> (accepted to BMVC 2024, coming up in November 2024)

Medium post: <https://medium.com/p/c653ac30e802>

Papers With Code: <https://paperswithcode.com/paper/aupimo-redefining-visual-anomaly-detection>

This research has been conducted during Google Summer of Code 2023 (GSoC 2023) at OpenVINO (Intel).

GSoC 2023 page: <https://summerofcode.withgoogle.com/archive/2023/projects/SPMopugd>
