# ASGD Falsification by Verified Beam Search

This repository implements a verified (sound) search framework for **worst-case delay scheduling** in deterministic asynchronous SGD (ASGD) on quadratic objectives. The core idea is to combine:

- an **upper-bounding tail estimate** derived from spectral / modal analysis (sound but possibly loose), and
- a **beam-search style exploration** that maintains both a feasible lower bound (LB) and a sound global upper bound (UB),
  enabling an instance-level **optimality gap certificate**.

Gurobi (MIQCP) is supported as an **optional** cross-check / baseline, but the main pipeline is designed to work without it.

## Repository structure

- `alg/` — search algorithms (beam search with leakage accounting, bounding utilities)
- `core/` — ASGD system model, dynamics, and bound model components
- `examples/` — example instances / synthetic benchmarks
- `tests/` — unit tests and sanity checks
- Top-level scripts:
  - `main_verify_beam.py` — main verified-beam runner for a single instance
  - `run_experiments.py` — batch experiments / paper-style runs

## Installation

### Option A: pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Developer
[Chao Huang](https://chaohuang2018.github.io/)