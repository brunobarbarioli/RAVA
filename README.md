# RAVA: Regulatory-Aligned Verification for AI Agents

Complete, runnable research codebase implementing the RAVA framework with:
- Formal constraint specs (`HARD`, `SOFT`, `STATISTICAL`) and compositional semantics (`AND`/`OR`/`IMPLIES` with `UNCERTAIN` propagation)
- Three-layer verifier (pre-execution, runtime monitoring, post-hoc auditing)
- Domain-weighted reliability scoring and certification tiers aligned with the paper
- End-to-end experiments for healthcare, finance, and HR across verification ablations

## Repository Layout

```text
configs/
  base.yaml
  domains/{healthcare,finance,hr}.yaml
  models/*.yaml
  experiments/{default_sweep,smoke}.yaml
specs/{healthcare,finance,hr}.yaml
scripts/
src/rava/
tests/
```

## Installation

### Option A: venv + pip

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

### Option B: uv

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .[dev]
```

Optional extras for full dataset/model integrations:

```bash
pip install -e .[full]
```

## Unified CLI

```bash
rava run_agent --domain healthcare --model-config configs/models/ollama_ministral3_cloud.yaml "What should I do for severe chest pain?"
rava download_datasets --domains healthcare,finance,hr --profile core
rava preprocess_datasets --domains healthcare,finance,hr --profile core
rava run_experiment --sweep-config configs/experiments/smoke.yaml
rava evaluate runs/<timestamp>/healthcare/mock-v1/full/42
rava make_tables --runs-root runs --output-dir outputs/tables
```

Equivalent module invocation:

```bash
python -m rava.cli --help
```

## Dataset Download & Preprocessing

The framework includes robust download wrappers and BYO fallbacks. Raw files go to `data/raw/<dataset>/`; processed standardized JSONL goes to `data/processed/<domain>/<dataset>/data.jsonl`.

Common processed schema:
- `id`
- `domain`
- `task`
- `input`
- `reference`
- `metadata`
- `split`

### Healthcare
- MedQA (USMLE): BYO loader (distribution-dependent licensing)
- PubMedQA: Hugging Face attempt (`pubmed_qa`) with fallback instructions
- MedHalt: BYO fallback by default
- Enhanced profile adds: PubHealth, EHRSQL, MIMIC-IV BHC (BYO/restricted)

### Finance
- FinBen: BYO fallback by default
- FLUE: BYO fallback by default
- ConvFinQA: Hugging Face attempt (`ibm/convfinqa`) with fallback
- Enhanced profile adds: FinanceBench, TAT-QA, FinQA (BYO wrappers)

### HR
- BBQ: Hugging Face attempt (`heegyu/bbq`) with fallback
- WinoBias: BYO fallback by default
- Jigsaw Unintended Bias (Kaggle): Kaggle API download, requires credentials
- Synthetic resumes: deterministic template generator (`2,000` rows default)
- Enhanced profile adds: Bias in Bios, FairJob, ACS PUMS HR slices (BYO wrappers)

Generate synthetic resumes:

```bash
rava generate_synthetic_resumes --n 2000 --seed 42
# or
python scripts/generate_synthetic_resumes.py --n 2000 --seed 42
```

### Kaggle Credentials (Jigsaw)

Set either env vars:

```bash
export KAGGLE_USERNAME=...
export KAGGLE_KEY=...
```

Or place `~/.kaggle/kaggle.json`.

If credentials are missing, the downloader emits actionable instructions and creates BYO guidance files.

## Dataset License / Terms Notes

This repository writes a `LICENSE_NOTICE.txt` per dataset folder and requires users to comply with source terms. Official references:
- PubMedQA: <https://pubmedqa.github.io/>
- MedQA: <https://github.com/jind11/MedQA>
- MedHalt: <https://github.com/allenai/medhalt>
- ConvFinQA: <https://github.com/czyssrs/ConvFinQA>
- FLUE: <https://arxiv.org/abs/2202.12005>
- BBQ: <https://github.com/nyu-mll/BBQ>
- WinoBias: <https://uclanlp.github.io/corefBias/overview>
- Jigsaw Unintended Bias: <https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification>

## Provider Interface (No Hardcoded Proprietary APIs)

Implemented providers:
- `MockProvider` (deterministic, offline)
- `LangChainOllamaCloudProvider` (`OLLAMA_API_KEY`, OpenAI-compatible endpoint `https://ollama.com/v1`)
- `OpenAIProvider` placeholder (`OPENAI_API_KEY`)
- `AnthropicProvider` placeholder (`ANTHROPIC_API_KEY`)
- `GoogleProvider` placeholder (`GOOGLE_API_KEY`)
- `LocalHFProvider` placeholder

### Ollama Cloud Setup

Create `.env` in repo root:

```bash
OLLAMA_API_KEY=<your_ollama_cloud_key>
OLLAMA_BASE_URL=https://ollama.com/v1
```

The CLI auto-loads `.env` via `python-dotenv`.

Use model configs in `configs/models/`.

## Running Experiments

### Smoke run (<1 minute, mock provider)

```bash
rava preprocess_datasets --domains healthcare,finance,hr
rava run_experiment --sweep-config configs/experiments/smoke.yaml
```

Outputs per run:

```text
runs/<timestamp>/<domain>/<model>/<config>/<seed>/
  trajectory.jsonl
  predictions.jsonl
  verdicts.jsonl
  metrics.json
  report.json
  timing.json
```

### Full sweep template

```bash
rava run_experiment --sweep-config configs/experiments/default_sweep.yaml
```

### Ollama Cloud comparison (ministral-3 vs qwen3-next)

```bash
rava preprocess_datasets --domains healthcare,finance,hr --profile core
rava run_experiment --sweep-config configs/experiments/ollama_cloud_eval.yaml
rava make_tables --runs-root runs --output-dir outputs/tables
```

Enhanced-dataset sweep:

```bash
rava preprocess_datasets --domains healthcare,finance,hr --profile enhanced
rava run_experiment --sweep-config configs/experiments/ollama_cloud_eval_enhanced.yaml
```

Adversarial stress sweep:

```bash
rava run_experiment --sweep-config configs/experiments/ollama_cloud_eval_stress.yaml
```

## Evaluation, Scoring, and Tables

Metrics implemented:
- Hard/soft violation rates
- Factual grounding (`claim_precision`)
- Claim citation coverage + evidence support rate
- Calibration (`ECE`, 10 bins)
- Fairness: 4/5ths, demographic parity difference, equalized odds difference
- Source attribution score
- Abstention rate
- Latency metrics (overall + per verification layer)
- Estimated token/cost metrics for assurance-cost frontiers

Reliability score `R`:
- Weighted by domain-specific configs (`configs/domains/*.yaml`)
- Tiered certification (paper-aligned): `Tier 2 (Supervised Autonomy)`, `Tier 1 (Advisory)`, `Tier 3 (Human-in-the-Loop Required)`

Generate paper-ready tables:

```bash
rava make_tables --runs-root runs --output-dir outputs/tables
```

Produces:
- `outputs/tables/healthcare.csv`
- `outputs/tables/finance.csv`
- `outputs/tables/hr.csv`
- `outputs/tables/ablation.csv`
- `outputs/tables/latency.csv`
- `outputs/tables/prevention_detection.csv`
- `outputs/tables/cost_frontier.csv`
- `outputs/tables/significance.csv`
- and matching `.tex` files

## Verification Layers

1. **Pre-execution verifier**
   - Runs before tool calls and final answer
   - Decision: `APPROVE`/`BLOCK`/`MODIFY`/`FLAG` (implemented with repair prompt path)
2. **Runtime monitor**
   - Checks action/observation transitions
   - Dual-judge consensus (primary + heuristic secondary) with state trackers
   - Supports halting on hard failures
3. **Post-hoc auditor**
   - Per-constraint verdict report
   - Claim decomposition + retrieval-backed verification + claim-evidence graph

## Reproducibility

- Deterministic seeding (`42`, `123`, `456` defaults)
- Structured JSONL artifacts and logs
- Environment capture in run root (`environment.json`)
- Config-driven sweeps

## Testing

Run tests (no external datasets required):

```bash
pytest
```

Included tests:
- Spec composition semantics (`AND`/`OR`/`IMPLIES` + `UNCERTAIN`)
- Fairness metric math
- Reliability scoring and tiering
- CLI smoke (`python -m rava.cli --help` and tiny mock run)

## Troubleshooting

- `datasets` import errors: install optional deps (`pip install -e .[full]`) or use BYO files.
- Kaggle download fails: verify `KAGGLE_USERNAME/KAGGLE_KEY` or `~/.kaggle/kaggle.json`.
- Missing processed files: run `rava preprocess_datasets`; fallback toy splits are generated when raw data is absent.
- No citations in outputs: switch from `MockProvider` to a real provider integration and retrieval backends.
