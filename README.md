# RAVA

RAVA is a research codebase for regulatory-aligned verification of AI agents in high-stakes domains. It implements:

- a typed specification language with `HARD`, `SOFT`, and `STATISTICAL` constraints
- three verification layers: pre-execution, runtime monitoring, and post-hoc auditing
- audited and operational scoring tracks
- validity-envelope gating with `R_raw` vs `R_certified`
- a LangGraph-first agent runtime with provider abstractions for Ollama Cloud, OpenAI, and mock/offline execution
- end-to-end experiment, table, plot, and paper artifact generation

## Repository layout

```text
configs/
  base.yaml
  domains/
  experiments/
  models/
specs/
scripts/
src/rava/
  agent/
  experiments/
  metrics/
  scoring/
  verification/
tests/
```

Large artifacts are intentionally ignored:

- `data/`
- `runs/`
- `outputs/`
- paper sources and generated paper assets

That keeps the Git repository light. Reproducible artifacts are regenerated locally from the configs and code.

## Requirements

- Python `3.11` to `3.14`
- Linux or macOS shell environment
- `latexmk` + a TeX distribution to compile the paper
- optional cloud credentials for Ollama Cloud / OpenAI

Pinned agent stack:

- `langchain==1.2.10`
- `langchain-core==1.2.18`
- `langchain-openai==1.1.11`
- `langgraph==1.1.0`

## Installation

### venv + pip

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
pip install -e .[full]
```

### Conda

```bash
conda env create -f configs/envs/py314.conda.yml
conda activate rava-py314
pip install -e .[full,dev]
```

## Environment variables

Copy [.env.example](.env.example) to `.env` and fill the keys you need.

```bash
cp .env.example .env
```

Supported variables:

- `OLLAMA_API_KEY`
- `OLLAMA_BASE_URL`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`

`.env` is ignored by Git.

## Dataset handling

Raw and processed datasets are not versioned. Use the dataset download and preprocessing commands instead.

### Final benchmark profile

```bash
rava download_datasets --domains healthcare,finance,hr --profile final_a6
rava preprocess_datasets --domains healthcare,finance,hr --profile final_a6 --split-strategy temporal --disallow-toy-fallback
```

### Profiles

- `core`: minimal default profile
- `paper_hybrid`: broader mirror-backed paper profile
- `paper6_fast`: 6-dataset fast publication profile
- `paper3_mini`: 3-dataset certifiable mini profile
- `primary_certification`: primary certified benchmark subset
- `diagnostic_secondary`: secondary diagnostic benchmark subset
- `final_a6`: final six-dataset dual-model paper profile

### Dataset notes

- `pubmedqa`, `convfinqa`, `bias_in_bios` are the strongest certification-friendly datasets in the current setup.
- `medqa`, `finben`, and `winobias` are included in the final six-dataset paper sweep.
- Kaggle-backed datasets are supported by the codebase, but they are not required for the final `final_a6` paper configuration.
- Synthetic datasets remain supplementary and are not part of the final primary capability tables.

## Quickstart

### CLI smoke test

```bash
PYTHONPATH=src python -m rava.cli --help
```

### Offline mock smoke run

```bash
rava preprocess_datasets --domains healthcare,finance,hr --profile core
rava run_experiment \
  --sweep-config configs/experiments/smoke.yaml \
  --base-config configs/base.yaml \
  --agentic-backend langgraph \
  --example-parallelism-per-run 2 \
  --sync-model-invocation
```

### Single example

```bash
rava run_agent \
  --domain healthcare \
  --model-config configs/models/mock.yaml \
  --verification-config full \
  "What should I do for severe chest pain?"
```

## Reproducing the final paper pipeline

### 1. Provider preflight

```bash
rava preflight_provider --model-config configs/models/ollama_ministral3_cloud.yaml --n-probes 5 --timeout 30
rava preflight_provider --model-config configs/models/openai_gpt54.yaml --n-probes 5 --timeout 30
```

### 2. Calibration

```bash
rava run_experiment \
  --sweep-config configs/experiments/final_a6_dual_model.yaml \
  --base-config configs/base.yaml \
  --stage calibration \
  --agentic-backend langgraph \
  --example-parallelism-per-run 4 \
  --sync-model-invocation
```

### 3. Full certified sweep

```bash
rava run_experiment \
  --sweep-config configs/experiments/final_a6_dual_model.yaml \
  --base-config configs/base.yaml \
  --stage full \
  --agentic-backend langgraph \
  --example-parallelism-per-run 4 \
  --sync-model-invocation
```

### 4. Tables, evidence, and plots

```bash
rava make_tables \
  --runs-root runs/20260311_final_a6_dual_model_full_v2 \
  --output-dir outputs/tables/final_a6_dual_model_full_v2 \
  --comparison-track audited \
  --certified-only

rava evidence_report \
  --runs-root runs/20260311_final_a6_dual_model_full_v2 \
  --output-path outputs/evidence/final_a6_dual_model_full_v2.json \
  --comparison-track audited

rava make_result_plots \
  --tables-dir outputs/tables/final_a6_dual_model_full_v2 \
  --output-dir figures \
  --prefix final_a6

rava make_paper_artifacts \
  --output-dir outputs/paper_generated \
  --base-config configs/base.yaml \
  --sweep-configs configs/experiments/final_a6_dual_model.yaml \
  --model-configs configs/models/ollama_ministral3_cloud.yaml,configs/models/openai_gpt54.yaml \
  --compliance-summary outputs/compliance/challenge_summary.json
```

### 5. Optional paper build

If you also keep the manuscript source locally, regenerate the paper artifacts and build it with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error RAVA_revised_emnlp2026.tex
```

## Main CLI commands

```bash
rava download_datasets
rava preprocess_datasets
rava run_agent
rava preflight_provider
rava run_experiment
rava evaluate
rava make_tables
rava evidence_report
rava evaluate_compliance_challenges
rava make_result_plots
rava make_paper_artifacts
rava generate_synthetic_resumes
rava generate_agentic_stress_hr
```

## Paper artifact policy

The repository generates paper tables, plots, and appendix fragments locally, but those files are not meant to be part of the code-only Git push. Use:

```bash
rava make_tables
rava make_result_plots
rava make_paper_artifacts
```

and compile the manuscript locally if you need the paper outputs.

## Testing

Run the full test suite:

```bash
PYTHONPATH=src pytest -q
```

Targeted paper/plot checks:

```bash
PYTHONPATH=src pytest -q tests/test_paper_artifacts.py tests/test_result_plots.py
```

## Troubleshooting

### Missing dataset credentials

- Kaggle datasets require `KAGGLE_USERNAME` / `KAGGLE_KEY` or `~/.kaggle/kaggle.json`.
- Restricted datasets fall back to BYO instructions instead of failing silently.

### Provider issues

- use `rava preflight_provider` before long sweeps
- publication sweeps hard-fail on preflight failure when configured
- `gpt-5.4` finance instability was fixed by:
  - raising `max_tokens`
  - classifying `length_limit`
  - targeted retry budgets
  - routing `finben` through direct structured chat

### Paper build

- `latexmk` is the supported build path
- appendix tables come from generated LaTeX fragments in `outputs/paper_generated`
- if those files are missing, regenerate them with `rava make_paper_artifacts`

## Artifact policy

The repository is configured to ignore:

- datasets under `data/`
- experiment runs under `runs/`
- generated evaluation outputs under `outputs/`
- paper sources, generated figures, and local LaTeX build files

This is intentional. The source-controlled part of the project is the code, configs, scripts, and tests required to regenerate the experimental and paper artifacts locally.

## License

MIT for the code in this repository. Dataset licenses remain governed by their original sources.
