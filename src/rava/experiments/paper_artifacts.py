from __future__ import annotations

from pathlib import Path
from typing import Any

from rava.specs.parser import load_spec
from rava.utils.serialization import read_json, read_yaml


def _tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = str(text)
    for src, target in replacements.items():
        out = out.replace(src, target)
    out = out.replace("Σ", r"$\Sigma$")
    return out


def _spec_table_tex(spec_path: Path) -> str:
    spec = load_spec(spec_path)
    lines = [
        f"\\subsubsection*{{{_tex_escape(spec.domain.title())} Specification Catalog}}",
        "\\begingroup",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{longtable}{@{}p{0.12\\linewidth}p{0.45\\linewidth}p{0.14\\linewidth}p{0.18\\linewidth}@{}}",
        "\\toprule",
        "ID & Description & Type & Implementation \\\\",
        "\\midrule",
        "\\endhead",
    ]
    for constraint in spec.constraints:
        trigger = ",".join(constraint.trigger.events)
        impl = _tex_escape(constraint.predicate)
        if constraint.type.value == "STATISTICAL":
            metric = _tex_escape(str(constraint.metric or ""))
            threshold = _tex_escape(str(constraint.threshold if constraint.threshold is not None else ""))
            grouping = _tex_escape(",".join(constraint.grouping_fields))
            impl = f"{impl}; metric={metric}; threshold={threshold}; groups={grouping}"
        if trigger:
            impl = f"{impl}; events={_tex_escape(trigger)}"
        lines.append(
            f"{_tex_escape(constraint.id)} & {_tex_escape(constraint.description)} & "
            f"{_tex_escape(constraint.type.value)} & {impl} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{longtable}", "\\endgroup"])
    return "\n".join(lines) + "\n"


def _implementation_facts_tex(
    *,
    base_config_path: str | Path,
    sweep_config_paths: list[str | Path] | None,
    model_config_paths: list[str | Path] | None,
) -> str:
    base_cfg = read_yaml(base_config_path)
    model_paths = [Path(p) for p in (model_config_paths or [])]
    sweep_paths = [Path(p) for p in (sweep_config_paths or [])]

    lines = [
        "\\subsubsection*{Implementation Facts (Generated from Configs)}",
        "Code artifacts are the source of truth for this appendix. The table below is generated from the checked-in YAML configs.",
        "",
        "\\paragraph{Python compatibility.} Supported interpreter versions: 3.11, 3.12, 3.13, 3.14.",
        "",
        "\\paragraph{Runtime defaults.}",
        "\\begin{itemize}",
        f"\\item Agent backend: \\texttt{{{_tex_escape(str(base_cfg.get('agentic_framework', {}).get('backend', 'langgraph')))}}}",
        f"\\item Tool cache scope: \\texttt{{{_tex_escape(str(base_cfg.get('agentic_framework', {}).get('tool_cache_scope', 'example')))}}}",
        f"\\item Max tool iterations: {_tex_escape(str(base_cfg.get('agentic_framework', {}).get('max_tool_iterations', 64)))}",
        f"\\item Default seeds: {_tex_escape(', '.join(str(x) for x in base_cfg.get('experiments', {}).get('seeds', [])))}",
        "\\end{itemize}",
        "",
    ]
    if model_paths:
        lines.extend(
            [
                "\\paragraph{Model configs.}",
                "\\begingroup",
                "\\small",
                "\\begin{longtable}{@{}p{0.18\\linewidth}p{0.18\\linewidth}p{0.18\\linewidth}p{0.14\\linewidth}p{0.12\\linewidth}p{0.12\\linewidth}@{}}",
                "\\toprule",
                "Config & Provider & Model & Temperature & Max tokens & Timeout \\\\",
                "\\midrule",
                "\\endhead",
            ]
        )
        for path in model_paths:
            cfg = read_yaml(path)
            lines.append(
                f"{_tex_escape(path.name)} & {_tex_escape(str(cfg.get('provider', 'unknown')))} & "
                f"{_tex_escape(str(cfg.get('model', cfg.get('name', 'unknown'))))} & "
                f"{_tex_escape(str(cfg.get('temperature', '')))} & "
                f"{_tex_escape(str(cfg.get('max_tokens', '')))} & "
                f"{_tex_escape(str(cfg.get('timeout', '')))} \\\\"
            )
        lines.extend(["\\bottomrule", "\\end{longtable}", "\\endgroup", ""])

    if sweep_paths:
        lines.extend(
            [
                "\\paragraph{Sweep configs.}",
                "\\begingroup",
                "\\small",
                "\\begin{longtable}{@{}p{0.2\\linewidth}p{0.18\\linewidth}p{0.12\\linewidth}p{0.16\\linewidth}p{0.24\\linewidth}@{}}",
                "\\toprule",
                "Sweep & Dataset profile & Eval split & Verification configs & Seeds \\\\",
                "\\midrule",
                "\\endhead",
            ]
        )
        for path in sweep_paths:
            cfg = read_yaml(path)
            lines.append(
                f"{_tex_escape(path.name)} & {_tex_escape(str(cfg.get('dataset_profile', 'core')))} & "
                f"{_tex_escape(str(cfg.get('eval_split', 'preserve')))} & "
                f"{_tex_escape(', '.join(str(x) for x in cfg.get('verification_configs', [])))} & "
                f"{_tex_escape(', '.join(str(x) for x in cfg.get('seeds', [])))} \\\\"
            )
        lines.extend(["\\bottomrule", "\\end{longtable}", "\\endgroup", ""])

    lines.append(
        "Unsupported manuscript claims are intentionally omitted here: only implemented providers, judges, tools, and schedules appear in generated facts."
    )
    return "\n".join(lines) + "\n"


def _compliance_tex(compliance_summary_path: str | Path | None) -> str:
    if compliance_summary_path is None or not Path(compliance_summary_path).exists():
        return (
            "\\subsubsection*{Compliance Challenge Evaluation}\n"
            "Empirical detector-recall claims are withheld until a compliance challenge summary has been generated from "
            "\\texttt{rava evaluate\\_compliance\\_challenges}.\n"
        )

    payload = read_json(compliance_summary_path)
    lines = [
        "\\subsubsection*{Compliance Challenge Evaluation}",
        "The following table is generated from the repo-tracked compliance challenge set and reports detector recall and joint miss estimates with Wilson confidence intervals.",
        "\\begingroup",
        "\\small",
        "\\begin{longtable}{@{}p{0.18\\linewidth}p{0.11\\linewidth}p{0.2\\linewidth}p{0.2\\linewidth}p{0.2\\linewidth}@{}}",
        "\\toprule",
        "Domain & N & $r_{pre}$ & $r_{rt}$ & $\\hat{q}$ \\\\",
        "\\midrule",
        "\\endhead",
    ]
    domains = payload.get("domains", {}) if isinstance(payload, dict) else {}
    for domain in ["healthcare", "finance", "hr"]:
        row = domains.get(domain)
        if not isinstance(row, dict):
            continue
        lines.append(
            f"{_tex_escape(domain.title())} & {int(row.get('n_cases', 0))} & "
            f"{float(row.get('r_pre', 0.0)):.3f} "
            f"[{float(row.get('r_pre_ci_low', 0.0)):.3f}, {float(row.get('r_pre_ci_high', 0.0)):.3f}] & "
            f"{float(row.get('r_rt', 0.0)):.3f} "
            f"[{float(row.get('r_rt_ci_low', 0.0)):.3f}, {float(row.get('r_rt_ci_high', 0.0)):.3f}] & "
            f"{float(row.get('q_hat', 0.0)):.3f} "
            f"[{float(row.get('q_hat_ci_low', 0.0)):.3f}, {float(row.get('q_hat_ci_high', 0.0)):.3f}] \\\\"
        )
    overall = payload.get("overall", {}) if isinstance(payload, dict) else {}
    if isinstance(overall, dict) and overall:
        lines.append("\\midrule")
        lines.append(
            f"Overall & {int(overall.get('n_cases', 0))} & "
            f"{float(overall.get('r_pre', 0.0)):.3f} "
            f"[{float(overall.get('r_pre_ci_low', 0.0)):.3f}, {float(overall.get('r_pre_ci_high', 0.0)):.3f}] & "
            f"{float(overall.get('r_rt', 0.0)):.3f} "
            f"[{float(overall.get('r_rt_ci_low', 0.0)):.3f}, {float(overall.get('r_rt_ci_high', 0.0)):.3f}] & "
            f"{float(overall.get('q_hat', 0.0)):.3f} "
            f"[{float(overall.get('q_hat_ci_low', 0.0)):.3f}, {float(overall.get('q_hat_ci_high', 0.0)):.3f}] \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{longtable}", "\\endgroup"])
    return "\n".join(lines) + "\n"


def make_paper_artifacts(
    *,
    output_dir: str | Path = "outputs/paper_generated",
    base_config_path: str | Path = "configs/base.yaml",
    sweep_config_paths: list[str | Path] | None = None,
    model_config_paths: list[str | Path] | None = None,
    compliance_summary_path: str | Path | None = None,
) -> list[Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    spec_out = out_dir / "appendix_specs.tex"
    spec_tex = []
    for domain in ["healthcare", "finance", "hr"]:
        spec_tex.append(_spec_table_tex(Path("specs") / f"{domain}.yaml"))
    spec_out.write_text("\n".join(spec_tex), encoding="utf-8")
    outputs.append(spec_out)

    facts_out = out_dir / "implementation_facts.tex"
    facts_out.write_text(
        _implementation_facts_tex(
            base_config_path=base_config_path,
            sweep_config_paths=sweep_config_paths,
            model_config_paths=model_config_paths,
        ),
        encoding="utf-8",
    )
    outputs.append(facts_out)

    compliance_out = out_dir / "compliance_challenges.tex"
    compliance_out.write_text(_compliance_tex(compliance_summary_path), encoding="utf-8")
    outputs.append(compliance_out)

    return outputs
