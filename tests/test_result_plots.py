from __future__ import annotations

from pathlib import Path

import pandas as pd

from rava.experiments.plots import make_result_plots


def test_make_result_plots(tmp_path: Path) -> None:
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "domain": "healthcare",
                "model": "gpt-5.4",
                "n_paired_seeds": 4,
                "meets_min_seed_requirement": False,
                "R_delta_full_minus_none": 0.058,
                "p_value_permutation": float("nan"),
                "effect_size_dz": 1.2,
                "p_value_holm": float("nan"),
                "significant_holm_0p05": False,
                "underpowered": True,
            },
            {
                "domain": "finance",
                "model": "gpt-5.4",
                "n_paired_seeds": 5,
                "meets_min_seed_requirement": True,
                "R_delta_full_minus_none": 0.038,
                "p_value_permutation": 0.062,
                "effect_size_dz": 2.1,
                "p_value_holm": 0.248,
                "significant_holm_0p05": False,
                "underpowered": False,
            },
            {
                "domain": "hr",
                "model": "gpt-5.4",
                "n_paired_seeds": 5,
                "meets_min_seed_requirement": True,
                "R_delta_full_minus_none": 0.198,
                "p_value_permutation": 0.062,
                "effect_size_dz": 9.6,
                "p_value_holm": 0.248,
                "significant_holm_0p05": False,
                "underpowered": False,
            },
            {
                "domain": "healthcare",
                "model": "ministral-3-cloud",
                "n_paired_seeds": 4,
                "meets_min_seed_requirement": False,
                "R_delta_full_minus_none": 0.027,
                "p_value_permutation": float("nan"),
                "effect_size_dz": 0.9,
                "p_value_holm": float("nan"),
                "significant_holm_0p05": False,
                "underpowered": True,
            },
            {
                "domain": "finance",
                "model": "ministral-3-cloud",
                "n_paired_seeds": 5,
                "meets_min_seed_requirement": True,
                "R_delta_full_minus_none": 0.058,
                "p_value_permutation": 0.062,
                "effect_size_dz": 3.0,
                "p_value_holm": 0.248,
                "significant_holm_0p05": False,
                "underpowered": False,
            },
            {
                "domain": "hr",
                "model": "ministral-3-cloud",
                "n_paired_seeds": 5,
                "meets_min_seed_requirement": True,
                "R_delta_full_minus_none": 0.113,
                "p_value_permutation": 0.062,
                "effect_size_dz": 1.8,
                "p_value_holm": 0.248,
                "significant_holm_0p05": False,
                "underpowered": False,
            },
        ]
    ).to_csv(tables_dir / "significance.csv", index=False)

    frontier_rows = []
    latency_rows = []
    for domain in ["healthcare", "finance", "hr"]:
        for model in ["gpt-5.4", "ministral-3-cloud"]:
            for idx, config in enumerate(["none", "pre", "runtime", "posthoc", "full"]):
                frontier_rows.append(
                    {
                        "domain": domain,
                        "model": model,
                        "config": config,
                        "R_mean": 0.3 + 0.03 * idx,
                        "estimated_cost_usd": 0.0,
                        "estimated_total_tokens": 1000 + idx,
                        "cost_per_R": 0.0,
                    }
                )
                latency_rows.append(
                    {
                        "domain": domain,
                        "model": model,
                        "config": config,
                        "latency_avg_ms": 1000 + idx * 250,
                    }
                )
    pd.DataFrame(frontier_rows).to_csv(tables_dir / "cost_frontier.csv", index=False)
    pd.DataFrame(latency_rows).to_csv(tables_dir / "latency.csv", index=False)

    outputs = make_result_plots(tables_dir=tables_dir, output_dir=tmp_path / "figures", prefix="smoke")

    assert len(outputs) == 2
    for path in outputs:
        assert path.exists()
        assert path.stat().st_size > 0
