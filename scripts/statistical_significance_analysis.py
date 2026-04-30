#!/usr/bin/env python3
"""
Statistical significance analysis for method performance in LPTD_Results.json.

This script:
1) Builds paired per-protein scores for each method using available Helix/Strand entries.
2) Runs full pairwise paired t-tests across all methods.
3) Adds significance labels based on raw p-values.
4) Saves TXT reports (including full pairwise matrix).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


DEFAULT_METHODS = [
    "Voronoi (1N KNN)",
    "SVM Linear",
    "Random Forest",
    "SVM RBF",
    "LPTD",
]

VALID_METRICS = {
    "accuracy",
    "precision",
    "recall",
    "f1_measure",
    "mismatch_rate",
}


def significance_label(p_value: float) -> str:
    """Return a conventional significance label for a raw p-value."""
    if p_value < 0.05:
        return "s"
    return "ns"


def normalize_metric(metric: str, value: float) -> float:
    """Normalize metric scales to percentage-like values when needed."""
    if value is None:
        return np.nan

    numeric_value = float(value)

    if metric == "accuracy" and numeric_value <= 1.0:
        return numeric_value * 100.0

    return numeric_value


def per_protein_method_score(
    protein_data: dict,
    method: str,
    metric: str,
    structure_types: Tuple[str, ...] = ("Helix", "Strand"),
) -> float:
    """
    Compute one score per protein/method by averaging available structure-level metrics.
    """
    values: List[float] = []

    for structure_type in structure_types:
        structure_data = protein_data.get(structure_type)
        if not isinstance(structure_data, dict):
            continue

        method_data = structure_data.get(method)
        if not isinstance(method_data, dict):
            continue

        if metric in method_data:
            values.append(normalize_metric(metric, method_data[metric]))

    if not values:
        return np.nan

    return float(np.mean(values))


def build_paired_scores(
    results: Dict[str, dict], methods: List[str], metric: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build dataframe of per-protein scores, then return:
    - full score table (may contain NaN)
    - paired score table (complete cases only)
    """
    rows = []

    for protein, protein_data in results.items():
        row = {"Protein": protein}
        for method in methods:
            row[method] = per_protein_method_score(protein_data, method, metric)
        rows.append(row)

    full_df = pd.DataFrame(rows).sort_values("Protein").reset_index(drop=True)
    paired_df = full_df.dropna(subset=methods).reset_index(drop=True)

    return full_df, paired_df


def run_statistics(
    paired_df: pd.DataFrame,
    methods: List[str],
) -> dict:
    """Run full pairwise paired t-tests and annotate raw p-values."""
    if paired_df.shape[0] < 2:
        raise ValueError(
            "Not enough paired proteins to run statistical tests. "
            "Need at least 2 complete paired observations."
        )

    comparisons = []

    for i, method_a in enumerate(methods):
        vals_a = paired_df[method_a].to_numpy(dtype=float)
        for method_b in methods[i + 1 :]:
            vals_b = paired_df[method_b].to_numpy(dtype=float)
            ttest_stat, ttest_p_value = ttest_rel(vals_a, vals_b, alternative="two-sided")
            comparisons.append(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "ttest_stat": float(ttest_stat),
                    "ttest_p_raw": float(ttest_p_value),
                    "significance": significance_label(float(ttest_p_value)),
                }
            )

    return {"pairwise_ttest": comparisons}


def write_reports(
    output_prefix: Path,
    analysis_result: dict,
    metric: str,
    methods: List[str],
    full_df: pd.DataFrame,
    paired_df: pd.DataFrame,
) -> Tuple[Path, Path, Path]:
    txt_path = output_prefix.with_suffix(".txt")

    pairwise_df = pd.DataFrame(analysis_result["pairwise_ttest"])

    matrix_df = pd.DataFrame("-", index=methods, columns=methods)
    for method in methods:
        matrix_df.loc[method, method] = "-"

    for _, row in pairwise_df.iterrows():
        method_a = row["method_a"]
        method_b = row["method_b"]
        p_value = row["ttest_p_raw"]
        label = row["significance"]
        formatted = "<0.001" if p_value < 0.001 else f"{p_value:.3f}"
        formatted = f"{formatted} {label}"
        matrix_df.loc[method_a, method_b] = formatted
        matrix_df.loc[method_b, method_a] = formatted

    lines = [
        "Statistical Significance Analysis",
        "=" * 36,
        f"Metric: {metric}",
        f"Methods: {', '.join(methods)}",
        f"Proteins total: {full_df.shape[0]}",
        f"Proteins used (paired): {paired_df.shape[0]}",
        "",
        "Full pairwise paired t-test matrix (raw p-values + significance labels)",
        "-" * 62,
        matrix_df.to_string(),
        "",
        "Detailed pairwise paired t-test results",
        "-" * 40,
    ]

    if not pairwise_df.empty:
        lines.append(pairwise_df.to_string(index=False))
    else:
        lines.append("No pairwise comparisons performed.")

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    return txt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full pairwise paired t-tests across methods."
    )
    parser.add_argument(
        "--json-file",
        type=Path,
        default=Path(__file__).parent / 'artifacts' / "LPTD_Results.json",
        help="Path to input results JSON (default: LPTD_Results.json).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1_measure",
        choices=sorted(VALID_METRICS),
        help="Metric to compare across methods.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="Methods to include in testing (space-separated).",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(__file__).parent / 'artifacts' / "statistical_significance_report",
        help="Output prefix for generated report files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.json_file.exists():
        print(f"Error: input JSON not found: {args.json_file}")
        return 1

    with args.json_file.open("r", encoding="utf-8") as f:
        results = json.load(f)

    full_df, paired_df = build_paired_scores(results, args.methods, args.metric)

    try:
        analysis_result = run_statistics(
            paired_df=paired_df,
            methods=args.methods,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    txt_path = write_reports(
        output_prefix=args.output_prefix,
        analysis_result=analysis_result,
        metric=args.metric,
        methods=args.methods,
        full_df=full_df,
        paired_df=paired_df,
    )

    print("Statistical analysis completed.")
    print(f"Input JSON: {args.json_file}")
    print(f"Metric: {args.metric}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Paired proteins used: {paired_df.shape[0]} / {full_df.shape[0]}")
    print(f"TXT report:  {txt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
