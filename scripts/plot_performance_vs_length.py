#!/usr/bin/env python3
"""
Plot average Voronoi performance across four protein-length groups.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METHOD = "Voronoi (1N KNN)"
DEFAULT_METRIC = "f1_measure"
DEFAULT_OUTPUT = "artifacts/performance_vs_length.png"
DEFAULT_JSON = "artifacts/LPTD_Results.json"
DEFAULT_EXCEL = "artifacts/proteins_list.xlsx"

COLORS = {'SVM Linear': 'red', 'SVM RBF': 'yellow', 'Random Forest': 'green', 'Voronoi (1N KNN)': 'blue', 'LPTD': 'purple'}


def normalize_accuracy(value: object) -> float:
    """Convert accuracy to percentage scale when it is stored as 0-1."""
    if value is None:
        return float("nan")

    numeric = float(value)
    if numeric <= 1.0:
        return numeric * 100.0
    return numeric


def load_length_table(excel_file: Path) -> pd.DataFrame:
    """Load protein names and lengths from the Excel sheet."""
    df = pd.read_excel(excel_file)

    if df.shape[1] < 5:
        raise ValueError(
            f"Expected at least 5 columns in {excel_file}, found {df.shape[1]}."
        )

    table = pd.DataFrame(
        {
            "Protein": df.iloc[:, 0].astype(str).str.strip(),
            "Length": pd.to_numeric(df.iloc[:, 4], errors="coerce"),
        }
    )
    return table.dropna(subset=["Protein", "Length"]).reset_index(drop=True)


def compute_all_metrics(
    results: Dict[str, dict]
) -> pd.DataFrame:
    """Compute per-protein metrics for all methods averaged over Helix/Strand."""
    rows = []

    for protein, protein_data in results.items():
        row = {"Protein": protein}
        has_any_value = False

        for method in COLORS.keys():
            values = []
            for structure_type in ("Helix", "Strand"):
                structure_data = protein_data.get(structure_type)
                if not isinstance(structure_data, dict):
                    continue

                method_data = structure_data.get(method)
                if not isinstance(method_data, dict):
                    continue

                if "confusion_matrix_detailed" not in method_data:
                    continue

                value = method_data.get(DEFAULT_METRIC)
                if value is not None:
                    values.append(float(value))

            if values:
                row[method] = float(np.mean(values))
                has_any_value = True
            else:
                row[method] = np.nan

        if has_any_value:
            rows.append(row)

    return pd.DataFrame(rows)


def build_length_groups(df: pd.DataFrame, num_groups: int = 4) -> Tuple[pd.DataFrame, List[str]]:
    """Assign proteins to length groups using quantile-based bins."""
    if df.empty:
        raise ValueError("No proteins available after merging length and performance data.")

    lengths = df["Length"].astype(float)

    try:
        grouped, bins = pd.qcut(lengths, q=num_groups, retbins=True, duplicates="raise")
    except ValueError:
        # Fallback for highly duplicated length values.
        bins = np.linspace(lengths.min(), lengths.max(), num_groups + 1)
        grouped = pd.cut(lengths, bins=bins, include_lowest=True, duplicates="drop")

    labels: List[str] = []
    for interval in grouped.cat.categories:
        left = int(round(interval.left))
        right = int(round(interval.right))
        labels.append(f"{left}–{right}")
        
    grouped = grouped.cat.rename_categories(labels)

    result = df.copy()
    result["Length Group"] = grouped.astype(str)

    return result, labels


def summarize_groups(df: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    """Return a summary table with metric means per length group."""
    group_order = list(df["Length Group"].dropna().unique())
    summary_rows = []

    for label in group_order:
        group_df = df[df["Length Group"] == label]
        row = {"Length Group": label, "Proteins": int(group_df.shape[0])}
        for method in COLORS.keys():
            if method in group_df.columns:
                row[method] = float(group_df[method].mean())
            else:
                row[method] = np.nan
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["Length Group"] = pd.Categorical(summary["Length Group"], categories=group_order, ordered=True)
        summary = summary.sort_values("Length Group").reset_index(drop=True)

    return summary


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot a bar chart of F1-measure by property length group."""
    if summary.empty:
        raise ValueError("No summary data to plot.")

    groups = summary["Length Group"].astype(str).tolist()
    # Format labels to remove parentheses and commas
    formatted_groups = [g.replace('(', '').replace(']', '').replace(', ', '-') for g in groups]
    x = np.arange(len(groups))
    
    methods = list(COLORS.keys())
    num_methods = len(methods)
    width = 0.15 # 5 bars -> 5*0.15 = 0.75 width total
    
    plt.figure(figsize=(12, 6))
    
    # Create bars for each method
    for i, method in enumerate(methods):
        offset = (i - num_methods / 2) * width + width / 2
        
        if method in summary.columns:
            f1_values = summary[method].to_numpy(dtype=float)
        else:
            f1_values = np.full(len(groups), np.nan)
            
        plt.bar(x + offset, f1_values, width, color=COLORS.get(method, 'gray'), label=method, alpha=0.5)
        
        # Place text on bars
        for idx, (xi, val) in enumerate(zip(x, f1_values)):
            if not np.isnan(val):
                plt.text(xi + offset, val + 1, f"{val:.1f}", ha="center", va="bottom", fontsize=10, rotation=90)
    
    # Customize the plot
    plt.xlabel('Length ranges', fontsize=15)
    plt.ylabel('F1-Measure Average (%)', fontsize=15)
    
    plt.xticks(x, formatted_groups, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 110)
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    
    # Place the legend outside the plot, at the top center
    plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot average Voronoi performance across four protein-length groups."
    )
    parser.add_argument(
        "--json-file",
        type=Path,
        default=DEFAULT_JSON,
        help="Path to LPTD_Results.json.",
    )
    parser.add_argument(
        "--excel-file",
        type=Path,
        default=DEFAULT_EXCEL,
        help="Path to proteins_list.xlsx.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.json_file.exists():
        print(f"Error: JSON file not found: {args.json_file}")
        return 1

    if not args.excel_file.exists():
        print(f"Error: Excel file not found: {args.excel_file}")
        return 1

    with args.json_file.open("r", encoding="utf-8") as f:
        results = json.load(f)

    length_df = load_length_table(args.excel_file)
    metric_df = compute_all_metrics(results)

    merged = length_df.merge(metric_df, on="Protein", how="inner")
    if merged.empty:
        print("Error: no overlapping proteins found between the Excel file and JSON results.")
        return 1

    grouped_df, labels = build_length_groups(merged)
    summary = summarize_groups(grouped_df, labels)

    if summary.empty:
        print("Error: unable to compute grouped averages.")
        return 1

    plot_summary(summary, args.output)

    print("Performance vs. Length plot created successfully.")
    print(f"Input JSON:   {args.json_file}")
    print(f"Input Excel:  {args.excel_file}")
    print(f"Output plot:  {args.output}")
    print("\nGrouped averages:")
    print(summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())