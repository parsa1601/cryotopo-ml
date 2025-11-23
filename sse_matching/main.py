"""
Main entry point for protein structure analysis.
Runs the complete analysis pipeline for both helix and strand structures.
"""

import os
import sys
import warnings
import json
import numpy as np

from protein_trainer import ProteinTrainer
from config import CSV_DATASET, HELIX_PROTEIN_LIST, STRAND_PROTEIN_LIST
from plot_results import (
    plot_accuracy_charts,
    plot_metrics_bar_chart,
    plot_error_rate_line_chart,
)
from collections import defaultdict

sys.path.append(f"{os.path.dirname(os.getcwd())}")
warnings.filterwarnings("ignore")


def plot_charts_from_json(json_file_path="Final_Results.json"):
    """
    Load results from JSON file and generate all charts.

    Args:
        json_file_path (str): Path to the Final_Results.json file
    """
    try:
        # Check if file exists
        if not os.path.exists(json_file_path):
            print(f"Error: {json_file_path} not found!")
            return

        # Load the JSON data
        with open(json_file_path, "r") as json_file:
            final_accuracy_report = json.load(json_file)

        print(f"Loaded data from {json_file_path}")
        print("Generating charts...")

        plot_accuracy_charts(final_accuracy_report, 'f1_measure')
        plot_metrics_bar_chart(final_accuracy_report)
        plot_error_rate_line_chart(final_accuracy_report)

        print("All charts have been generated successfully!")

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file: {e}")
    except Exception as e:
        print(f"Error generating charts: {e}")


def main():
    """
    Main function to run the protein structure analysis.
    The first list contains proteins with records of Helix
    The second list contains proteins with records of Strand
    """

    trainer = ProteinTrainer(csv_path=CSV_DATASET, use_grid_search=False)
    trainer.file_handler.reset_report_file()

    trainer.file_handler.print_and_save(
        "\n\n\n\n*************************HELIX RESULTS:**********************"
    )
    final_accuracy_report = trainer.train_with_all_algorithms(
        HELIX_PROTEIN_LIST, "Helix"
    )
    trainer.print_overall_direction_summary()
    trainer.find_globally_best_parameters()

    trainer.file_handler.print_and_save(
        "\n\n\n\n*************************STRAND RESULTS:**********************"
    )

    trainer.reset_direction_stats()
    final_accuracy_report = trainer.train_with_all_algorithms(
        STRAND_PROTEIN_LIST, "Strand"
    )
    trainer.print_overall_direction_summary()
    trainer.find_globally_best_parameters()

    plot_accuracy_charts(final_accuracy_report, 'f1_measure')
    plot_metrics_bar_chart(final_accuracy_report)
    plot_error_rate_line_chart(final_accuracy_report)
    
    with open("Final_Results.json", "w") as json_file:
        json.dump(
            convert_dd_to_dict(final_accuracy_report),
            json_file,
            indent=4,
            default=make_serializable,
        )
    print(
        f"\nDirection analysis report has been saved to: {trainer.file_handler.report_file}"
    )


def convert_dd_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: convert_dd_to_dict(v) for k, v in d.items()}
    return d


def make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # convert ndarray → list
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()  # convert numpy numbers → Python scalars
    return str(obj)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Protein Structure Analysis")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Generate charts from existing Final_Results.json file only",
    )
    parser.add_argument(
        "--json-file",
        default="Final_Results.json",
        help="Path to JSON results file (default: Final_Results.json)",
    )

    args = parser.parse_args()

    if args.plot_only:
        print("Generating charts from existing results...")
        plot_charts_from_json(args.json_file)
    else:
        print("Running full protein structure analysis...")
        main()
