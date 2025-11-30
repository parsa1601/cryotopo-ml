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
from run_lptd_comparison import run_lptd_comparison_workflow

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
            performance_report = json.load(json_file)

        print(f"Loaded data from {json_file_path}")
        print("Generating charts...")

        plot_accuracy_charts(performance_report, 'f1_measure')
        plot_metrics_bar_chart(performance_report)
        plot_error_rate_line_chart(performance_report)

        print("All charts have been generated successfully!")

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file: {e}")
    except Exception as e:
        print(f"Error generating charts: {e}")


def run_direction_analysis_from_results(json_file_path="Final_Results.json"):
    """
    Load results from JSON file, determine best overall algorithm,
    and run direction analysis using that algorithm.
    
    Args:
        json_file_path (str): Path to the Final_Results.json file
    """
    try:
        if not os.path.exists(json_file_path):
            print(f"Error: {json_file_path} not found!")
            return

        with open(json_file_path, "r") as json_file:
            performance_report = json.load(json_file)

        print(f"Loaded data from {json_file_path}")
        
        # Calculate best algorithm based on F1-measure
        algorithm_f1_scores = defaultdict(list)
        
        for protein, structures in performance_report.items():
            for structure_type, algorithms in structures.items():
                for algorithm, metrics in algorithms.items():
                    if 'f1_measure' in metrics:
                        algorithm_f1_scores[algorithm].append(metrics['f1_measure'])
        
        # Calculate average F1 score for each algorithm
        algorithm_averages = {}
        for algorithm, f1_scores in algorithm_f1_scores.items():
            if f1_scores:
                algorithm_averages[algorithm] = sum(f1_scores) / len(f1_scores)
        
        # Find the best algorithm
        if algorithm_averages:
            best_algorithm = max(algorithm_averages, key=algorithm_averages.get)
            best_f1_score = algorithm_averages[best_algorithm]
            
            print(f"\n{'='*60}")
            print("BEST ALGORITHM DETERMINED FROM RESULTS")
            print(f"{'='*60}")
            print("Average F1-measure by algorithm:")
            for algorithm in sorted(algorithm_averages.keys()):
                print(f"  {algorithm:<20}: {algorithm_averages[algorithm]:.4f}")
            print(f"\nBest Algorithm: {best_algorithm} (Avg F1-measure: {best_f1_score:.4f})")
            print(f"{'='*60}\n")
            
            # Now run direction analysis with the best algorithm
            trainer = ProteinTrainer(csv_path=CSV_DATASET, use_grid_search=False)
            trainer.file_handler.reset_report_file()
            
            # Run for helices
            helix_proteins = [p for p, data in performance_report.items() if 'Helix' in data]
            if helix_proteins:
                trainer.file_handler.print_and_save(
                    f"\n\n{'='*60}\nDIRECTION ANALYSIS - HELIX (using {best_algorithm})\n{'='*60}"
                )
                trainer.run_direction_analysis_with_best_algorithm(
                    helix_proteins, "Helix", best_algorithm
                )
                trainer.print_overall_direction_summary()
            
            # Run for strands
            strand_proteins = [p for p, data in performance_report.items() if 'Strand' in data]
            if strand_proteins:
                trainer.file_handler.print_and_save(
                    f"\n\n{'='*60}\nDIRECTION ANALYSIS - STRAND (using {best_algorithm})\n{'='*60}"
                )
                trainer.reset_direction_stats()
                trainer.run_direction_analysis_with_best_algorithm(
                    strand_proteins, "Strand", best_algorithm
                )
                trainer.print_overall_direction_summary()
            
            print(f"\nDirection analysis report has been saved to: {trainer.file_handler.report_file}")
        else:
            print("No F1-measure data available for algorithm selection.")
            
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file: {e}")
    except Exception as e:
        print(f"Error in direction analysis: {e}")
        import traceback
        traceback.print_exc()


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
    performance_report = trainer.train_with_all_algorithms(
        HELIX_PROTEIN_LIST, "Helix"
    )
    trainer.find_globally_best_parameters()

    trainer.file_handler.print_and_save(
        "\n\n\n\n*************************STRAND RESULTS:**********************"
    )

    performance_report = trainer.train_with_all_algorithms(
        STRAND_PROTEIN_LIST, "Strand"
    )
    trainer.find_globally_best_parameters()

    plot_accuracy_charts(performance_report, 'f1_measure')
    plot_metrics_bar_chart(performance_report)
    plot_error_rate_line_chart(performance_report)
    
    with open("Final_Results.json", "w") as json_file:
        json.dump(
            convert_dd_to_dict(performance_report),
            json_file,
            indent=4,
            default=make_serializable,
        )
    print("\nResults saved to: Final_Results.json")
    print("\nTo run direction analysis with the best algorithm, use:")
    print("  python main.py --direction-analysis")


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
        "--direction-analysis",
        action="store_true",
        help="Run direction analysis using best algorithm from Final_Results.json",
    )
    parser.add_argument(
        "--lptd-comparison",
        action="store_true",
        help="Run LPTD comparison using results from Final_Results.json",
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
    elif args.direction_analysis:
        print("Running direction analysis with best algorithm from results...")
        run_direction_analysis_from_results(args.json_file)
    elif args.lptd_comparison:
        print("Running LPTD comparison workflow...")
        run_lptd_comparison_workflow()
    else:
        print("Running full protein structure analysis...")
        main()
