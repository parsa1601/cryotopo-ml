# ruff: noqa: E402
import os
import sys
import json
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sse_matching.lptd_method.lptd import LPTDMethod
from sse_matching.plot_results import plot_runtime_comparison, plot_accuracy_charts, plot_metrics_bar_chart
from sse_matching.config import HELIX_PROTEIN_LIST, STRAND_PROTEIN_LIST, CSV_DATASET
from sse_matching.protein_trainer import ProteinTrainer


def run_lptd_comparison_workflow(best_ml_algorithm="SVM RBF"):
    """
    Runs the full comparison workflow:
    1. Runs ML training (ProteinTrainer) to get SVM runtimes and accuracy.
    2. Runs LPTD method on the same proteins.
    3. Compares runtimes and accuracies and generates plots.
    """
    print(f"\n{'='*60}")
    print("STARTING LPTD COMPARISON WORKFLOW")
    print(f"{'='*60}")

    # 1. Initialize Trainer and Run ML Training
    trainer = ProteinTrainer(csv_path=CSV_DATASET, use_grid_search=False)
    trainer.file_handler.reset_report_file()

    print("\n--- Running ML Training for Helices ---")
    trainer.train_with_all_algorithms(HELIX_PROTEIN_LIST, "Helix")
    
    print("\n--- Running ML Training for Strands ---")
    trainer.train_with_all_algorithms(STRAND_PROTEIN_LIST, "Strand")

    # 2. Run LPTD and Compare
    lptd = LPTDMethod()

    # Combine lists for processing
    all_proteins = []
    for p in HELIX_PROTEIN_LIST:
        all_proteins.append((p, "Helix"))
    for p in STRAND_PROTEIN_LIST:
        all_proteins.append((p, "Strand"))

    print(f"\n{'='*60}")
    print(f"RUNNING LPTD METHOD AND COMPARING WITH {best_ml_algorithm}")
    print(f"{'='*60}")

    for protein, mode in all_proteins:
        print(f"Processing {protein} ({mode})...")
        try:
            # Load Data
            if mode == "Helix":
                result = trainer.data_loader.generate_protein_helix_stick(protein)
            else:
                result = trainer.data_loader.generate_protein_strand_stick(protein)
            
            if result is None:
                print(f"Skipping {protein}: Data not found.")
                continue

            X_train, X_test, y_train, y_test, num_train, num_test = result
            
            # Get ground truth mapping for LPTD and evaluation
            mapping, _ = trainer.data_loader.read_mapping_topology(protein, mode)
            
            # --- Run LPTD ---
            topology, lptd_runtime = lptd.run(
                X_train, X_test, y_train, y_test, mode=mode, 
                run_dtw=False, ground_truth_mapping=mapping
            )

            # Convert topology to y_pred for evaluation
            stick_to_structure_pred = {}
            key_structure = "num_helix" if mode == "Helix" else "num_strand"
            
            for item in topology:
                s_id = item["num_stick"]
                h_id = item[key_structure]
                if s_id != 0:
                    stick_to_structure_pred[s_id] = h_id

            y_pred_lptd = np.zeros_like(y_test)
            for i, s_id in enumerate(y_test):
                if s_id in stick_to_structure_pred:
                    y_pred_lptd[i] = stick_to_structure_pred[s_id]
                else:
                    y_pred_lptd[i] = -1  # Unassigned


            # Calculate Metrics LPTD
            confusion_matrix_lptd, metrics_lptd = trainer.evaluation_metrics.calculate_custom_metrics(
                y_test, y_pred_lptd, mapping
            )
            
            # Add LPTD to performance report
            trainer.ml_classifiers.performance_report[protein][mode]["LPTD"]["runtime"] = lptd_runtime
            trainer.ml_classifiers.performance_report[protein][mode]["LPTD"]["confusion_matrix_detailed"] = confusion_matrix_lptd
            trainer.ml_classifiers.performance_report[protein][mode]["LPTD"].update(metrics_lptd)
            
        except Exception as e:
            print(f"Error processing {protein}: {e}")
            continue

    # Generate Comparison Plot
    plot_runtime_comparison(trainer.ml_classifiers.performance_report, best_ml_algorithm)
    print("\nRuntime comparison chart generated successfully.")
    
    # Generate Accuracy Charts including LPTD
    print("\nGenerating accuracy charts including LPTD...")
    plot_accuracy_charts(trainer.ml_classifiers.performance_report, "f1_measure")
    plot_metrics_bar_chart(trainer.ml_classifiers.performance_report)
    print("Accuracy charts generated successfully.")

    with open("LPTD_Results.json", "w") as json_file:
        json.dump(
            trainer.ml_classifiers.performance_report,
            json_file,
            indent=4,
        )
    print("\nResults saved to: LPTD_Results.json")

if __name__ == "__main__":
    run_lptd_comparison_workflow()
