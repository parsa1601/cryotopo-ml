import os
import sys
import json
import numpy as np
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.getcwd()))

from sse_matching.data_loader import DataLoader
from sse_matching.lptd_method.lptd import LPTDMethod
from sse_matching.evaluation_metrics import EvaluationMetrics
from sse_matching.config import HELIX_PROTEIN_LIST, STRAND_PROTEIN_LIST

def run_lptd_comparison():
    data_loader = DataLoader()
    lptd = LPTDMethod()
    metrics_calc = EvaluationMetrics()
    
    results = {}
    
    # Process Helices
    print("Processing Helices...")
    for protein in HELIX_PROTEIN_LIST:
        print(f"Running LPTD for {protein} (Helix)...")
        try:
            # Load Data
            helices_datapoints, cryo_datapoints, classes, sticks, k_helices, k_stick = \
                data_loader.generate_protein_helix_stick(protein)
                
            # Run LPTD
            topology, runtime = lptd.run(helices_datapoints, cryo_datapoints, classes, sticks, mode="Helix")
            
            # Convert topology to y_pred for evaluation
            # y_test is 'sticks' (stick IDs for each voxel)
            # y_train is 'classes' (helix IDs for each voxel)
            
            # Create a map from Stick ID -> Predicted Helix ID
            stick_to_helix_pred = {}
            for item in topology:
                s_id = item['num_stick']
                h_id = item['num_helix']
                if s_id != 0:
                    stick_to_helix_pred[s_id] = h_id
            
            # Generate y_pred
            y_pred = np.zeros_like(sticks)
            for i, s_id in enumerate(sticks):
                if s_id in stick_to_helix_pred:
                    y_pred[i] = stick_to_helix_pred[s_id]
                else:
                    y_pred[i] = -1 # Unassigned
            
            # Get Ground Truth Mapping
            mapping, direction_mapping = data_loader.read_mapping_topology(protein, "Helix")
            # mapping is [[HelixID, StickID, Direction], ...]
            # We need StickID -> HelixID for evaluation metrics
            # EvaluationMetrics expects test_to_train_map: {StickID: HelixID}
            
            test_to_train_map = {}
            for row in mapping:
                h_id = int(row[0])
                s_id = int(row[1])
                test_to_train_map[s_id] = h_id
                
            # Calculate Metrics
            # Note: calculate_custom_metrics expects y_test (sticks), y_pred (predicted helix), y_train (helices)
            # But wait, y_pred should be in the same domain as y_train (Helix IDs).
            # Yes, stick_to_helix_pred maps Stick -> Helix.
            
            confusion_matrix, metrics = metrics_calc.calculate_custom_metrics(
                sticks, y_pred, classes, test_to_train_map
            )
            
            metrics['runtime'] = runtime
            
            if protein not in results:
                results[protein] = {}
            if "Helix" not in results[protein]:
                results[protein]["Helix"] = {}
                
            results[protein]["Helix"]["LPTD"] = metrics
            
            print(f"  Runtime: {runtime:.4f}s")
            print(f"  Accuracy: {metrics['accuracy']:.2f}%")
            
        except Exception as e:
            print(f"Error processing {protein}: {e}")
            import traceback
            traceback.print_exc()

    # Process Strands
    print("\nProcessing Strands...")
    for protein in STRAND_PROTEIN_LIST:
        print(f"Running LPTD for {protein} (Strand)...")
        try:
            # Load Data
            strands_datapoints, cryo_datapoints, classes, sticks, k_strands, k_stick = \
                data_loader.generate_protein_strand_stick(protein)
                
            # Run LPTD
            topology, runtime = lptd.run(strands_datapoints, cryo_datapoints, classes, sticks, mode="Strand")
            
            # Convert topology to y_pred
            stick_to_strand_pred = {}
            for item in topology:
                s_id = item['num_stick']
                st_id = item['num_strand']
                if s_id != 0:
                    stick_to_strand_pred[s_id] = st_id
            
            y_pred = np.zeros_like(sticks)
            for i, s_id in enumerate(sticks):
                if s_id in stick_to_strand_pred:
                    y_pred[i] = stick_to_strand_pred[s_id]
                else:
                    y_pred[i] = -1
            
            # Get Ground Truth Mapping
            mapping, direction_mapping = data_loader.read_mapping_topology(protein, "Strand")
            
            test_to_train_map = {}
            for row in mapping:
                st_id = int(row[0])
                s_id = int(row[1])
                test_to_train_map[s_id] = st_id
                
            # Calculate Metrics
            confusion_matrix, metrics = metrics_calc.calculate_custom_metrics(
                sticks, y_pred, classes, test_to_train_map
            )
            
            metrics['runtime'] = runtime
            
            if protein not in results:
                results[protein] = {}
            if "Strand" not in results[protein]:
                results[protein]["Strand"] = {}
                
            results[protein]["Strand"]["LPTD"] = metrics
            
            print(f"  Runtime: {runtime:.4f}s")
            print(f"  Accuracy: {metrics['accuracy']:.2f}%")
            
        except Exception as e:
            print(f"Error processing {protein}: {e}")
            import traceback
            traceback.print_exc()
            
    # Save Results
    with open("LPTD_Results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nResults saved to LPTD_Results.json")
    
    # Merge with Final_Results.json if exists to allow comparison plotting
    if os.path.exists("Final_Results.json"):
        with open("Final_Results.json", "r") as f:
            final_results = json.load(f)
            
        # Merge LPTD into final_results
        for protein, p_data in results.items():
            if protein not in final_results:
                final_results[protein] = {}
            
            for type_, t_data in p_data.items():
                if type_ not in final_results[protein]:
                    final_results[protein][type_] = {}
                
                final_results[protein][type_]["LPTD"] = t_data["LPTD"]
        
        with open("Final_Results_With_LPTD.json", "w") as f:
            json.dump(final_results, f, indent=4)
            
        print("Merged results saved to Final_Results_With_LPTD.json")
        
        # Generate Plots
        from sse_matching.plot_results import (
            plot_accuracy_charts,
            plot_metrics_bar_chart,
            plot_error_rate_line_chart,
        )
        
        print("Generating charts...")
        try:
            plot_accuracy_charts(final_results, 'f1_measure')
            plot_metrics_bar_chart(final_results)
            plot_error_rate_line_chart(final_results)
            print("All charts have been generated successfully!")
        except Exception as e:
            print(f"Error generating charts: {e}")

if __name__ == "__main__":
    run_lptd_comparison()
