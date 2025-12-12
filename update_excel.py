#!/usr/bin/env python3
"""
Script to update the Proteins List.xlsx file with metrics from LPTD_Results.json
"""

import sys
from pathlib import Path
import json
import pandas as pd


def update_proteins_excel(json_file_path, excel_file_path):
    """
    Process LPTD_Results.json and update the Proteins List.xlsx file with confusion matrix
    metrics and performance measures.
    
    Args:
        json_file_path (str): Path to the LPTD_Results.json file
        excel_file_path (str): Path to the Proteins List.xlsx file
    
    Returns:
        pd.DataFrame: Updated dataframe with all metrics
    """
    # Load the JSON data
    with open(json_file_path, 'r') as f:
        final_accuracy_report = json.load(f)
    
    # Load existing Excel file
    df = pd.read_excel(excel_file_path)
    
    # Define all methods to process
    methods = ['SVM Linear', 'SVM RBF', 'Random Forest', 'Voronoi (1N KNN)', 'LPTD']
    
    # Create Excel writer object
    with pd.ExcelWriter("Proteins List.xlsx", engine='openpyxl') as writer:
        # First page: Keep original data as is
        df.to_excel(writer, sheet_name='Original', index=False)
        print("Original sheet preserved")
        
        # Create a page for each method
        for method in methods:
            data_rows = []
            
            for protein in df.iloc[:, 0]:  # Assuming first column has protein names
                # Initialize accumulators for this protein
                total_tp = 0
                total_tn = 0
                total_fp = 0
                total_fn = 0
                precision_values = []
                recall_values = []
                f1_values = []
                accuracy_values = []
                mismatch_values = []
                train_times = []
                test_times = []
                
                if protein in final_accuracy_report:
                    protein_data = final_accuracy_report[protein]
                    
                    # Process each structure type (Helix, Strand)
                    for structure_type in ['Helix', 'Strand']:
                        if structure_type in protein_data:
                            structure_data = protein_data[structure_type]
                            
                            if method in structure_data:
                                method_data = structure_data[method]
                                
                                # Sum confusion matrix values
                                if 'confusion_matrix_detailed' in method_data:
                                    detailed = method_data['confusion_matrix_detailed']
                                    total_tp += detailed.get('tp', 0)
                                    total_tn += detailed.get('tn', 0)
                                    total_fp += detailed.get('fp', 0)
                                    total_fn += detailed.get('fn', 0)
                                    
                                    # For LPTD, accuracy might be calculated differently
                                    if 'accuracy' in method_data:
                                        accuracy_values.append(method_data['accuracy'] * 100)
                                    precision_values.append(method_data.get('precision', 0))
                                    recall_values.append(method_data.get('recall', 0))
                                    f1_values.append(method_data.get('f1_measure', 0))
                                    mismatch_values.append(method_data.get('mismatch_rate', 0))
                                    
                                    # LPTD uses 'runtime' instead of train/test time
                                    if method == 'LPTD':
                                        test_times.append(method_data.get('runtime', 0))
                                    else:
                                        train_times.append(method_data.get('train_time', 0))
                                        test_times.append(method_data.get('test_time', 0))
                
                # Calculate averages
                avg_precision = sum(precision_values) / len(precision_values) if precision_values else 0
                avg_recall = sum(recall_values) / len(recall_values) if recall_values else 0
                avg_f1 = sum(f1_values) / len(f1_values) if f1_values else 0
                avg_accuracy = sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0
                avg_mismatch = sum(mismatch_values) / len(mismatch_values) if mismatch_values else 0
                train_time = sum(train_times)
                test_time = sum(test_times)
                
                data_rows.append({
                    'Protein': protein,
                    'TP': total_tp,
                    'TN': total_tn,
                    'FP': total_fp,
                    'FN': total_fn,
                    'Precision (%)': round(avg_precision, 2),
                    'Recall (%)': round(avg_recall, 2),
                    'F1-Measure (%)': round(avg_f1, 2),
                    'Accuracy (%)': round(avg_accuracy, 2),
                    'Mismatch Rate (%)': round(avg_mismatch, 2),
                    'Train Time (s)': round(train_time, 6),
                    'Test Time (s)': round(test_time, 6)
                })
            
            # Create dataframe for this method
            method_df = pd.DataFrame(data_rows)
            
            # Use a valid sheet name (Excel has 31 char limit)
            sheet_name = method.replace(' ', '_')[:31]
            method_df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Sheet '{sheet_name}' created with {len(method_df)} proteins")
    
    print(f"Excel file updated successfully: {excel_file_path}")
    print(f"Total methods processed: {len(methods)}")
    
    # Return the SVM RBF dataframe for backward compatibility
    return pd.DataFrame(data_rows)


def main():
    # Define file paths
    json_file = Path(__file__).parent / 'LPTD_Results.json'
    excel_file = Path(__file__).parent / 'proteins_list.xlsx'
    
    # Check if files exist
    if not json_file.exists():
        print(f"Error: {json_file} not found!")
        return 1
    
    if not excel_file.exists():
        print(f"Error: {excel_file} not found!")
        return 1
    
    print("Updating Excel file with metrics from LPTD_Results.json...")
    print(f"JSON file: {json_file}")
    print(f"Excel file: {excel_file}")
    print("-" * 60)
    
    # Update the Excel file
    df = update_proteins_excel(str(json_file), str(excel_file))
    
    print("-" * 60)
    print("\nPreview of updated data:")
    print(df.head(10))
    print(f"\nColumns: {', '.join(df.columns)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
