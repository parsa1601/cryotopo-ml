import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import seaborn as sns

def calculate_combined_metrics(final_accuracy_report):
    """
    Calculate combined metrics for each protein by averaging across all methods and structure types.
    
    Args:
        final_accuracy_report (dict): Nested dictionary with structure 
                                     {protein: {structure_type: {method: {metrics}}}}
    
    Returns:
        dict: Combined metrics per protein {protein: {metric: value}}
    """
    combined_metrics = {}
    
    for protein, structures in final_accuracy_report.items():
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_mismatch_rate = 0
        count = 0
        
        for structure_type, methods in structures.items():
            for method, results in methods.items():
                if 'confusion_matrix_detailed' in results:
                    detailed = results['confusion_matrix_detailed']
                    total_precision += detailed.get('precision', 0)
                    total_recall += detailed.get('recall', 0)
                    total_f1 += detailed.get('f1_measure', 0)
                    total_mismatch_rate += detailed.get('mismatch_rate', 0)
                    count += 1
        
        if count > 0:
            combined_metrics[protein] = {
                'precision': total_precision / count,
                'recall': total_recall / count,
                'f1_measure': total_f1 / count,
                'mismatch_rate': total_mismatch_rate / count
            }
    
    return combined_metrics

def plot_metrics_bar_chart(final_accuracy_report):
    """
    Plot bar chart showing Precision, Recall, and F1-measure for each protein.
    
    Args:
        final_accuracy_report (dict): The complete accuracy report dictionary
    """
    combined_metrics = calculate_combined_metrics(final_accuracy_report)
    
    if not combined_metrics:
        print("No data available for metrics bar chart.")
        return
    
    proteins = sorted(combined_metrics.keys())
    precision_values = [combined_metrics[p]['precision'] for p in proteins]
    recall_values = [combined_metrics[p]['recall'] for p in proteins]
    f1_values = [combined_metrics[p]['f1_measure'] for p in proteins]
    
    x = np.arange(len(proteins))
    width = 0.25
    
    plt.figure(figsize=(10, 5))
    
    # Create bars
    plt.bar(x - width, precision_values, width, label='Precision', color='blue', alpha=0.8)
    plt.bar(x, recall_values, width, label='Recall', color='orange', alpha=0.8)
    plt.bar(x + width, f1_values, width, label='F1-measure', color='green', alpha=0.8)
    
    # Customize the plot
    plt.xlabel('PDB ID', fontsize=15)
    plt.ylabel('Measurements', fontsize=15)
    plt.xticks(x, proteins, rotation='vertical')
    plt.ylim(0, 100)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Place the legend outside the plot, at the top center
    plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    
    # Save the plot
    plt.savefig('protein_metrics_bar_chart.png', dpi=300, bbox_inches='tight')
    print("Metrics bar chart saved as protein_metrics_bar_chart.png")
    plt.close()

def plot_error_rate_line_chart(final_accuracy_report):
    """
    Plot line chart showing Error rate (Mismatch Rate) for each protein.
    
    Args:
        final_accuracy_report (dict): The complete accuracy report dictionary
    """
    combined_metrics = calculate_combined_metrics(final_accuracy_report)
    
    if not combined_metrics:
        print("No data available for error rate line chart.")
        return
    
    proteins = sorted(combined_metrics.keys())
    error_rates = [combined_metrics[p]['mismatch_rate'] for p in proteins]
    
    # Reduce figure width to compress x-axis
    plt.figure(figsize=(10, 5))
    
    # Create line plot with markers
    plt.plot(proteins, error_rates, marker='o', linestyle='-', linewidth=2, 
             markersize=8, color='orange', markerfacecolor='orange')
    
    # Customize the plot
    plt.xlabel('PDB ID', fontsize=15)
    plt.ylabel('Error rate (%)', fontsize=15)
    plt.xticks(rotation='vertical')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('protein_error_rate_line_chart.png', dpi=300, bbox_inches='tight')
    print("Error rate line chart saved as protein_error_rate_line_chart.png")
    plt.close()

def plot_accuracy_charts(final_accuracy_report):
    """
    Generates and saves two line charts (Helix and Strand) for protein classification accuracy.

    Args:
        final_accuracy_report (dict): A 3-level nested dictionary with the structure:
                                      {protein_name: {'Helix'/'Strand': {method: {'accuracy': float}}}}
    """
    # Define the methods and their corresponding colors
    methods = ['SVM Linear', 'SVM RBF', 'Random Forest', 'Voronoi (1N KNN)']
    colors = {'SVM Linear': 'red', 'SVM RBF': 'yellow', 'Random Forest': 'green', 'Voronoi (1N KNN)': 'blue'}

    # Separate data for Helix and Strand
    helix_data = {}
    strand_data = {}

    for protein, structures in final_accuracy_report.items():
        if 'Helix' in structures:
            helix_data[protein] = {method: data.get('accuracy') for method, data in structures['Helix'].items()}
        if 'Strand' in structures:
            strand_data[protein] = {method: data.get('accuracy') for method, data in structures['Strand'].items()}

    # A helper function to create each plot
    def create_plot(data, title):
        if not data:
            print(f"No data available to plot for {title}.")
            return

        plt.figure(figsize=(20, 10))
        
        # Get a sorted list of protein names for the x-axis
        protein_names = sorted(data.keys())
        
        min_accuracy = 1.0

        # Plot a line for each method
        for method in methods:
            accuracies = [data[protein].get(method) for protein in protein_names]
            
            # Find the overall minimum accuracy for setting the y-axis limit
            # We filter out None values before finding the min
            valid_accuracies = [acc for acc in accuracies if acc is not None]
            if valid_accuracies:
                current_min = min(valid_accuracies)
                if current_min < min_accuracy:
                    min_accuracy = current_min

            plt.plot(protein_names, accuracies, marker='o', linestyle='-', label=method, color=colors.get(method, 'black'))

        # --- Chart Customization ---
        plt.title(title, fontsize=20)
        plt.xlabel('Protein', fontsize=15)
        plt.ylabel('Accuracy', fontsize=15)
        
        # Set x-axis labels vertically
        plt.xticks(rotation='vertical')
        
        # Adjust y-axis to better visualize the results
        plt.ylim(min_accuracy * 0.99, 1.01) # Start slightly below the min accuracy
        
        plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout() # Adjust layout to make room for labels

        # --- Save the plot ---
        file_name = f"{title.lower().replace(' ', '_')}_accuracy_chart.png"
        plt.savefig(file_name)
        print(f"Chart saved as {file_name}")
        plt.close()


    create_plot(helix_data, 'Helix')
    create_plot(strand_data, 'Strand')

def plot_and_save_confusion_matrices(report_dict, base_dir='confusion_matrices'):
    """
    Generates and saves confusion matrix plots from a nested dictionary.

    This function creates a directory structure based on classification method
    and secondary structure type, then saves a plot for each confusion matrix.

    Args:
        report_dict (dict): A nested dictionary with the structure:
                            {protein_id: {structure_type: {method_name: 
                            {'accuracy': float, 'confusion_matrix': np.array}}}}
        base_dir (str): The root directory where plots will be saved.
    """
    # Loop through each protein in the main dictionary
    for protein_id, structures in report_dict.items():
        # Loop through 'Helix', 'Strand', etc. for that protein
        for structure_type, methods in structures.items():
            # Loop through 'SVM Linear', 'Random Forest', etc. for that structure
            for method_name, results in methods.items():

                # --- 1. Create the directory structure ---
                # Define the path: base_dir/Method Name/Structure Type/
                output_dir = os.path.join(base_dir, method_name, structure_type)
                
                # Create the directories if they don't already exist
                os.makedirs(output_dir, exist_ok=True)

                # --- 2. Extract data for plotting ---
                cm = results['confusion_matrix']
                accuracy = results['accuracy']
                
                # --- 3. Plot the confusion matrix ---
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Predicted Negative', 'Predicted Positive'],
                            yticklabels=['Actual Negative', 'Actual Positive'])
                
                # Add titles and labels for clarity
                plt.title(f'Confusion Matrix: {protein_id} - {structure_type}\n({method_name})', fontsize=14)
                plt.ylabel('True Label', fontsize=12)
                plt.xlabel('Predicted Label', fontsize=12)
                plt.suptitle(f'Accuracy: {accuracy:.4f}', fontsize=10, y=0.93)
                
                # --- 4. Save the plot ---
                # Define the full file path for the image
                file_path = os.path.join(output_dir, f'{protein_id}.png')
                
                # Save the figure and close the plot to free up memory
                plt.savefig(file_path, bbox_inches='tight')
                plt.close()

    print(f"✅ Plotting complete! Check the '{base_dir}' directory.")


