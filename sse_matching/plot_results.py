import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 

def calculate_combined_metrics(final_accuracy_report, best_method="SVM RBF"):
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
            results = methods[best_method]
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

def calculate_method_metrics(final_accuracy_report, metric='f1_measure'):
    """
    Calculate metrics for each method across all proteins, averaging over structure types.
    
    Args:
        final_accuracy_report (dict): Nested dictionary with structure 
                                     {protein: {structure_type: {method: {metrics}}}}
        metric (str): The metric to analyze ('f1_measure', 'precision', 'recall', 'accuracy')
    
    Returns:
        dict: Method-specific data {method: {'values': [avg_per_protein], 'proteins': [proteins], 'avg': float}}
    """
    methods = ['SVM Linear', 'SVM RBF', 'Random Forest', 'Voronoi (1N KNN)']
    method_data = {method: {'values': [], 'proteins': []} for method in methods}
    
    for protein, structures in final_accuracy_report.items():
        # For each method, collect values for all structure_types, then average
        for method in methods:
            values = []
            for structure_type, methods_dict in structures.items():
                if method in methods_dict:
                    results = methods_dict[method]
                    if metric == 'accuracy':
                        value = results.get('accuracy', 0) * 100  # Convert to percentage
                    elif 'confusion_matrix_detailed' in results:
                        value = results['confusion_matrix_detailed'].get(metric, 0)
                    else:
                        continue
                    values.append(value)
            if values:
                avg_value = sum(values) / len(values)
                method_data[method]['values'].append(avg_value)
                method_data[method]['proteins'].append(protein)
    
    # Calculate averages
    for method in method_data:
        values = method_data[method]['values']
        method_data[method]['avg'] = sum(values) / len(values) if values else 0
    
    return method_data


def print_analytical_report(final_accuracy_report, metric='f1_measure'):
    """
    Generate analytical report comparing methods.
    
    Args:
        final_accuracy_report (dict): The complete accuracy report dictionary
        metric (str): The metric to analyze ('f1_measure', 'precision', 'recall', 'accuracy')
    
    Returns:
        str: Formatted analytical report
    """
    method_data = calculate_method_metrics(final_accuracy_report, metric)
    methods = ['SVM Linear', 'SVM RBF', 'Random Forest', 'Voronoi (1N KNN)']
    
    # Calculate averages and sort methods by performance
    method_averages = [(method, method_data[method]['avg']) for method in methods if method_data[method]['values']]
    method_averages.sort(key=lambda x: x[1], reverse=True)
    
    report = f"\n{'='*60}\n"
    report += f"ANALYTICAL REPORT - {metric.upper().replace('_', ' ')}\n"
    report += f"{'='*60}\n\n"
    
    # Overall averages
    report += "AVERAGE PERFORMANCE BY METHOD:\n"
    report += "-" * 40 + "\n"
    for i, (method, avg) in enumerate(method_averages, 1):
        total_cases = len(method_data[method]['values'])
        report += f"{i}. {method:<20}: {avg:6.2f}% (n={total_cases})\n"
    print(report)

def plot_accuracy_charts(final_accuracy_report, metric='f1_measure'):
    """
    Generates and saves a combined line chart for protein classification metrics.
    Uses calculate_combined_metrics to combine Helix and Strand results.

    Args:
        final_accuracy_report (dict): A 3-level nested dictionary with the structure:
                                      {protein_name: {'Helix'/'Strand': {method: {'accuracy': float}}}}
        metric (str): The metric to plot ('f1_measure', 'precision', 'recall', 'accuracy')
    """
    print_analytical_report(final_accuracy_report, metric)
    combined_metrics = calculate_combined_metrics(final_accuracy_report)
    
    if not combined_metrics:
        print("No data available for combined metrics chart.")
        return
    
    # Define the methods and their corresponding colors
    methods = ['SVM Linear', 'SVM RBF', 'Random Forest', 'Voronoi (1N KNN)']
    colors = {'SVM Linear': 'red', 'SVM RBF': 'yellow', 'Random Forest': 'green', 'Voronoi (1N KNN)': 'blue'}

    # Calculate method-specific data for plotting
    method_data = calculate_method_metrics(final_accuracy_report, metric)
    
    # Get sorted protein names for x-axis
    proteins = sorted(combined_metrics.keys())
    
    plt.figure(figsize=(20, 10))
    
    min_metric = 100.0
    
    # Plot a line for each method
    for method in methods:
        if method in method_data and method_data[method]['values']:
            # Group values by protein
            protein_values = {}
            for i, protein in enumerate(method_data[method]['proteins']):
                if protein not in protein_values:
                    protein_values[protein] = []
                protein_values[protein].append(method_data[method]['values'][i])
            
            # Average values for each protein (combining Helix and Strand)
            method_averages = []
            for protein in proteins:
                if protein in protein_values:
                    avg_value = sum(protein_values[protein]) / len(protein_values[protein])
                    method_averages.append(avg_value / 100.0)  # Convert to 0-1 scale for plotting
                    min_metric = min(min_metric, avg_value / 100.0)
                else:
                    method_averages.append(None)
            
            plt.plot(proteins, method_averages, marker='o', linestyle='-', 
                    label=method, color=colors.get(method, 'black'), linewidth=2, markersize=6)

    # Chart Customization
    metric_label = metric.replace('_', ' ').title() if metric != 'accuracy' else 'Accuracy'
    plt.title(f'{metric_label} Performance by Protein', fontsize=20)
    plt.xlabel('Protein', fontsize=15)
    plt.ylabel(metric_label, fontsize=15)
    
    plt.xticks(rotation='vertical')
    
    plt.ylim(min_metric * 0.99, 1.01)
    
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    file_name = f"combined_{metric}_chart.png"
    plt.savefig(file_name)
    print(f"Combined chart saved as {file_name}")
    plt.close()
