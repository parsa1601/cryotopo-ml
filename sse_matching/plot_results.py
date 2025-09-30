import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import seaborn as sns

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


