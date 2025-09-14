import os
import sys
import numpy as np 
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from protein_visualization import ProteinVisualizer
sys.path.append(f'{os.path.dirname(os.getcwd())}')
import warnings
warnings.filterwarnings("ignore")  




"The address of Protein records"
CSV_DATASET = 'Archive/'


class ProteinAssignmentUsingMultipleML():
    def __init__(self):
        """
        Initializing 3 classifiers to use all of them!
        """
        self.svm_linear = svm.SVC(kernel="linear")
        self.svm_rbf = svm.SVC(kernel="rbf")
        self.random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        self.visualizer = ProteinVisualizer()
        
    def train_with_all_algorithms(self, proteins_list, csv_path, mode: str):
        """
        This function trains and tests the proteins list
        it first decides to train the model on helix records or strand records. When creating 
        x and y records, it trains the 3 models and tests the classes for stick records and prints the results.
        
        protein_list: 
        this is the list of proteins we want to train and test the model on them.

        csv_path:
        the path containing proteins and their records

        mode:
        if the mode is Helix, the given proteins_list should contain helix records of helix and 
        sticks exactly in their directory.
        if the mode is Strand, the given proteins_list should contain a folder named "Sheet" and have 
        the records of strands and sticks in that folder/
        """
        print("=== Comparing All Algorithms ===")
        for protein in proteins_list:
            if mode == "Helix":
                if self.generate_protein_helix_stick(protein, csv_path) is not None:
                    mappings = self.read_mapping_topology(protein, csv_path, mode)
                    test_to_train_map = {test_label: train_label for train_label, test_label in mappings}
                    X_train, X_test, y_train, y_test, num_train, num_test = self.generate_protein_helix_stick(protein, csv_path)                    
                    print(f"\nProtein: {protein}")
                    print(f"#Helices: (Train Classes): {num_train}")
                    print(f"#Sticks: (Test Classes): {num_test}")
                    self.train_and_evaluation(X_train, X_test, y_train, y_test, test_to_train_map, protein, "Helix")
                else:
                    print(f"Error matching in the number of sticks and helices for protein: {protein}")
            

            if mode == "Strand":
                if self.generate_protein_strand_stick(protein, csv_path) is not None:
                    mappings = self.read_mapping_topology(protein, csv_path, mode)
                    test_to_train_map = {test_label: train_label for train_label, test_label in mappings}
                    X_train, X_test, y_train, y_test, num_train, num_test = self.generate_protein_strand_stick(protein, csv_path)                    
                    print(f"\nProtein: {protein}")
                    print(f"#Strands: (Train Classes): {num_train}")
                    print(f"#Sticks: (Test Classes): {num_test}")
                    self.train_and_evaluation(X_train, X_test, y_train, y_test, test_to_train_map, protein, "Strand")
                else:
                    print(f"Error matching in the number of sticks and helices for protein: {protein}")


    def train_and_evaluation(self, X_train, X_test, y_train, y_test, test_to_train_map, protein_name="Unknown", structure_type="Helix"):
        """
        This function is being called in the "train_with_all_algorithms" function.
        It trains the model on all 3 algorithms and finally test it.
        """
        algorithms = [
            ('SVM Linear', self.svm_linear),
            ('SVM RBF', self.svm_rbf),
            ('Random Forest', self.random_forest)
        ]
        
        accuracies = {}
        
        for name, classifier in algorithms:
            print(f"\n--- {name} Results ---")
            
            # Train the classifier
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            
            # Calculate mapped accuracy
            accuracy = self.calculate_mapped_accuracy(y_test, y_pred, test_to_train_map)
            accuracies[name] = accuracy
            print(f"Accuracy: {accuracy:.4f}")
            
            # Plot 3D visualization for SVM models
            if 'SVM' in name:
                self.visualizer.plot_3d_svm_model(X_train, y_train, classifier, protein_name, f"({name})")
            
            # Plot cylindrical structures for the first SVM model (to avoid duplicates)
            if name == 'SVM Linear':
                self.visualizer.plot_3d_cylindrical_structures(X_train, y_train, protein_name, structure_type)
            
            # # Map predictions for proper evaluation
            # y_test_mapped = np.array([test_to_train_map.get(label, label) for label in y_test])
            
            # # Classification Report
            # print("\nClassification Report:")
            # print(classification_report(y_test_mapped, y_pred, zero_division=0))
            
            # Confusion Matrix
            # print("Confusion Matrix:")
            # cm = confusion_matrix(y_test_mapped, y_pred)
            # print(cm)
            # print()
        
        # Find best performing algorithm
        best_algorithm = max(accuracies, key=accuracies.get)
        print(f"Best Algorithm: {best_algorithm} ({accuracies[best_algorithm]:.4f})")
        print("-" * 50)



    def calculate_mapped_accuracy(self, y_test, y_pred, mapping):
        """
        Since the topology file is like a mapping to tell which stick is related to which class of train class,
        this function uses this mapping to decide the accuracy.

        If the relation between classes and sticks was direct, we did not need this class, but in this scenarion
        we calculate the accuracy based on the mapping in the topology

        y_test:
        the stick in the sticks file csv (4th column of stick csv file)

        y_pred:
        the class generated by the model based on the stick's dimension

        mapping:
        maps the y_pred to currect class and then calculated the performance of model
        """
        correct = 0
        total = len(y_test)
        
        for true_test_label, pred_train_label in zip(y_test, y_pred):
            # Get the expected train label for this test label
            expected_train_label = mapping.get(true_test_label, None)
            
            # Check if prediction matches the expected mapping
            if expected_train_label is not None and pred_train_label == expected_train_label:
                correct += 1
        
        return correct / total if total > 0 else 0


    def read_mapping_topology(self, protein_name, csv_path, mode):
        """
        reading topology csv file based on the protein name to create the mapping needed for the accuracy calculate.
        Since the path for strands and helices is different, it checks first!
        """
        if mode == "Helix":
            topology_record = f"{csv_path}/{protein_name}/{protein_name}_Topology.csv"
            topology_df = pd.read_csv(topology_record, header=None)
        if mode == "Strand":
            topology_record = f"{csv_path}STRANDS/{protein_name}/Sheet/{protein_name}_Topology.csv"
            topology_df = pd.read_csv(topology_record, header=None)


        mapping = topology_df.iloc[:, :2].to_numpy()
        return mapping
        

    def generate_protein_helix_stick(self, protein_name: str, csv_path):
        """
        This function reads the csv files for helices. To test the code, the proteins should be in the csv_path with 
        the following naming:
        PROTEIN NAME IN ALL CAPITAL_Helices.csv
        PROTEIN NAME IN ALL CAPITAL_Stick.csv
        """
        helix_records = f"{csv_path}/{protein_name}/{protein_name}_Helices.csv"
        stick_records = f"{csv_path}/{protein_name}/{protein_name}_Sticks.csv"

        if not os.path.exists(stick_records):
            stick_records = f"{csv_path}/{protein_name}/{protein_name}_Stick.csv"

        # Helix records
        helix_df = pd.read_csv(helix_records, header=None)
        helices_datapoints = helix_df.iloc[:, :3].to_numpy()
        classes = helix_df.iloc[:, 3].to_numpy().astype(int)
        k_helices = len(np.unique(classes))

        # Stick records
        stick_df = pd.read_csv(stick_records, header=None)
        cryo_datapoints = stick_df.iloc[:, :3].to_numpy()
        sticks = stick_df.iloc[:, 3].to_numpy().astype(int)
        k_stick = len(np.unique(sticks))

        return helices_datapoints, cryo_datapoints, classes, sticks, k_helices, k_stick


    def generate_protein_strand_stick(self, protein_name: str, csv_path):
        """
        This function reads the csv files for strands. To test the code, the proteins should be in the csv_path with 
        the following naming:
        these proteins should be in a folder named STRANDS/ 
        PROTEIN NAME IN ALL CAPITAL_Strands.csv
        PROTEIN NAME IN ALL CAPITAL_Sticks_Strands.csv
        """
        strands_records = f"{csv_path}STRANDS/{protein_name}/Sheet/{protein_name}_Strands.csv"
        stick_records = f"{csv_path}STRANDS/{protein_name}/Sheet/{protein_name}_Sticks_Strands.csv"

        # Strand records
        strand_df = pd.read_csv(strands_records, header=None)
        strands_datapoints = strand_df.iloc[:, :3].to_numpy()
        classes = strand_df.iloc[:, 3].to_numpy().astype(int)
        k_strands = len(np.unique(classes))

        # Stick records
        stick_df = pd.read_csv(stick_records, header=None)
        cryo_datapoints = stick_df.iloc[:, :3].to_numpy()
        sticks = stick_df.iloc[:, 3].to_numpy().astype(int)
        k_stick = len(np.unique(sticks))

        return strands_datapoints, cryo_datapoints, classes, sticks, k_strands, k_stick

    def remap_labels(labels, label_mapping):
        return np.array([label_mapping.get(label, label) for label in labels])


    def preprocess_labels(self, labels : list):
        pass


if __name__ == "__main__":
    """
    Runnign the code.
    The first list is the named of proteins with records of Helix
    The second list is the named of proteins with records of Strand
    """

    new_protein_list = [
        "1A7D", "1HG5", "1LWB", "1P5X", "1Z1L", "2XVV", "3C91", 
        "3HJL", "3LTJ", "4OXW", "5M50", "6EM3", "1BZ4",           #"4YOK" Only Sheets data available   
        "1HZ4", "1NG6", "1XQO", "2OVJ", "2Y4Z", "3FIN", "3IEE", 
        "3ODS", "5I1M", "5O8O", "6F36", "1FLP", "1ICX",           #"4R9A" Only Sheets data available
        "1OZ9", "1YD0", "2XB5", "3ACW", "3HBE", "3IXV", "4CHV", 
        "4UE4", "5KBU", "5UZB", "6UXW"
    ]


    strands_protein_list = [
        "1ICX", "1OZ9", "1YD0", "2Y4Z", "3C91", "4CHV", "4OXW", 
        "4R9A", "4YOK", "5KBU", "5M50", "5O8O", "6EM3", "6UXW"
    ]

    ml_classifier = ProteinAssignmentUsingMultipleML()
    print("\n\n\n\n*************************HELIX RESULTS:**********************")
    ml_classifier.train_with_all_algorithms(new_protein_list, CSV_DATASET, "Helix")

    print("\n\n\n\n*************************Strands RESULTS:**********************")
    ml_classifier.train_with_all_algorithms(strands_protein_list, CSV_DATASET, "Strand")
