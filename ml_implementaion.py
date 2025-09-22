import os
import sys
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from protein_visualization import ProteinVisualizer

sys.path.append(f"{os.path.dirname(os.getcwd())}")
import warnings

warnings.filterwarnings("ignore")


"The address of Protein records"
CSV_DATASET = "Archive/"


class ProteinAssignmentUsingMultipleML:
    def __init__(self, report_file="direction_analysis_report.txt"):
        """
        Initializing 3 classifiers to use all of them!
        """
        self.svm_linear = svm.SVC(kernel="linear")
        self.svm_rbf = svm.SVC(kernel="rbf")
        self.random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        self.visualizer = ProteinVisualizer()
        self.report_file = report_file

        self.overall_direction_stats = {
            "total_directions": 0,
            "correct_directions": 0,
            "protein_results": [],
        }

    def print_and_save(self, message):
        print(message)
        with open(self.report_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def dtw_distance(self, ts1, ts2):
        """
        This function computes the Dynamic Time Warping (DTW) distance between two time series.
        For 3D coordinates, it uses Euclidean distance between points.
        """
        n, m = len(ts1), len(ts2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # Calculate Euclidean distance between 3D points
                diff = ts1[i - 1] - ts2[j - 1]
                cost = np.sqrt(np.sum(diff**2))
                last_min = np.min(
                    [
                        dtw_matrix[i - 1, j],
                        dtw_matrix[i, j - 1],
                        dtw_matrix[i - 1, j - 1],
                    ]
                )
                dtw_matrix[i, j] = cost + last_min
        return dtw_matrix[n, m]

    def determine_direction_with_dtw(self, model_coords, stick_coords):
        """
        This function determines the optimal direction of a stick by comparing it with the model's coordinates using DTW.
        """
        # Stick in forward direction
        forward_distance = self.dtw_distance(model_coords, stick_coords)

        # Stick in backward direction
        backward_coords = np.flipud(stick_coords)
        backward_distance = self.dtw_distance(model_coords, backward_coords)

        return 1 if forward_distance <= backward_distance else -1

    def analyze_best_mappings(
        self,
        protein_name,
        best_algorithm,
        X_train,
        y_train,
        X_test,
        y_test,
        test_to_train_map,
        direction_mapping,
    ):
        """
        This function analyzes the best mappings and determines the direction of the sticks.
        Also calculates and reports direction detection accuracy.
        """
        self.print_and_save(
            f"\n--- Direction Analysis for {protein_name} using {best_algorithm} ---"
        )

        # Get predictions from the best algorithm
        if best_algorithm == "SVM Linear":
            y_pred = self.svm_linear.predict(X_test)
        elif best_algorithm == "SVM RBF":
            y_pred = self.svm_rbf.predict(X_test)
        else:  # Random Forest
            y_pred = self.random_forest.predict(X_test)

        correct_directions = 0
        total_directions = 0
        direction_results = []

        unique_sticks = np.unique(y_test)
        for stick_label in unique_sticks:
            # Find the predicted train label for this stick
            stick_indices = np.where(y_test == stick_label)[0]
            if len(stick_indices) > 0:
                predicted_train_label = y_pred[stick_indices[0]]

                # Get the actual train label from the mapping
                actual_train_label = test_to_train_map.get(stick_label)

                if (
                    actual_train_label is not None
                    and predicted_train_label == actual_train_label
                ):
                    # Extract coordinates for the model and the stick
                    model_coords = X_train[y_train == actual_train_label]
                    stick_coords = X_test[y_test == stick_label]

                    detected_direction = self.determine_direction_with_dtw(
                        model_coords, stick_coords
                    )

                    # Get actual direction from topology
                    actual_direction = direction_mapping.get(stick_label)

                    if actual_direction is not None:
                        total_directions += 1
                        is_correct = detected_direction == actual_direction
                        if is_correct:
                            correct_directions += 1

                        direction_results.append(
                            {
                                "stick": stick_label,
                                "model": actual_train_label,
                                "detected": detected_direction,
                                "actual": actual_direction,
                                "correct": is_correct,
                            }
                        )

                        self.print_and_save(
                            f"Stick {stick_label} -> Model {actual_train_label}: Detected={detected_direction}, Actual={actual_direction}, Correct={is_correct}"
                        )

        if total_directions > 0:
            direction_accuracy = (correct_directions / total_directions) * 100
            self.print_and_save("\n--- Direction Detection Results ---")
            self.print_and_save(f"Total directions analyzed: {total_directions}")
            self.print_and_save(f"Correctly detected directions: {correct_directions}")
            self.print_and_save(
                f"Direction detection accuracy: {direction_accuracy:.2f}%"
            )

            self.update_overall_direction_stats(
                protein_name, correct_directions, total_directions, direction_accuracy
            )
        else:
            self.print_and_save("\nNo direction information available for analysis.")
            # Still track proteins with no direction data
            self.update_overall_direction_stats(protein_name, 0, 0, 0.0)

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
                    mappings, direction_mapping = self.read_mapping_topology(
                        protein, csv_path, mode
                    )
                    test_to_train_map = {
                        test_label: train_label for train_label, test_label in mappings
                    }
                    X_train, X_test, y_train, y_test, num_train, num_test = (
                        self.generate_protein_helix_stick(protein, csv_path)
                    )
                    print(f"\nProtein: {protein}")
                    print(f"#Helices: (Train Classes): {num_train}")
                    print(f"#Sticks: (Test Classes): {num_test}")
                    self.train_and_evaluation(
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        test_to_train_map,
                        direction_mapping,
                        protein,
                        "Helix",
                    )
                else:
                    print(
                        f"Error matching in the number of sticks and helices for protein: {protein}"
                    )

            if mode == "Strand":
                if self.generate_protein_strand_stick(protein, csv_path) is not None:
                    mappings, direction_mapping = self.read_mapping_topology(
                        protein, csv_path, mode
                    )
                    test_to_train_map = {
                        test_label: train_label for train_label, test_label in mappings
                    }
                    X_train, X_test, y_train, y_test, num_train, num_test = (
                        self.generate_protein_strand_stick(protein, csv_path)
                    )
                    print(f"\nProtein: {protein}")
                    print(f"#Strands: (Train Classes): {num_train}")
                    print(f"#Sticks: (Test Classes): {num_test}")
                    self.train_and_evaluation(
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        test_to_train_map,
                        direction_mapping,
                        protein,
                        "Strand",
                    )
                else:
                    print(
                        f"Error matching in the number of sticks and helices for protein: {protein}"
                    )

    def train_and_evaluation(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        test_to_train_map,
        direction_mapping,
        protein_name="Unknown",
        structure_type="Helix",
    ):
        """
        This function is being called in the "train_with_all_algorithms" function.
        It trains the model on all 3 algorithms and finally test it.
        Also performs direction detection analysis and reports accuracy.
        """
        algorithms = [
            ("SVM Linear", self.svm_linear),
            ("SVM RBF", self.svm_rbf),
            ("Random Forest", self.random_forest),
        ]

        accuracies = {}

        for name, classifier in algorithms:
            print(f"\n--- {name} Results ---")

            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            accuracy = self.calculate_mapped_accuracy(y_test, y_pred, test_to_train_map)
            accuracies[name] = accuracy
            print(f"Accuracy: {accuracy:.4f}")

            # Plot 3D visualization with cylindrical structures and SVM decision boundaries
            # if 'SVM Linear' in name:
            #     self.visualizer.plot_3d_cylindrical_structures_with_svm(X_train, y_train, protein_name, structure_type, classifier)

        best_algorithm = max(accuracies, key=accuracies.get)
        print(f"Best Algorithm: {best_algorithm} ({accuracies[best_algorithm]:.4f})")
        print("-" * 50)

        self.analyze_best_mappings(
            protein_name,
            best_algorithm,
            X_train,
            y_train,
            X_test,
            y_test,
            test_to_train_map,
            direction_mapping,
        )

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
            expected_train_label = mapping.get(true_test_label, None)

            if (
                expected_train_label is not None
                and pred_train_label == expected_train_label
            ):
                correct += 1

        return correct / total if total > 0 else 0

    def read_mapping_topology(self, protein_name, csv_path, mode):
        """
        reading topology csv file based on the protein name to create the mapping needed for the accuracy calculate.
        Since the path for strands and helices is different, it checks first!
        Returns both mapping and direction information.
        """
        if mode == "Helix":
            topology_record = f"{csv_path}/{protein_name}/{protein_name}_Topology.csv"
            topology_df = pd.read_csv(topology_record, header=None)
        if mode == "Strand":
            topology_record = (
                f"{csv_path}STRANDS/{protein_name}/Sheet/{protein_name}_Topology.csv"
            )
            topology_df = pd.read_csv(topology_record, header=None)

        # columns: train_label, test_label, direction
        mapping = topology_df.iloc[:, :2].to_numpy()

        # Create direction mapping (test_label -> actual_direction)
        direction_mapping = {}
        for _, row in topology_df.iterrows():
            if len(row) >= 3 and pd.notna(row.iloc[2]):
                test_label = int(row.iloc[1])
                actual_direction = int(row.iloc[2])
                direction_mapping[test_label] = actual_direction

        return mapping, direction_mapping

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
        strands_records = (
            f"{csv_path}STRANDS/{protein_name}/Sheet/{protein_name}_Strands.csv"
        )
        stick_records = (
            f"{csv_path}STRANDS/{protein_name}/Sheet/{protein_name}_Sticks_Strands.csv"
        )

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

    def preprocess_labels(self, labels: list):
        pass

    def update_overall_direction_stats(
        self, protein_name, correct_directions, total_directions, direction_accuracy
    ):
        """
        Update the overall direction detection statistics across all proteins.
        """
        self.overall_direction_stats["total_directions"] += total_directions
        self.overall_direction_stats["correct_directions"] += correct_directions
        self.overall_direction_stats["protein_results"].append(
            {
                "protein": protein_name,
                "correct": correct_directions,
                "total": total_directions,
                "accuracy": direction_accuracy,
            }
        )

    def print_overall_direction_summary(self):
        """
        Print a summary of direction detection accuracy across all analyzed proteins.
        """
        stats = self.overall_direction_stats
        if stats["total_directions"] > 0:
            overall_accuracy = (
                stats["correct_directions"] / stats["total_directions"]
            ) * 100
            self.print_and_save("\n" + "=" * 60)
            self.print_and_save("OVERALL DIRECTION DETECTION SUMMARY")
            self.print_and_save("=" * 60)
            self.print_and_save(
                f"Total proteins analyzed: {len(stats['protein_results'])}"
            )
            self.print_and_save(
                f"Total directions analyzed: {stats['total_directions']}"
            )
            self.print_and_save(
                f"Total correct directions: {stats['correct_directions']}"
            )
            self.print_and_save(
                f"Overall direction detection accuracy: {overall_accuracy:.2f}%"
            )
            self.print_and_save("\nPer-protein results:")
            self.print_and_save("-" * 40)
            for result in stats["protein_results"]:
                self.print_and_save(
                    f"{result['protein']}: {result['correct']}/{result['total']} ({result['accuracy']:.2f}%)"
                )
            self.print_and_save("=" * 60)
        else:
            self.print_and_save("\nNo direction detection data available for summary.")


if __name__ == "__main__":
    """
    Runnign the code.
    The first list is the named of proteins with records of Helix
    The second list is the named of proteins with records of Strand
    """

    new_protein_list = [
        "1A7D",
        "1HG5",
        "1LWB",
        "1P5X",
        "1Z1L",
        "2XVV",
        "3C91",
        "3HJL",
        "3LTJ",
        "4OXW",
        "5M50",
        "6EM3",
        "1BZ4",  # "4YOK" Only Sheets data available
        "1HZ4",
        "1NG6",
        "1XQO",
        "2OVJ",
        "2Y4Z",
        "3FIN",
        "3IEE",
        "3ODS",
        "5I1M",
        "5O8O",
        "6F36",
        "1FLP",
        "1ICX",  # "4R9A" Only Sheets data available
        "1OZ9",
        "1YD0",
        "2XB5",
        "3ACW",
        "3HBE",
        "3IXV",
        "4CHV",
        "4UE4",
        "5KBU",
        "5UZB",
        "6UXW",
    ]
    # new_protein_list = ["6EM3"]

    strands_protein_list = [
        "1ICX",
        "1OZ9",
        "1YD0",
        "2Y4Z",
        "3C91",
        "4CHV",
        "4OXW",
        "4R9A",
        "4YOK",
        "5KBU",
        "5M50",
        "5O8O",
        "6EM3",
        "6UXW",
    ]

    ml_classifier = ProteinAssignmentUsingMultipleML()
    print("\n\n\n\n*************************HELIX RESULTS:**********************")
    ml_classifier.print_and_save(
        "\n\n\n\n*************************HELIX RESULTS:**********************"
    )
    ml_classifier.train_with_all_algorithms(new_protein_list, CSV_DATASET, "Helix")
    ml_classifier.print_overall_direction_summary()

    print("\n\n\n\n*************************Strands RESULTS:**********************")
    ml_classifier.print_and_save(
        "\n\n\n\n*************************Strands RESULTS:**********************"
    )
    # Reset statistics for strand analysis
    ml_classifier.overall_direction_stats = {
        "total_directions": 0,
        "correct_directions": 0,
        "protein_results": [],
    }
    ml_classifier.train_with_all_algorithms(strands_protein_list, CSV_DATASET, "Strand")
    ml_classifier.print_overall_direction_summary()

    print(f"\nDirection analysis report has been saved to: {ml_classifier.report_file}")
