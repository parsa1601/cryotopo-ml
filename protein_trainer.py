"""
Main protein trainer module.
Orchestrates the entire training and evaluation process for protein structure analysis.
"""

from data_loader import DataLoader
from ml_classifiers import MLClassifiers
from hyperparameter_optimizer import HyperparameterOptimizer
from direction_analyzer import DirectionAnalyzer
from evaluation_metrics import EvaluationMetrics
from file_utils import FileHandler
from config import PARAM_GRIDS


class ProteinTrainer:
    """Main class that orchestrates the protein structure analysis workflow."""

    def __init__(
        self,
        csv_path="Archive/",
        report_file="direction_analysis_report.txt",
        use_grid_search=False,
        best_params_file="best_hyperparameters.json",
    ):
        self.csv_path = csv_path
        self.use_grid_search = use_grid_search

        self.file_handler = FileHandler(report_file)
        self.data_loader = DataLoader(csv_path)
        self.evaluation_metrics = EvaluationMetrics()
        self.direction_analyzer = DirectionAnalyzer(self.file_handler)

        if not use_grid_search:
            best_params = self.file_handler.load_best_parameters(best_params_file)
            self.ml_classifiers = MLClassifiers(best_params)
        else:
            self.ml_classifiers = MLClassifiers()
            self.hyperparameter_optimizer = HyperparameterOptimizer(
                PARAM_GRIDS, self.file_handler
            )

    def train_with_all_algorithms(self, proteins_list, mode: str):
        """
        This function trains and tests the proteins list
        it first decides to train the model on helix records or strand records. When creating
        x and y records, it trains the 3 models and tests the classes for stick records and prints the results.

        protein_list:
        this is the list of proteins we want to train and test the model on them.

        mode:
        if the mode is Helix, the given proteins_list should contain helix records of helix and
        sticks exactly in their directory.
        if the mode is Strand, the given proteins_list should contain a folder named "Sheet" and have
        the records of strands and sticks in that folder/
        """
        print("=== Comparing All Algorithms ===")
        for protein in proteins_list:
            if mode == "Helix":
                try:
                    result = self.data_loader.generate_protein_helix_stick(protein)
                    if result is not None:
                        mappings, direction_mapping = (
                            self.data_loader.read_mapping_topology(protein, mode)
                        )
                        test_to_train_map = {
                            test_label: train_label
                            for train_label, test_label in mappings
                        }
                        X_train, X_test, y_train, y_test, num_train, num_test = result
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
                except Exception as e:
                    print(f"Error processing protein {protein}: {e}")

            elif mode == "Strand":
                try:
                    result = self.data_loader.generate_protein_strand_stick(protein)
                    if result is not None:
                        mappings, direction_mapping = (
                            self.data_loader.read_mapping_topology(protein, mode)
                        )
                        test_to_train_map = {
                            test_label: train_label
                            for train_label, test_label in mappings
                        }
                        X_train, X_test, y_train, y_test, num_train, num_test = result
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
                            f"Error matching in the number of sticks and strands for protein: {protein}"
                        )
                except Exception as e:
                    print(f"Error processing protein {protein}: {e}")

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
        if self.use_grid_search:
            optimized_algorithms, best_params_dict = (
                self.hyperparameter_optimizer.optimize_for_algorithms(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    test_to_train_map,
                    protein_name,
                    self.evaluation_metrics,
                )
            )

            algorithms = optimized_algorithms + [
                ("Voronoi (1N KNN)", self.ml_classifiers.knn)
            ]

            accuracies = {}
            for name, classifier in algorithms:
                print(f"\n--- {name} Results ---")

                if name in best_params_dict:
                    print(f"Using optimized parameters: {best_params_dict[name]}")

                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)

                accuracy = self.evaluation_metrics.calculate_mapped_accuracy(
                    y_test, y_pred, test_to_train_map
                )
                accuracies[name] = accuracy
                print(f"Accuracy: {accuracy:.4f}")

            best_algorithm = max(accuracies, key=accuracies.get)
            print(
                f"\nBest Algorithm: {best_algorithm} (Accuracy: {accuracies[best_algorithm]:.4f})"
            )

            if best_algorithm in best_params_dict:
                print(
                    f"Best parameters for {best_algorithm}: {best_params_dict[best_algorithm]}"
                )

            best_classifier = None
            for name, classifier in algorithms:
                if name == best_algorithm:
                    best_classifier = classifier
                    break
        else:
            best_algorithm, best_classifier, accuracies = (
                self.ml_classifiers.train_and_evaluate_algorithms(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    test_to_train_map,
                    self.evaluation_metrics,
                )
            )

        self.direction_analyzer.analyze_best_mappings(
            protein_name,
            best_algorithm,
            best_classifier,
            X_train,
            y_train,
            X_test,
            y_test,
            test_to_train_map,
            direction_mapping,
        )

    def print_overall_direction_summary(self):
        """Print overall direction detection summary."""
        self.direction_analyzer.print_overall_direction_summary()

    def reset_direction_stats(self):
        """Reset direction statistics for new analysis."""
        self.direction_analyzer.reset_stats()

    def find_globally_best_parameters(self):
        """Find globally best parameters if using grid search."""
        if self.use_grid_search and hasattr(self, "hyperparameter_optimizer"):
            return self.hyperparameter_optimizer.find_globally_best_parameters()
        return {}
