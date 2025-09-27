"""
Hyperparameter optimization module.
Handles grid search and hyperparameter tuning for ML algorithms.
"""

import numpy as np
from itertools import product
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


class HyperparameterOptimizer:
    """Handles hyperparameter optimization using grid search."""

    def __init__(self, param_grids, file_handler):
        self.param_grids = param_grids
        self.file_handler = file_handler
        self.hyperparameter_scores = {
            "SVM Linear": [],
            "SVM RBF": [],
            "Random Forest": [],
        }

    def evaluate_hyperparameters(
        self,
        classifier,
        param_grid,
        X_train,
        y_train,
        X_test,
        y_test,
        test_to_train_map,
        algorithm_name,
        protein_name,
        evaluation_metrics,
    ):
        """
        Evaluate different hyperparameter combinations for a classifier using our mapped accuracy.
        Track results across proteins to find globally best parameters.
        """
        print(f"\n--- Hyperparameter Search for {algorithm_name} on {protein_name} ---")

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))

        best_score = -1
        best_params = None
        best_estimator = None

        print(f"Testing {len(param_combinations)} parameter combinations...")

        for i, param_combination in enumerate(param_combinations):
            params = dict(zip(param_names, param_combination))

            current_classifier = classifier.__class__(**params)

            current_classifier.fit(X_train, y_train)

            y_pred = current_classifier.predict(X_test)

            current_score = evaluation_metrics.calculate_mapped_accuracy(
                y_test, y_pred, test_to_train_map
            )

            self.hyperparameter_scores[algorithm_name].append(
                {
                    "protein": protein_name,
                    "params": params.copy(),
                    "accuracy": current_score,
                }
            )

            if current_score > best_score:
                best_score = current_score
                best_params = params.copy()
                best_estimator = current_classifier

            if (i + 1) % 10 == 0 or (i + 1) == len(param_combinations):
                print(
                    f"Progress: {i + 1}/{len(param_combinations)} combinations tested"
                )

        print(f"Best parameters for {algorithm_name} on {protein_name}: {best_params}")
        print(f"Best accuracy for {algorithm_name} on {protein_name}: {best_score:.4f}")

        return best_estimator, best_params, best_score

    def find_globally_best_parameters(self):
        """
        Analyze hyperparameter performance across all proteins to find the globally best parameters.
        """
        print("\n" + "=" * 60)
        print("ANALYZING BEST HYPERPARAMETERS ACROSS ALL PROTEINS")
        print("=" * 60)

        global_best_params = {}

        for algorithm_name in self.hyperparameter_scores:
            if not self.hyperparameter_scores[algorithm_name]:
                continue

            print(f"\n--- {algorithm_name} ---")

            param_performance = {}
            for result in self.hyperparameter_scores[algorithm_name]:
                param_key = str(sorted(result["params"].items()))
                if param_key not in param_performance:
                    param_performance[param_key] = {
                        "params": result["params"],
                        "accuracies": [],
                        "proteins": [],
                    }
                param_performance[param_key]["accuracies"].append(result["accuracy"])
                param_performance[param_key]["proteins"].append(result["protein"])

            best_avg_accuracy = -1
            best_param_combo = None

            for param_key, performance in param_performance.items():
                avg_accuracy = np.mean(performance["accuracies"])
                std_accuracy = np.std(performance["accuracies"])

                print(f"Params: {performance['params']}")
                print(f"  Average accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
                print(f"  Tested on {len(performance['proteins'])} proteins")

                if avg_accuracy > best_avg_accuracy:
                    best_avg_accuracy = avg_accuracy
                    best_param_combo = performance["params"]

            if best_param_combo:
                global_best_params[algorithm_name] = best_param_combo
                print(f"Best parameters for {algorithm_name}: {best_param_combo}")
                print(f"Average accuracy: {best_avg_accuracy:.4f}")

        if global_best_params:
            self.file_handler.save_best_parameters(
                global_best_params, "best_hyperparameters.json"
            )

        return global_best_params

    def optimize_for_algorithms(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        test_to_train_map,
        protein_name,
        evaluation_metrics,
    ):
        """Optimize hyperparameters for all algorithms."""
        algorithms_for_grid_search = [
            ("SVM Linear", svm.SVC(), "SVM Linear"),
            ("SVM RBF", svm.SVC(), "SVM RBF"),
            (
                "Random Forest",
                RandomForestClassifier(random_state=42),
                "Random Forest",
            ),
        ]

        optimized_algorithms = []
        best_params_dict = {}

        for name, base_classifier, param_key in algorithms_for_grid_search:
            if param_key in self.param_grids:
                best_classifier, best_params, best_score = (
                    self.evaluate_hyperparameters(
                        base_classifier,
                        self.param_grids[param_key],
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        test_to_train_map,
                        name,
                        protein_name,
                        evaluation_metrics,
                    )
                )
                optimized_algorithms.append((name, best_classifier))
                best_params_dict[name] = best_params

        return optimized_algorithms, best_params_dict
