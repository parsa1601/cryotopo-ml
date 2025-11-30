"""
Machine learning classifiers module.
Contains the ML algorithms and training logic for protein structure analysis.
"""

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from protein_visualization import ProteinVisualizer
from collections import defaultdict
import time


class MLClassifiers:
    """Handles machine learning classifiers and training operations."""

    def __init__(self, best_params=None):
        self.best_params = best_params if best_params else {}
        self.visualizer = ProteinVisualizer()
        self._initialize_classifiers()
        self.performance_report = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )

    def _initialize_classifiers(self):
        """Initialize classifiers with best parameters if available."""
        self.classifiers = {}
        
        if "SVM Linear" in self.best_params:
            self.classifiers["SVM Linear"] = svm.SVC(**self.best_params["SVM Linear"])
        else:
            self.classifiers["SVM Linear"] = svm.SVC(kernel="linear")

        if "SVM RBF" in self.best_params:
            self.classifiers["SVM RBF"] = svm.SVC(**self.best_params["SVM RBF"])
        else:
            self.classifiers["SVM RBF"] = svm.SVC(kernel="rbf")

        if "Random Forest" in self.best_params:
            self.classifiers["Random Forest"] = RandomForestClassifier(
                random_state=42, **self.best_params["Random Forest"]
            )
        else:
            self.classifiers["Random Forest"] = RandomForestClassifier(
                n_estimators=100, random_state=42
            )

        self.classifiers["Voronoi (1N KNN)"] = KNeighborsClassifier(n_neighbors=1)

    def train_and_evaluate_algorithms(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        test_to_train_map,
        evaluation_metrics,
        structure_type,
        protein_name,
    ):
        """Train and evaluate all algorithms, store results."""        
        for name, classifier in self.classifiers.items():
            print(f"\n--- {name} Results ---")

            if name in self.best_params:
                print(f"Using optimized parameters: {self.best_params[name]}")

            start_train = time.time()
            classifier.fit(X_train, y_train)
            train_time = time.time() - start_train

            start_test = time.time()
            y_pred = classifier.predict(X_test)
            test_time = time.time() - start_test

            confusion_matrix, metrics = evaluation_metrics.calculate_custom_metrics(
                y_test, y_pred, y_train, test_to_train_map
            )
            
            # Add runtime metrics
            metrics['train_time'] = train_time
            metrics['test_time'] = test_time

            self.performance_report[protein_name][structure_type][name][
                "confusion_matrix_detailed"
            ] = confusion_matrix
            self.performance_report[protein_name][structure_type][name].update(
                metrics
            )
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1-measure: {metrics['f1_measure']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"Train Time: {train_time:.4f}s")
            print(f"Test Time: {test_time:.4f}s")

        print("-" * 50)

    def get_best_algorithm_by_f1_measure(self):
        """Calculate the best algorithm based on average F1-measure across all proteins."""
        algorithm_f1_scores = defaultdict(list)
        
        # Collect all F1 scores for each algorithm across all proteins and structure types
        for protein, structures in self.performance_report.items():
            for structure_type, algorithms in structures.items():
                for algorithm, metrics in algorithms.items():
                    if 'f1_measure' in metrics:
                        algorithm_f1_scores[algorithm].append(metrics['f1_measure'])
        
        # Calculate average F1 score for each algorithm
        algorithm_averages = {}
        for algorithm, f1_scores in algorithm_f1_scores.items():
            if f1_scores:  # Make sure we have scores
                algorithm_averages[algorithm] = sum(f1_scores) / len(f1_scores)
        
        # Find the algorithm with the highest average F1-measure
        if algorithm_averages:
            best_algorithm = max(algorithm_averages, key=algorithm_averages.get)
            best_f1_score = algorithm_averages[best_algorithm]
            
            print(f"\n{'='*60}")
            print("OVERALL BEST ALGORITHM SELECTION")
            print(f"{'='*60}")
            print("Average F1-measure by algorithm:")
            for algorithm in sorted(algorithm_averages.keys()):
                print(f"  {algorithm:<20}: {algorithm_averages[algorithm]:.4f}")
            print(f"\nBest Algorithm Overall: {best_algorithm} (Avg F1-measure: {best_f1_score:.4f})")
            print(f"{'='*60}")
            
            return best_algorithm, algorithm_averages
        else:
            print("No F1-measure data available for algorithm selection.")
            return None, {}
