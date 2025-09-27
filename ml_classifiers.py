"""
Machine learning classifiers module.
Contains the ML algorithms and training logic for protein structure analysis.
"""

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from protein_visualization import ProteinVisualizer


class MLClassifiers:
    """Handles machine learning classifiers and training operations."""

    def __init__(self, best_params=None):
        self.best_params = best_params if best_params else {}
        self.visualizer = ProteinVisualizer()
        self._initialize_classifiers()

    def _initialize_classifiers(self):
        """Initialize classifiers with best parameters if available."""
        if "SVM Linear" in self.best_params:
            self.svm_linear = svm.SVC(**self.best_params["SVM Linear"])
        else:
            self.svm_linear = svm.SVC(kernel="linear")

        if "SVM RBF" in self.best_params:
            self.svm_rbf = svm.SVC(**self.best_params["SVM RBF"])
        else:
            self.svm_rbf = svm.SVC(kernel="rbf")

        if "Random Forest" in self.best_params:
            self.random_forest = RandomForestClassifier(
                random_state=42, **self.best_params["Random Forest"]
            )
        else:
            self.random_forest = RandomForestClassifier(
                n_estimators=100, random_state=42
            )

        self.knn = KNeighborsClassifier(n_neighbors=1)

    def get_algorithms(self):
        """Return list of algorithm tuples (name, classifier)."""
        return [
            ("SVM Linear", self.svm_linear),
            ("SVM RBF", self.svm_rbf),
            ("Random Forest", self.random_forest),
            ("Voronoi (1N KNN)", self.knn),
        ]

    def train_and_evaluate_algorithms(
        self, X_train, y_train, X_test, y_test, test_to_train_map, evaluation_metrics
    ):
        """Train and evaluate all algorithms, return results."""
        algorithms = self.get_algorithms()
        accuracies = {}

        for name, classifier in algorithms:
            print(f"\n--- {name} Results ---")

            if name in self.best_params:
                print(f"Using optimized parameters: {self.best_params[name]}")

            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            accuracy = evaluation_metrics.calculate_mapped_accuracy(
                y_test, y_pred, test_to_train_map
            )
            accuracies[name] = accuracy
            print(f"Accuracy: {accuracy:.4f}")

        best_algorithm = max(accuracies, key=accuracies.get)
        print(
            f"\nBest Algorithm: {best_algorithm} (Accuracy: {accuracies[best_algorithm]:.4f})"
        )

        if best_algorithm in self.best_params:
            print(
                f"Best parameters for {best_algorithm}: {self.best_params[best_algorithm]}"
            )

        print(f"Best Algorithm: {best_algorithm} ({accuracies[best_algorithm]:.4f})")
        print("-" * 50)

        best_classifier = None
        for name, classifier in algorithms:
            if name == best_algorithm:
                best_classifier = classifier
                break

        return best_algorithm, best_classifier, accuracies
