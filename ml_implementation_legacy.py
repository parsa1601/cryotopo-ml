"""
Compatibility wrapper for the original ProteinAssignmentUsingMultipleML class.
This maintains the original interface while using the refactored modules.
"""

import warnings
from protein_trainer import ProteinTrainer

warnings.filterwarnings("ignore")


class ProteinAssignmentUsingMultipleML:
    """
    Compatibility wrapper that maintains the original class interface
    while delegating to the refactored ProteinTrainer.
    """

    def __init__(
        self,
        report_file="direction_analysis_report.txt",
        use_grid_search=False,
        best_params_file="best_hyperparameters.json",
    ):
        """
        Initializing 3 classifiers to use all of them!
        """
        self.trainer = ProteinTrainer(
            csv_path="Archive/",
            report_file=report_file,
            use_grid_search=use_grid_search,
            best_params_file=best_params_file,
        )

        self.svm_linear = self.trainer.ml_classifiers.svm_linear
        self.svm_rbf = self.trainer.ml_classifiers.svm_rbf
        self.random_forest = self.trainer.ml_classifiers.random_forest
        self.knn = self.trainer.ml_classifiers.knn
        self.visualizer = self.trainer.ml_classifiers.visualizer
        self.report_file = report_file
        self.use_grid_search = use_grid_search
        self.best_params_file = best_params_file
        self.overall_direction_stats = (
            self.trainer.direction_analyzer.overall_direction_stats
        )

    def print_and_save(self, message):
        """
        Print message to console and save to report file.
        This method should ONLY be used for direction detection analysis reports.
        """
        self.trainer.file_handler.print_and_save(message)

    def train_with_all_algorithms(self, proteins_list, csv_path, mode: str):
        """
        This function trains and tests the proteins list
        it first decides to train the model on helix records or strand records. When creating
        x and y records, it trains the 3 models and tests the classes for stick records and prints the results.
        """
        self.trainer.train_with_all_algorithms(proteins_list, mode)

    def print_overall_direction_summary(self):
        """
        Print a summary of direction detection accuracy across all analyzed proteins.
        """
        self.trainer.print_overall_direction_summary()

    def find_globally_best_parameters(self):
        """
        Analyze hyperparameter performance across all proteins to find the globally best parameters.
        """
        return self.trainer.find_globally_best_parameters()

    def save_best_parameters(self, best_params):
        """Save the best hyperparameters to a JSON file"""
        self.trainer.file_handler.save_best_parameters(
            best_params, self.best_params_file
        )

    def load_best_parameters(self):
        """Load the best hyperparameters from JSON file"""
        return self.trainer.file_handler.load_best_parameters(self.best_params_file)
