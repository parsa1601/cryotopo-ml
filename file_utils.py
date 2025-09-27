"""
File utilities module for handling I/O operations.
Contains functions for saving and loading parameters, and report generation.
"""

import json
import os


class FileHandler:
    """Handles file I/O operations for the protein analysis system."""

    def __init__(self, report_file="direction_analysis_report.txt"):
        self.report_file = report_file

    def print_and_save(self, message):
        """
        Print message to console and save to report file.
        This method should ONLY be used for direction detection analysis reports.
        """
        print(message)
        with open(self.report_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def reset_report_file(self):
        """Reset the report file by clearing its contents."""
        try:
            with open(self.report_file, "w", encoding="utf-8") as f:
                f.write("")
            print(f"Report file {self.report_file} has been reset")
        except Exception as e:
            print(f"Error resetting report file: {e}")

    def save_best_parameters(self, best_params, best_params_file):
        """Save the best hyperparameters to a JSON file"""
        try:
            with open(best_params_file, "w") as f:
                json.dump(best_params, f, indent=4)
            print(f"Best hyperparameters saved to {best_params_file}")
        except Exception as e:
            print(f"Error saving best parameters: {e}")

    def load_best_parameters(self, best_params_file):
        """Load the best hyperparameters from JSON file"""
        try:
            if os.path.exists(best_params_file):
                with open(best_params_file, "r") as f:
                    params = json.load(f)
                print(f"Loaded best hyperparameters from {best_params_file}")
                return params
            else:
                print(f"No saved parameters file found at {best_params_file}")
                return {}
        except Exception as e:
            print(f"Error loading best parameters: {e}")
            return {}
