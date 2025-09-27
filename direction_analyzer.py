"""
Direction analysis module using Dynamic Time Warping.
Handles direction detection and analysis of protein structures.
"""

import numpy as np


class DirectionAnalyzer:
    """Handles direction detection using DTW and analysis of results."""

    def __init__(self, file_handler):
        self.file_handler = file_handler
        self.overall_direction_stats = {
            "total_directions": 0,
            "correct_directions": 0,
            "protein_results": [],
        }

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
        forward_distance = self.dtw_distance(model_coords, stick_coords)

        backward_coords = np.flipud(stick_coords)
        backward_distance = self.dtw_distance(model_coords, backward_coords)

        return 1 if forward_distance <= backward_distance else -1

    def analyze_best_mappings(
        self,
        protein_name,
        best_algorithm,
        best_classifier,
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
        self.file_handler.print_and_save(
            f"\n--- Direction Analysis for {protein_name} using {best_algorithm} ---"
        )

        y_pred = best_classifier.predict(X_test)

        correct_directions = 0
        total_directions = 0
        direction_results = []

        unique_sticks = np.unique(y_test)
        for stick_label in unique_sticks:
            stick_indices = np.where(y_test == stick_label)[0]
            if len(stick_indices) > 0:
                predicted_train_label = y_pred[stick_indices[0]]

                actual_train_label = test_to_train_map.get(stick_label)

                if (
                    actual_train_label is not None
                    and predicted_train_label == actual_train_label
                ):
                    model_coords = X_train[y_train == actual_train_label]
                    stick_coords = X_test[y_test == stick_label]

                    detected_direction = self.determine_direction_with_dtw(
                        model_coords, stick_coords
                    )

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

                        self.file_handler.print_and_save(
                            f"Stick {stick_label} -> Model {actual_train_label}: Detected={detected_direction}, Actual={actual_direction}, Correct={is_correct}"
                        )

        if total_directions > 0:
            direction_accuracy = (correct_directions / total_directions) * 100
            self.file_handler.print_and_save("\n--- Direction Detection Results ---")
            self.file_handler.print_and_save(
                f"Total directions analyzed: {total_directions}"
            )
            self.file_handler.print_and_save(
                f"Correctly detected directions: {correct_directions}"
            )
            self.file_handler.print_and_save(
                f"Direction detection accuracy: {direction_accuracy:.2f}%"
            )

            self.update_overall_direction_stats(
                protein_name, correct_directions, total_directions, direction_accuracy
            )
        else:
            self.file_handler.print_and_save(
                "\nNo direction information available for analysis."
            )
            self.update_overall_direction_stats(protein_name, 0, 0, 0.0)

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
            self.file_handler.print_and_save("\n" + "=" * 60)
            self.file_handler.print_and_save("OVERALL DIRECTION DETECTION SUMMARY")
            self.file_handler.print_and_save("=" * 60)
            self.file_handler.print_and_save(
                f"Total proteins analyzed: {len(stats['protein_results'])}"
            )
            self.file_handler.print_and_save(
                f"Total directions analyzed: {stats['total_directions']}"
            )
            self.file_handler.print_and_save(
                f"Total correct directions: {stats['correct_directions']}"
            )
            self.file_handler.print_and_save(
                f"Overall direction detection accuracy: {overall_accuracy:.2f}%"
            )
            self.file_handler.print_and_save("\nPer-protein results:")
            self.file_handler.print_and_save("-" * 40)
            for result in stats["protein_results"]:
                self.file_handler.print_and_save(
                    f"{result['protein']}: {result['correct']}/{result['total']} ({result['accuracy']:.2f}%)"
                )
            self.file_handler.print_and_save("=" * 60)
        else:
            self.file_handler.print_and_save(
                "\nNo direction detection data available for summary."
            )

    def reset_stats(self):
        """Reset the direction statistics for a new analysis."""
        self.overall_direction_stats = {
            "total_directions": 0,
            "correct_directions": 0,
            "protein_results": [],
        }
