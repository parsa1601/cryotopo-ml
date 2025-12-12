"""
Evaluation metrics module for calculating accuracy and performance.
Handles mapping-based accuracy calculations for protein structure analysis.
"""
import numpy as np
from collections import Counter

class EvaluationMetrics:
    """Handles calculation of evaluation metrics for protein structure analysis."""

    @staticmethod
    def group_datapoints(y_test, y_pred, mapping):
        """
        Groups datapoints by their SSE (Secondary Structure Element) based on the mapping.
        Aggregates predictions using majority voting.
        """
        test_to_train_map = {
            test_label: train_label for train_label, test_label in mapping.items()
        }

        sse_groups = {}
        for i, stick_id in enumerate(y_test):
            sse_id = test_to_train_map.get(stick_id)
            if sse_id is not None:
                if sse_id not in sse_groups:
                    sse_groups[sse_id] = []
                sse_groups[sse_id].append(y_pred[i])
        
        new_y_test = []
        new_y_pred = []
        
        for sse_id, predictions in sse_groups.items():
            new_y_test.append(sse_id)
            # Majority vote for prediction
            counts = Counter(predictions)
            most_common = counts.most_common(1)[0][0]
            new_y_pred.append(most_common)
        
        return np.array(new_y_test), np.array(new_y_pred)
    
    @staticmethod
    def calculate_custom_metrics(y_test, y_pred, mapping):
        """
        Calculates custom TP, TN, FP, FN counts and a confusion matrix.

        Args:
             y_test: Array of stick IDs (ground truth test labels)
             y_pred: Array of predicted SSE IDs (train labels)
             mapping: Dictionary maps SSE IDs to their ground truth stick IDs

        Returns:
            tuple: A tuple containing:
                - confusion_matrix (dict): Contains 'tp', 'tn', 'fp', 'fn' counts
                - metrics (dict): Contains 'accuracy', 'precision', 'recall', 'f1_measure', 'mismatch_rate' (all as percentages except accuracy which is 0-1)

        This function uses a specific, two-part logic to derive the metrics,
        separating the evaluation of 'matched' predictions from 'unmatched' labels.

        Calculation Logic:
        ------------------
        1.  **True Positives (TP)**:
            - A prediction is counted as a TP if its mapping value aligns with the y_test.
            - Example: `tp_count += 1` if `mapping[y_pred[i]] == y_test[i]`.

        4.  **False Negatives (FN)**:
            - A prediction is counted as a FP if its mapping value is not 0 and does not align with the y_test.

        3.  **True Negatives (TN)**:
            - If a SSE ID were not present in mapping keys and not in predictions, it is considered as unmatched;
            - Then If it's value in mapping were 0, it's considered a "correct rejection."

        2.  **False Positives (FP)**:
            - If a SSE ID were not present in mapping keys and not in predictions, it is considered as unmatched;
            - Then If it's value in mapping were not 0, it's considered a "wrong rejection."
        """
        # Group datapoints by SSE
        y_test, y_pred = EvaluationMetrics.group_datapoints(
            y_test, y_pred, mapping
        )

        # Helper logic to find matched and unmatched labels
        train_labels = set(mapping.keys())
        pred_labels = set(np.unique(y_pred))
        unmatched = sorted(list(train_labels.symmetric_difference(pred_labels)))

        # Part 1: Calculate TP and FP based on predictions
        tp_count = 0
        fn_count = 0
        for i in range(len(y_pred)):
            pred_val = y_pred[i]
            true_val = y_test[i]
            if true_val == pred_val:
                tp_count += 1
            else:
                fn_count += 1

        # Part 2: Calculate TN and FN based on unmatched labels
        tn_count = 0
        fp_count = 0
        for unmatched_label in unmatched:
            if unmatched_label not in mapping or mapping.get(unmatched_label) == 0:
                tn_count += 1
            else:
                fp_count += 1

        precision = (tp_count / (tp_count + fp_count)) * 100 if (tp_count + fp_count) > 0 else 0
        recall = (tp_count / (tp_count + fn_count)) * 100 if (tp_count + fn_count) > 0 else 0
        f1_measure = ((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
        accuracy = (tp_count + tn_count) / (tp_count + tn_count + fp_count + fn_count) if (tp_count + tn_count + fp_count + fn_count) > 0 else 0

        # Mismatch Rate = (FP + FN)/(Matched pairs + Unmatched pairs) * 100
        matched_pairs = tp_count + tn_count
        unmatched_pairs = fp_count + fn_count
        mismatch_rate = ((fp_count + fn_count) / (matched_pairs + unmatched_pairs)) * 100 if (matched_pairs + unmatched_pairs) > 0 else 0

        
        confusion_matrix = {
            'tp': tp_count,
            'tn': tn_count,
            'fp': fp_count,
            'fn': fn_count,
        }
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_measure': f1_measure,
            'mismatch_rate': mismatch_rate,
        }

        return confusion_matrix, metrics
