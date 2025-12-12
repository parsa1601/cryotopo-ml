"""
Configuration module for protein structure analysis.
Contains all configuration constants and settings.
"""

CSV_DATASET = "Archive/"

PARAM_GRIDS = {
    "SVM Linear": {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto"],
        "kernel": ["linear"],
    },
    "SVM RBF": {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
        "kernel": ["rbf"],
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
}

HELIX_PROTEIN_LIST = [
    "1FLP",
    "1NG6",
    "2XB5",
    "1BZ4",
    "3ACW",
    "1A7D",
    "3ODS",
    "3HJL",
    "1ICX",
    "1OZ9",
    "4OXW",
    "1YD0",
    "2Y4Z",
    "4YOK",
    "4R9A",
    "3FIN",
    "4CHV",
    "5I1M",
    "6F36",
    "6EM3",
    "4UE4",
    "5UZB",
    "3C91",
    "5O8O",
    "5M50",
    "5KBU",
    "6UXW",
]

STRAND_PROTEIN_LIST = [
    "1ICX",
    "1OZ9",
    "4OXW",
    "1YD0",
    "2Y4Z",
    "4YOK",
    "4R9A",
    "4CHV",
    "6EM3",
    "3C91",
    "5O8O",
    "5M50",
    "5KBU",
]

DEFAULT_REPORT_FILE = "direction_analysis_report.txt"
DEFAULT_BEST_PARAMS_FILE = "best_hyperparameters.json"
