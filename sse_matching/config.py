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
    "1A7D",
    "1BJ7",
    "1BZ4",
    "1FLP",
    "1HG5",
    "1HBE",
    "1HZ4",
    "1ICX",
    "1LWB",
    "1NG6",
    "1OZ9",
    "1P5X",
    "1XQO",
    "1YD0",
    "1Z1L",
    "2OVJ",
    "2XB5",
    "2XVV",
    "2Y4Z",
    "3ACW",
    "3C91",
    "3FIN",
    "3HJL",
    "3IEE",
    "3IXV",
    "3LTJ",
    "3ODS",
    "4CHV",
    "4OXW",
    "4R9A",
    "4UE4",
    "4YOK",
    "5I1M",
    "5KBU",
    "5M50",
    "5O8O",
    "5UZB",
    "6EM3",
    "6F36",
    "6UXW",
]

STRAND_PROTEIN_LIST = [
    "1ICX",
    "1YD0",
    "3C91",
    "4OXW",
    "4YOK",
    "5M50",
    "6EM3",
    "1OZ9",
    "2Y4Z",
    "4CHV",
    "4R9A",
    "5KBU",
    "5O8O",
    "6UXW",
]

DEFAULT_REPORT_FILE = "artifacts/direction_analysis_report.txt"
DEFAULT_BEST_PARAMS_FILE = "artifacts/best_hyperparameters.json"
USE_BRESENHAM_FOR_LPTD = False
