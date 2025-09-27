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
    "1HG5",
    "1LWB",
    "1P5X",
    "1Z1L",
    "2XVV",
    "3C91",
    "3HJL",
    "3LTJ",
    "4OXW",
    "5M50",
    "6EM3",
    "1BZ4",
    "1HZ4",
    "1NG6",
    "1XQO",
    "2OVJ",
    "2Y4Z",
    "3FIN",
    "3IEE",
    "3ODS",
    "5I1M",
    "5O8O",
    "6F36",
    "1FLP",
    "1ICX",
    "1OZ9",
    "1YD0",
    "2XB5",
    "3ACW",
    "3HBE",
    "3IXV",
    "4CHV",
    "4UE4",
    "5KBU",
    "5UZB",
    "6UXW",
]

STRAND_PROTEIN_LIST = [
    "1ICX",
    "1OZ9",
    "1YD0",
    "2Y4Z",
    "3C91",
    "4CHV",
    "4OXW",
    "4R9A",
    "4YOK",
    "5KBU",
    "5M50",
    "5O8O",
    "6EM3",
    "6UXW",
]

DEFAULT_REPORT_FILE = "direction_analysis_report.txt"
DEFAULT_BEST_PARAMS_FILE = "best_hyperparameters.json"
