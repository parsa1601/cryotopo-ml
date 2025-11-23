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
"1NG6",
"2XB5",
"2XVV",
"3ACW",
"3FIN",
"3HJL",
"3ODS",
"5KBU",
"1BZ4",
"1FLP",
"1ICX",
"2Y4Z",
"3C91",
"4CHV",
"5I1M",
"5O8O",
"5UZB",
"6EM3",
"6F36",
"1A7D",
"1OZ9",
"1YD0",
"4OXW",
"4R9A",
"4UE4",
"4YOK",
"5M50"
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
    # "6UXW",
]

DEFAULT_REPORT_FILE = "direction_analysis_report.txt"
DEFAULT_BEST_PARAMS_FILE = "best_hyperparameters.json"
