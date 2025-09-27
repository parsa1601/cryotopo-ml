"""
Main entry point for protein structure analysis.
Runs the complete analysis pipeline for both helix and strand structures.
"""

import os
import sys
import warnings

from protein_trainer import ProteinTrainer
from config import CSV_DATASET, HELIX_PROTEIN_LIST, STRAND_PROTEIN_LIST

sys.path.append(f"{os.path.dirname(os.getcwd())}")
warnings.filterwarnings("ignore")


def main():
    """
    Main function to run the protein structure analysis.
    The first list contains proteins with records of Helix
    The second list contains proteins with records of Strand
    """

    trainer = ProteinTrainer(csv_path=CSV_DATASET, use_grid_search=False)

    print("\n\n\n\n*************************HELIX RESULTS:**********************")
    trainer.file_handler.print_and_save(
        "\n\n\n\n*************************HELIX RESULTS:**********************"
    )
    trainer.train_with_all_algorithms(HELIX_PROTEIN_LIST, "Helix")
    trainer.print_overall_direction_summary()

    print("\n\n\n\n*************************STRAND RESULTS:**********************")
    trainer.file_handler.print_and_save(
        "\n\n\n\n*************************STRAND RESULTS:**********************"
    )

    trainer.reset_direction_stats()
    trainer.train_with_all_algorithms(STRAND_PROTEIN_LIST, "Strand")
    trainer.print_overall_direction_summary()

    print(
        f"\nDirection analysis report has been saved to: {trainer.file_handler.report_file}"
    )


if __name__ == "__main__":
    main()
