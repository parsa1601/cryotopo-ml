"""
Data loading module for protein structure analysis.
Handles loading and preprocessing of protein data from CSV files.
"""

import os
import pandas as pd
import numpy as np


class DataLoader:
    """Handles loading and preprocessing of protein data."""

    def __init__(self, csv_path="Archive/"):
        self.csv_path = csv_path

    def generate_protein_helix_stick(self, protein_name: str):
        """
        This function reads the csv files for helices. To test the code, the proteins should be in the csv_path with
        the following naming:
        PROTEIN NAME IN ALL CAPITAL_Helices.csv
        PROTEIN NAME IN ALL CAPITAL_Stick.csv
        """
        helix_records = f"{self.csv_path}/{protein_name}/{protein_name}_Helices.csv"
        stick_records = f"{self.csv_path}/{protein_name}/{protein_name}_Sticks.csv"

        if not os.path.exists(stick_records):
            stick_records = f"{self.csv_path}/{protein_name}/{protein_name}_Stick.csv"

        helix_df = pd.read_csv(helix_records, header=None)
        helices_datapoints = helix_df.iloc[:, :3].to_numpy()
        classes = helix_df.iloc[:, 3].to_numpy().astype(int)
        k_helices = len(np.unique(classes))

        stick_df = pd.read_csv(stick_records, header=None)
        cryo_datapoints = stick_df.iloc[:, :3].to_numpy()
        sticks = stick_df.iloc[:, 3].to_numpy().astype(int)
        k_stick = len(np.unique(sticks))

        return helices_datapoints, cryo_datapoints, classes, sticks, k_helices, k_stick

    def generate_protein_strand_stick(self, protein_name: str):
        """
        This function reads the csv files for strands. To test the code, the proteins should be in the csv_path with
        the following naming:
        these proteins should be in a folder named STRANDS/
        PROTEIN NAME IN ALL CAPITAL_Strands.csv
        PROTEIN NAME IN ALL CAPITAL_Sticks_Strands.csv
        """
        strands_records = (
            f"{self.csv_path}STRANDS/{protein_name}/Sheet/{protein_name}_Strands.csv"
        )
        stick_records = f"{self.csv_path}STRANDS/{protein_name}/Sheet/{protein_name}_Sticks_Strands.csv"

        strand_df = pd.read_csv(strands_records, header=None)
        strands_datapoints = strand_df.iloc[:, :3].to_numpy()
        classes = strand_df.iloc[:, 3].to_numpy().astype(int)
        k_strands = len(np.unique(classes))

        stick_df = pd.read_csv(stick_records, header=None)
        cryo_datapoints = stick_df.iloc[:, :3].to_numpy()
        sticks = stick_df.iloc[:, 3].to_numpy().astype(int)
        k_stick = len(np.unique(sticks))

        return strands_datapoints, cryo_datapoints, classes, sticks, k_strands, k_stick

    def read_mapping_topology(self, protein_name, mode):
        """
        reading topology csv file based on the protein name to create the mapping needed for the accuracy calculate.
        Since the path for strands and helices is different, it checks first!
        Returns both mapping and direction information.
        """
        if mode == "Helix":
            topology_record = (
                f"{self.csv_path}/{protein_name}/{protein_name}_Topology.csv"
            )
            topology_df = pd.read_csv(topology_record, header=None)
        if mode == "Strand":
            topology_record = f"{self.csv_path}STRANDS/{protein_name}/Sheet/{protein_name}_Topology.csv"
            topology_df = pd.read_csv(topology_record, header=None)

        mapping = topology_df.iloc[:, :2].to_numpy()

        direction_mapping = {}
        for _, row in topology_df.iterrows():
            if len(row) >= 3 and pd.notna(row.iloc[2]):
                test_label = int(row.iloc[1])
                actual_direction = int(row.iloc[2])
                direction_mapping[test_label] = actual_direction

        return mapping, direction_mapping

    @staticmethod
    def remap_labels(labels, label_mapping):
        return np.array([label_mapping.get(label, label) for label in labels])

    def preprocess_labels(self, labels: list):
        pass
