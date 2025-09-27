import os
import pandas as pd

def convert_excel_to_csv_in_subdirs(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.xlsx'):
                xlsx_path = os.path.join(dirpath, filename)
                csv_filename = os.path.splitext(filename)[0] + '.csv'
                csv_path = os.path.join(dirpath, csv_filename)

                # Only convert if the CSV does not already exist
                if not os.path.exists(csv_path):
                    try:
                        print(f"Converting: {xlsx_path} -> {csv_path}")
                        df = pd.read_excel(xlsx_path)
                        df.to_csv(csv_path, index=False)
                    except Exception as e:
                        print(f"Failed to convert {xlsx_path}: {e}")

# Replace '.' with your desired root directory path if needed
convert_excel_to_csv_in_subdirs('.')

