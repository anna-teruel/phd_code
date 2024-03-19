import pandas as pd
import os

input_directory = '/Users/annateruel/Desktop/QUINT/output/abeta_extra/'
output_directory = '/Users/annateruel/Desktop/QUINT/output/abeta_extra/'

group_data = {}

for file_name in os.listdir(input_directory):
    file_path = os.path.join(input_directory, file_name)
    if file_name.endswith(".csv"):
        prefix = file_name.split("-")[0]
        df = pd.read_csv(file_path)
        
        # Check if 'Region name' exists in the dataframe's columns
        if 'Region name' in df.columns:
            transposed_df = df.set_index('Region name').T
            if prefix not in group_data:
                group_data[prefix] = transposed_df
            else:
                group_data[prefix] = group_data[prefix].add(transposed_df, fill_value=0)
        else:
            print(f"Warning: 'Region name' not found in {file_path}")

for prefix, df in group_data.items():
    output_file_path = os.path.join(output_directory, f"{prefix}_combined.csv")
    df.to_csv(output_file_path)
