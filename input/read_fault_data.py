import pandas as pd

def read_fault_data(csv_file):
    """
    CSV file containing data from the breakage layer.
    """
    fault_df = pd.read_csv(csv_file)
    # Extract all unique fault names
    fault_names = fault_df['Level'].unique().tolist()

    return fault_df, fault_names
