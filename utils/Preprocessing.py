import pandas as pd  
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates



def combine_activities(df_one, df_two, output_path):
    """
    Combine two DataFrames based on specified curb_activity values and save the result.
    
    Parameters:
    -----------
    df_one : pandas.DataFrame
        First DataFrame containing curb_activity
    df_two : pandas.DataFrame
        Second DataFrame containing curb_activity
    output_path : str, default='../data/handlebar_processed.csv'
        Path where the combined DataFrame will be saved
    Returns:
    --------
    pandas.DataFrame
        Combined and sorted DataFrame
    """
    # Select data from each DataFrame based on curb_activity values
    activity_one = df_one[df_one['curb_activity'] == 1.0]
    activity_two = df_two[df_two['curb_activity'] == 2.0]
    # Combine the selected data
    df_combined = pd.concat([activity_one, activity_two], ignore_index=True)
    # Sort by timestamp and reset index
    df_combined = df_combined.sort_values('NTP').reset_index(drop=True)
    # Save the combined DataFrame
    df_combined.to_csv(output_path, index=False)
    
    return df_combined

def handle_missing_values_length(df_filtered):
    # Handle missing values length
    missing_info = {}  # List to store (start_index, current_count)
    current_count = 0
    start_index = None  # To store the start time of missing values
    for index, row in df_filtered.iterrows():
        if np.isnan(row['Acc-Z']):
            if start_index is None:
                start_index = index
                current_count += 1
            if start_index is not None:
                current_count += 1
        # If the value is not NaN, continue sequence broke
        else:
            if start_index is not None:
                missing_info[start_index] = current_count
                start_index = None
                current_count = 0
    # Handle case where the last segment has missing values
    if current_count > 0:
        missing_info[start_index] = current_count

    for key, value in missing_info.items():
        print(key, value)
        if value > 100:
            print("Start Index:", key, "Count:", value)

    print(missing_info)
    return missing_info