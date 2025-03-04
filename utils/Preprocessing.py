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

def handle_missing_values_length_index(df_filtered):
    """
    Identifies consecutive missing values (NaNs) in the 'Acc-Z' column 
    and checks if any consecutive sequence exceeds 100 samples.

    Parameters
    ----------
    df_filtered : pandas.DataFrame
        The input DataFrame with a column named 'Acc-Z'.

    Returns
    -------
    missing_info : dict
        A dictionary where:
        - Key is the start index of the missing-value sequence.
        - Value is the length of that missing-value sequence.
    """
    # Dictionary to store consecutive missing info (start_index: count_of_missing)
    missing_info = {}
    current_count = 0
    missing_value_flag = False
    # Will store the first index of a missing-values sequence (None means we're currently not in a missing sequence)
    start_index = None

    # Iterate over each row in df_filtered
    for index, row in df_filtered.iterrows():
        # Check if 'Acc-Z' is NaN
        if np.isnan(row['Acc-Z']):
            # If this is the first missing in a new sequence, record the start index
            if start_index is None:
                start_index = index
                # Start counting from 1 for this missing value
                current_count += 1
            else:
                # If we're already tracking a missing sequence, just increment
                current_count += 1
        else:
            # If 'Acc-Z' is not NaN so that the missing sequence broke
            if start_index is not None:
                # Store start_index -> length_of_missing_sequence in the dictionary
                missing_info[start_index] = current_count
                # Reset to indicate we're no longer in a missing sequence
                start_index = None
                current_count = 0
    # Handle case where the last samples in df_filtered are missing
    # (Because if the loop ends while still in a missing sequence, it's not recorded yet)
    if current_count > 0:
        missing_info[start_index] = current_count
    # Print sequences longer than 100 missing values (twice for demonstration)
    for key, value in missing_info.items():
        if value > 100:
            print("Start Index:", key, "Count:", value)
            missing_value_flag = True
    # If no sequences are longer than 100, print a message
    if not missing_value_flag:
        print("No missing values longer than 100")
    return missing_info